import traceback
from pathlib import Path
import modal
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Literal, TypeAlias, Optional

NGINX_CONF_DIR = Path("/root/nginx")

image = (
    modal.Image.debian_slim()
    .pip_install("PyGithub~=2.6.1")
    .add_local_dir(Path(__file__).resolve().parent / "nginx", NGINX_CONF_DIR)
)

with image.imports():
    from github import Github

redirector_image = modal.Image.debian_slim().pip_install("fastapi[standard]==0.139.0")

app = modal.App(
    name="training-gym-previews",
    image=image,
)

vol = modal.Volume.from_name("preview-artifacts", create_if_missing=True)
deployments = modal.Dict.from_name("training-gym-deployments", create_if_missing=True)

MOUNT_POINT = Path("/vol")
SANDBOX_APP_NAME = "training-gym-preview-sandboxes"
SANDBOX_PORT = 80
SANDBOX_TIMEOUT = timedelta(hours=24)


def get_mounted_artifact_path(name):
    return MOUNT_POINT / name


PreviewType: TypeAlias = Literal["dashboard", "docs"]


@dataclass
class PreviewDeployment:
    type: PreviewType
    artifact: str
    sandbox_id: Optional[str] = None
    url: Optional[str] = None
    expiration: datetime | None = None

    def deploy(self):
        print(f"Deploying {self.type} preview from {self.artifact}")

        path = get_mounted_artifact_path(self.artifact)
        image = (
            modal.Image.from_registry("nginx")
            .run_commands("rm /etc/nginx/conf.d/default.conf")
            .add_local_file(path, "/root/artifact.tar.gz", copy=True)
            .run_commands(
                "tar -xzf /root/artifact.tar.gz -C /usr/share/nginx/html",
                "rm /root/artifact.tar.gz",
            )
        )

        remote_conf = "/etc/nginx/conf.d/preview.conf"

        if self.type == "dashboard":
            image = image.add_local_file(NGINX_CONF_DIR / "dashboard.conf", remote_conf)

        if self.type == "docs":
            image = image.add_local_file(NGINX_CONF_DIR / "docs.conf", remote_conf)

        sb_app = modal.App.lookup(SANDBOX_APP_NAME, create_if_missing=True)
        self.expiration = datetime.now() + SANDBOX_TIMEOUT

        sb = modal.Sandbox.create(
            "nginx",
            "-g",
            "daemon off;",
            app=sb_app,
            image=image,
            encrypted_ports=[SANDBOX_PORT],
            idle_timeout=int(SANDBOX_TIMEOUT.total_seconds()),
            timeout=int(SANDBOX_TIMEOUT.total_seconds()),
        )

        print(f"Created sandbox {sb.object_id}")

        tunnels = sb.tunnels()
        tunnel = tunnels[SANDBOX_PORT]
        self.sandbox_id = sb.object_id
        self.url = tunnel.url
        sb.detach()

        print(f"Deployed {self.type} preview to {self.url} ({self.sandbox_id})")

    def terminate(self):
        print(f"Terminating {self.type} deploy for {self.artifact} ({self.sandbox_id})")

        sb = modal.Sandbox.from_id(self.sandbox_id)
        sb.terminate(wait=False)

        print(f"Requested termination of sandbox {self.sandbox_id}")

        self.sandbox_id = None
        self.url = None
        self.expiration = None

    def cleanup_artifact(self):
        path = get_mounted_artifact_path(self.artifact)
        path.unlink(missing_ok=True)
        print(f"Removed {path}")


@app.function(volumes={MOUNT_POINT: vol})
def deploy_preview(pr_number: int, type: PreviewType, artifact: str):
    print(f"Deploying {type} preview for #{pr_number}")
    key = (pr_number, type)

    vol.reload()

    try:
        deployment_dict = deployments.get((pr_number, type), None)
        if deployment_dict is not None:
            deployment = PreviewDeployment(**deployment_dict)
            deployment.terminate()
            deployment.cleanup_artifact()

        deployment = PreviewDeployment(type=type, artifact=artifact)
        deployment.deploy()
    finally:
        deployments[key] = asdict(deployment)


def needs_refresh(preview: PreviewDeployment):
    if preview.expiration is not None:
        if datetime.now() + timedelta(hours=1) >= preview.expiration:
            return True
    return False


def is_expired(preview: PreviewDeployment):
    if preview.expiration is not None:
        return datetime.now() >= preview.expiration
    return False


@app.function(schedule=modal.Period(hours=1), volumes={MOUNT_POINT: vol})
def refresh_sandboxes():
    from github.GithubException import UnknownObjectException

    print("Refreshing sandboxes")

    gh = Github()
    repo = gh.get_repo("modal-projects/training-gym")

    prs = defaultdict(list)
    for (pr_number, _), deployment in list(deployments.items()):
        prs[pr_number].append(PreviewDeployment(**deployment))

    for pr_number, pr_deployments in prs.items():
        try:
            pr = repo.get_pull(pr_number)
            is_open = pr.state == "open"

            for deployment in pr_deployments:
                try:
                    key = (pr_number, deployment.type)
                    if is_open:
                        if needs_refresh(deployment):
                            try:
                                print(
                                    f"Refreshing {deployment.type} deploy for {deployment.artifact}"
                                )
                                deployment.terminate()
                                deployment.deploy()
                            finally:
                                deployments[key] = asdict(deployment)
                    else:
                        if is_expired(deployment):
                            try:
                                deployment.terminate()
                            except:
                                deployments[key] = asdict(deployment)
                                raise
                            deployment.cleanup_artifact()
                            deployments.pop(key, None)
                except Exception as e:
                    action = "refresh" if is_open else "clean up"
                    print(
                        f"Error: Failed to {action} {deployment.type} preview for "
                        f"#{pr_number} (sandbox={deployment.sandbox_id}, "
                        f"artifact={deployment.artifact}): {e!r}"
                    )
                    traceback.print_exc()
        except UnknownObjectException:
            print(f"Warning: Unknown PR #{pr_number}")


@app.function(image=redirector_image)
@modal.asgi_app()
def preview_redirector():
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, RedirectResponse

    redirector = FastAPI()

    @redirector.get("/{pr_number}/{type}/{path:path}")
    @redirector.get("/{pr_number}/{type}")
    async def redirect_to_preview(
        request: Request, pr_number: int, type: PreviewType, path: str = ""
    ):
        deployment_dict = await deployments.get.aio((pr_number, type), None)
        if deployment_dict is None:
            return PlainTextResponse(
                content="Preview not found",
                status_code=404,
            )

        deployment = PreviewDeployment(**deployment_dict)
        if deployment.url is None:
            return PlainTextResponse(
                content="Deployment is missing a URL",
                status_code=503,
            )

        url = deployment.url
        if path:
            url = f"{url}/{path}"
        if request.query_params:
            url = f"{url}?{request.query_params}"
        return RedirectResponse(url)

    return redirector
