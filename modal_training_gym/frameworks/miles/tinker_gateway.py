"""Modal-hosted Miles multi-LoRA Tinker gateway.

Miles' multi-LoRA v2 is not a dataset-driven training job: ``serve_tinker.py``
brings up SGLang engines and a Megatron trainer once, then serves the Tinker
protocol (``tinker==0.26.2``) over HTTP for as long as the process lives.
Clients create LoRA training clients, push ``forward_backward`` /
``optim_step`` work, save sampler snapshots and sample from them — none of
which maps onto ``TrainConfig.train()``.

This module hosts that server as an authenticated Modal ``@app.server``:

- the recipe's Miles image (patches and source overlays included) plus the
  ``tinker`` SDK the gateway imports at runtime;
- one clustered container per Miles node, with Ray started the same way
  ordinary Miles runs start it;
- the head submits ``serve_tinker.py`` as a Ray job (so the Megatron / SGLang
  actors inherit the same runtime env as ordinary runs) and waits for
  ``/api/v1/healthz``; worker ranks forward the gateway port to the head so
  whichever container Modal routes a request to answers it;
- adapter state and sampler snapshots live under ``tinker_checkpoint_root``
  on the checkpoints Volume, so they survive gateway restarts.

Each launch is recorded in the Training Gym dashboard as a ``TrainingRun`` with
no dataset (``metadata.run_type == "tinker_gateway"``): it stays ``RUNNING`` /
``serving`` while the app is up, turns ``STOPPED`` when the head container
exits, and collects one ``forward_backward`` timing interval per batch the
trainer executes (see ``patches/patch_tinker_timing.py``).
"""

from __future__ import annotations

import asyncio
import os
import secrets as _secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from modal_training_gym.common import hf_secrets
from modal_training_gym.common.checkpoint import require_within_volume_mount
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.ids import create_hash
from modal_training_gym.common.launcher_helpers import resolve_checkpoint_volumes
from modal_training_gym.common.launcher_utils import resolve_checkpoint_ref
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.run_summary import TINKER_GATEWAY_RUN_TYPE
from modal_training_gym.common.status import MilesStatus
from modal_training_gym.common.status_reporter import (
    enqueue_framework_status,
    flush as flush_status_reporter,
)
from modal_training_gym.frameworks.miles.modal_helpers.utils import build_train_cmd
from modal_training_gym.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    MilesRecipe,
)
from modal_training_gym.utils.metadata import MetadataStore, vol_put

if TYPE_CHECKING:
    import tinker

TINKER_PORT = 10613
TINKER_HEALTH_PATH = "/api/v1/healthz"
TINKER_SDK_VERSION = "0.26.2"
TINKER_SERVE_SCRIPT = "serve_tinker.py"
TINKER_SERVER_CLASS = "TinkerGatewayServer"

_DEFAULT_STARTUP_TIMEOUT = 60 * 60
_CHECKPOINT_COMMIT_INTERVAL_S = 60.0

_PATCH_TINKER_TIMING_B64 = encode_patch(
    "patch_tinker_timing", Path(__file__).parent / "modal_helpers" / "patches"
)
TINKER_TIMING_PATCH_COMMAND = (
    f"echo {_PATCH_TINKER_TIMING_B64} | base64 -d | python3"
    " || echo 'WARNING: gateway timing patch did not apply; forward_backward"
    " timings will be missing from the dashboard'"
)


def require_tinker_gateway(miles: MilesRecipe) -> None:
    """Reject recipes that do not configure Miles' Tinker gateway."""
    if not miles.is_tinker_gateway:
        raise TrainingGymConfigError(
            f"{type(miles).__name__} does not enable multi-LoRA "
            "(multi_lora_n_adapters is unset); TinkerGateway.launch() serves "
            "Miles' serve_tinker.py, use TrainConfig.train() for ordinary runs."
        )


def gateway_checkpoint_root(
    miles: MilesRecipe, *, app_name: str, checkpoints_mount_path: str
) -> str:
    """Resolve where the gateway keeps adapter state and sampler snapshots.

    An explicit ``tinker_checkpoint_root`` is honored but must sit on the
    checkpoints Volume; otherwise the root defaults to
    ``<mount>/<app_name>/tinker`` (the ``<save>/tinker`` layout
    ``serve_tinker.py`` derives itself).
    """
    root = miles.tinker_checkpoint_root or (
        f"{checkpoints_mount_path.rstrip('/')}/{app_name}/tinker"
    )
    return require_within_volume_mount(root, checkpoints_mount_path)[0]


@dataclass(frozen=True)
class GatewayStorage:
    """Checkpoints Volume the trainer and every engine mount for adapter state."""

    volume_name: str
    mount_path: str
    volume: Any
    checkpoint_root: str


def resolve_gateway_storage(miles: MilesRecipe, *, app_name: str) -> GatewayStorage:
    volume_name, mount_path, volume = resolve_checkpoint_volumes(
        None,
        volume_prefix=miles.name or app_name,
        default_mount_path=str(CHECKPOINTS_PATH),
    )
    return GatewayStorage(
        volume_name=volume_name,
        mount_path=mount_path,
        volume=volume,
        checkpoint_root=gateway_checkpoint_root(
            miles, app_name=app_name, checkpoints_mount_path=mount_path
        ),
    )


def gateway_recipe(
    miles: MilesRecipe,
    *,
    model: ModelConfig,
    hf_path: str,
    checkpoint_root: str,
) -> MilesRecipe:
    """Copy of ``miles`` with the invariants ``serve_tinker.py`` asserts.

    Trainer and engines must load the same frozen HF base (``load ==
    hf_checkpoint``), the checkpoint root is pinned, and clients address the
    base model by its Hub id unless the recipe names it otherwise.
    """
    if miles.hf_checkpoint and miles.hf_checkpoint != hf_path:
        raise TrainingGymConfigError(
            f"{type(miles).__name__}.hf_checkpoint={miles.hf_checkpoint!r} does "
            f"not match the served base model {hf_path!r}; the Tinker gateway "
            "serves exactly one frozen base, leave hf_checkpoint unset."
        )
    return replace(
        miles,
        hf_checkpoint=hf_path,
        load=hf_path,
        save=str(PurePosixPath(checkpoint_root).parent),
        tinker_checkpoint_root=checkpoint_root,
        tinker_server_host=miles.tinker_server_host or "0.0.0.0",
        tinker_server_port=miles.tinker_server_port or TINKER_PORT,
        tinker_base_model=miles.tinker_base_model or model.model_name,
    )


def build_tinker_gateway_cmd(
    miles: MilesRecipe, *, model: ModelConfig, miles_root: str
) -> str:
    """Entrypoint for the Ray job that serves the gateway.

    Always ``serve_tinker.py``: the recipe's ``async_mode`` selects between
    ``train.py`` / ``train_async.py`` for ordinary runs and has no meaning here.
    """
    require_tinker_gateway(miles)
    return build_train_cmd(miles, miles_root, model=model, script=TINKER_SERVE_SCRIPT)


def gateway_port(miles: MilesRecipe) -> int:
    return miles.tinker_server_port or TINKER_PORT


def tinker_gateway_image(image: Any) -> Any:
    """Add the client SDK the gateway imports (``tinker.types``) at runtime and
    the ``forward_backward`` timing patch, after every source overlay."""
    return image.uv_pip_install(f"tinker=={TINKER_SDK_VERSION}").run_commands(
        TINKER_TIMING_PATCH_COMMAND
    )


@dataclass(frozen=True)
class GatewayRunRecord:
    """Dashboard identity of one gateway launch, shared with its containers."""

    training_run_id: str
    framework_status_url: str = ""
    framework_status_token: str = ""


def gateway_training_run_id(
    recipe: MilesRecipe, model: ModelConfig, *, app_name: str
) -> str:
    return create_hash(
        model.model_name,
        "",
        f"{type(recipe).__name__}:{Framework.MILES.value}-tinker",
        app_name,
        model.model_path or "",
    )


def gateway_run_config(
    recipe: MilesRecipe, model: ModelConfig, *, app_name: str
) -> dict[str, Any]:
    """``TrainingRun.config`` for a gateway: model + recipe, no dataset."""
    return {
        "app_name": app_name,
        "framework": Framework.MILES.value,
        "entrypoint": TINKER_SERVE_SCRIPT,
        "model": {
            "model_name": model.model_name,
            "model_path": model.model_path,
            "miles_model_name": recipe.miles_model_name,
        },
        "recipe": {
            "type": type(recipe).__name__,
            "name": recipe.name,
            "multi_lora_n_adapters": recipe.multi_lora_n_adapters,
            "lora_rank": recipe.lora_rank,
            "lora_alpha": recipe.lora_alpha,
            "gpu_type": recipe.gpu_type,
            "actor_num_nodes": recipe.actor_num_nodes,
            "actor_num_gpus_per_node": recipe.actor_num_gpus_per_node,
            "rollout_num_gpus": recipe.rollout_num_gpus,
            "total_nodes": recipe.total_nodes,
            "gpus_per_node": recipe.gpu_allocation.gpus_per_node,
        },
        "dataset": {},
    }


def gateway_run_metadata(
    recipe: MilesRecipe,
    model: ModelConfig,
    *,
    checkpoint_root: str,
    checkpoints_volume_name: str,
    unauthenticated: bool,
    url: str = "",
) -> dict[str, Any]:
    return {
        "run_type": TINKER_GATEWAY_RUN_TYPE,
        "gateway_url": url,
        "base_model": recipe.tinker_base_model or model.model_name,
        "n_slots": recipe.multi_lora_n_adapters or 0,
        "tinker_sdk_version": TINKER_SDK_VERSION,
        "checkpoint_root": checkpoint_root,
        "checkpoints_volume_name": checkpoints_volume_name,
        "unauthenticated": unauthenticated,
    }


def create_gateway_training_run(
    recipe: MilesRecipe,
    model: ModelConfig,
    *,
    app_name: str,
    checkpoint_root: str,
    checkpoints_volume_name: str,
    unauthenticated: bool,
) -> tuple[TrainingRun, GatewayRunRecord]:
    """Persist the dashboard record for a gateway launch before deploying it.

    The run exists before the containers post status or timing events, which
    the dashboard rejects for unknown runs.
    """
    from modal_training_gym.cli.setup import ensure_dashboard_deployed
    from modal_training_gym.common.config import get_framework_status_url

    training_run_id = gateway_training_run_id(recipe, model, app_name=app_name)
    ensure_dashboard_deployed()
    framework_status_url = get_framework_status_url() or ""
    framework_status_token = _secrets.token_urlsafe(32)

    created_at = int(time.time())
    run = TrainingRun(
        training_run_id=training_run_id,
        framework=Framework.MILES,
        config=gateway_run_config(recipe, model, app_name=app_name),
        dataset_id="",
        framework_status=MilesStatus.INITIALIZING,
        created_at=created_at,
        started_at=created_at,
        app_name=app_name,
        source_model=model,
        metadata=gateway_run_metadata(
            recipe,
            model,
            checkpoint_root=checkpoint_root,
            checkpoints_volume_name=checkpoints_volume_name,
            unauthenticated=unauthenticated,
        ),
    )
    run.save()
    try:
        vol_put(
            MetadataStore.FRAMEWORK_STATUS_TOKENS,
            training_run_id,
            {"token": framework_status_token},
        )
    except BaseException as exc:
        _mark_gateway_run_failed(run, exc)
        raise
    print(f"TrainingRun recorded: {training_run_id}")
    return run, GatewayRunRecord(
        training_run_id=training_run_id,
        framework_status_url=framework_status_url,
        framework_status_token=framework_status_token,
    )


def bind_gateway_run_app(
    training_run_id: str, *, modal_app_id: str, url: str
) -> TrainingRun:
    """Attach the deployed Modal app and gateway URL to the run.

    Re-reads the stored run first: the head container may already have posted
    ``framework_status`` transitions that a save of the pre-deploy object would
    otherwise overwrite.
    """
    run = TrainingRun.from_id(training_run_id)
    run.modal_app_id = modal_app_id
    run.modal_app_url = modal_app_dashboard_url(modal_app_id)
    run.metadata = {**(run.metadata or {}), "gateway_url": url}
    run.save()
    return run


def _terminalize_gateway_run(
    run: TrainingRun, status: TrainingRunStatus, error_message: str | None
) -> bool:
    if run.status is not TrainingRunStatus.RUNNING:
        return False
    ended_at = int(time.time())
    run.status = status
    if error_message:
        run.error_message = error_message
    run.ended_at = ended_at
    run.completed_at = ended_at
    run.duration_seconds = max(0, ended_at - run.started_at)
    run.save()
    return True


def mark_gateway_run_stopped(training_run_id: str) -> None:
    """Terminal ``STOPPED`` for a gateway deliberately torn down via ``stop()``.

    A head container exiting is not a signal on its own: Modal may replace the
    container while the deployment stays live. Apps stopped elsewhere (``modal
    app stop``) are terminalized by the dashboard's orphan reconciler once it
    sees the app is dead.
    """
    _terminalize_gateway_run(
        TrainingRun.from_id(training_run_id), TrainingRunStatus.STOPPED, None
    )


def mark_gateway_run_failed(training_run_id: str, error_message: str) -> None:
    """Terminal ``FAILED`` for a gateway whose server died while its app is up."""
    _terminalize_gateway_run(
        TrainingRun.from_id(training_run_id), TrainingRunStatus.FAILED, error_message
    )


def wait_for_gateway_ready(
    *,
    port: int,
    timeout: float,
    is_dead: Callable[[], str | None],
    poll_interval: float = 5.0,
    host: str = "127.0.0.1",
) -> None:
    """Poll the gateway health endpoint until it answers or the job dies."""
    deadline = time.time() + timeout
    url = f"http://{host}:{port}{TINKER_HEALTH_PATH}"
    while time.time() < deadline:
        failure = is_dead()
        if failure:
            raise RuntimeError(f"Tinker gateway exited before serving: {failure}")
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if 200 <= resp.getcode() < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"Tinker gateway health check timed out after {timeout}s")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(64 * 1024):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def forward_port(listen_port: int, target_host: str, target_port: int) -> None:
    """Relay TCP connections on ``listen_port`` to ``target_host:target_port``.

    Worker ranks of the clustered server do not run the gateway; this keeps
    the server port answering on every container in the cluster.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                target_host, target_port
            )
        except OSError:
            writer.close()
            return
        await asyncio.gather(
            _pipe(reader, upstream_writer), _pipe(upstream_reader, writer)
        )

    server = await asyncio.start_server(handle, "0.0.0.0", listen_port)
    async with server:
        await server.serve_forever()


def start_port_forwarder(
    listen_port: int, target_host: str, target_port: int
) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: asyncio.run(forward_port(listen_port, target_host, target_port)),
        daemon=True,
        name="tinker-gateway-forwarder",
    )
    thread.start()
    return thread


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def build_tinker_gateway_app(
    *,
    miles: MilesRecipe,
    model: ModelConfig,
    app_name: str,
    unauthenticated: bool = False,
    startup_timeout: int = _DEFAULT_STARTUP_TIMEOUT,
    run_record: GatewayRunRecord | None = None,
) -> Any:
    """Build the Modal app that serves ``serve_tinker.py`` behind ``@app.server``.

    The returned app exposes the ``_Server`` handle as
    ``app.TinkerGatewayServer`` (mirroring ``build_sglang_serve_app``) so the
    launcher can resolve its URL after ``deploy()``. With ``run_record`` the
    head container reports ``serving`` and ``forward_backward`` timings for
    that run and marks it failed if ``serve_tinker`` dies; ``TinkerGateway.stop()``
    marks it stopped.
    """
    import modal
    from modal import App, Volume

    from modal_training_gym.common.ray_cluster import ModalRayCluster, clustered_if
    from modal_training_gym.frameworks.miles.launcher import (
        MILES_ROOT,
        build_miles_source_image,
        build_ray_runtime_env,
    )

    require_tinker_gateway(miles)

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    storage = resolve_gateway_storage(miles, app_name=app_name)
    checkpoints_volume = storage.volume
    checkpoints_volume_name = storage.volume_name
    checkpoint_root = storage.checkpoint_root
    run_record = run_record or GatewayRunRecord(training_run_id="")
    training_run_id = run_record.training_run_id
    volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        storage.mount_path: checkpoints_volume,
    }

    image = tinker_gateway_image(
        build_miles_source_image(miles).add_local_python_source(
            "modal_training_gym", copy=True
        )
    )

    port = gateway_port(miles)
    n_nodes = miles.total_nodes
    multi_node = n_nodes > 1
    gpu_spec = f"{miles.gpu_type}:{miles.gpu_allocation.gpus_per_node}"
    experimental_options: dict[str, Any] = {"efa_enabled": True} if multi_node else {}

    tags = {
        "_modal_source": "training-gym",
        "_modal_job_type": "serving",
        "_modal_framework": "miles-tinker",
        **miles.app_tags,
    }
    app = App(app_name, tags=tags)

    @app.server(
        image=image,
        gpu=gpu_spec,
        memory=miles.memory,
        cpu=miles.cpu,
        cloud=miles.cloud,
        compute_region=miles.region,
        volumes=volumes,
        secrets=hf_secrets(),
        serialized=True,
        include_source=False,
        port=port,
        startup_timeout=startup_timeout,
        min_containers=n_nodes,
        max_containers=n_nodes,
        experimental_options=experimental_options,
        unauthenticated=unauthenticated,
    )
    @clustered_if(multi_node, n_nodes, gpu_type=miles.gpu_type)
    class TinkerGatewayServer:
        @modal.enter()
        def start(self):
            hf_cache_volume.reload()
            checkpoints_volume.reload()

            cluster = ModalRayCluster()
            cluster.discover_cluster(n_nodes)
            os.environ["MILES_HOST_IP"] = cluster.node_ip
            os.environ["SGLANG_HOST_IP"] = cluster.node_ip
            os.environ["HOST_IP"] = cluster.node_ip

            self.cluster = cluster
            self.job_result: Any = None
            self.stop_committing = threading.Event()

            if not cluster.is_head:
                # Ray workers must join before the head returns from start_ray.
                cluster.start_ray()
                start_port_forwarder(port, cluster.head_addr, port)
                while not _port_open(port):
                    time.sleep(0.5)
                print(
                    f"[tinker-gateway] rank {cluster.rank} forwarding :{port} -> head"
                )
                return

            def _report(status: MilesStatus) -> None:
                if training_run_id:
                    enqueue_framework_status(
                        training_run_id,
                        status.value,
                        url=run_record.framework_status_url or None,
                        token=run_record.framework_status_token or None,
                        is_active=True,
                    )

            _report(MilesStatus.DOWNLOAD_MODEL)
            print(f"[tinker-gateway] downloading {model.model_name}")
            model.download()
            miles.download_model()
            hf_path = resolve_checkpoint_ref(model.model_path or model.model_name)
            hf_cache_volume.commit()
            os.makedirs(checkpoint_root, exist_ok=True)

            _report(MilesStatus.INITIALIZING)
            cluster.start_ray()

            served = gateway_recipe(
                miles, model=model, hf_path=hf_path, checkpoint_root=checkpoint_root
            )
            cmd = build_tinker_gateway_cmd(served, model=model, miles_root=MILES_ROOT)
            runtime_env = build_ray_runtime_env(
                head_addr=cluster.head_addr,
                metric_env={},
                environment=served.environment,
                substep_timing="auto" if training_run_id else "off",
                extra_env={
                    "TRAINING_GYM_TRAINING_RUN_ID": training_run_id,
                    "TRAINING_GYM_APP_NAME": app_name,
                    "TRAINING_GYM_CHECKPOINTS_VOLUME_NAME": checkpoints_volume_name,
                    "TRAINING_GYM_FRAMEWORK_STATUS_URL": (
                        run_record.framework_status_url
                    ),
                },
                framework_status_token=run_record.framework_status_token,
            )
            print(
                f"[tinker-gateway] {app_name}: {n_nodes} node(s) x {gpu_spec}, "
                f"{served.multi_lora_n_adapters} adapter slots, "
                f"checkpoint root {checkpoint_root}"
            )
            print(served.gpu_allocation.summary())
            print(f"[tinker-gateway] command: {cmd}")

            async def _serve() -> None:
                self.job_result = await cluster.submit_and_tail(
                    cmd, runtime_env=runtime_env
                )

            self.job_thread = threading.Thread(
                target=lambda: asyncio.run(_serve()),
                daemon=True,
                name="tinker-gateway-job",
            )
            self.job_thread.start()

            def _job_failure() -> str | None:
                result = self.job_result
                if result is None:
                    if self.job_thread.is_alive():
                        return None
                    return "Ray job submission thread exited"
                return result.message or f"Ray job status {result.status}"

            wait_for_gateway_ready(
                port=port, timeout=float(startup_timeout), is_dead=_job_failure
            )

            def _commit_loop() -> None:
                while not self.stop_committing.wait(_CHECKPOINT_COMMIT_INTERVAL_S):
                    try:
                        checkpoints_volume.commit()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[tinker-gateway] checkpoint commit failed: {exc!r}")

            threading.Thread(
                target=_commit_loop, daemon=True, name="tinker-gateway-commit"
            ).start()
            _report(MilesStatus.SERVING)

            def _watch_job() -> None:
                self.job_thread.join()
                if self.stop_committing.is_set() or not training_run_id:
                    return
                reason = _job_failure() or "Ray job exited"
                print(f"[tinker-gateway] serve_tinker exited: {reason}", flush=True)
                try:
                    mark_gateway_run_failed(training_run_id, reason)
                except Exception as exc:  # noqa: BLE001
                    print(f"[tinker-gateway] could not mark run failed: {exc!r}")

            threading.Thread(
                target=_watch_job, daemon=True, name="tinker-gateway-job-watch"
            ).start()
            print(
                f"[tinker-gateway] serving {served.tinker_base_model} on :{port} "
                f"(tinker=={TINKER_SDK_VERSION})"
            )

        @modal.exit()
        def stop(self):
            self.stop_committing.set()
            if self.cluster.is_head and self.job_result is None:
                client = self.cluster.client
                try:
                    for job in client.list_jobs():
                        if job.status.value in {"PENDING", "RUNNING"}:
                            client.stop_job(job.submission_id)
                except Exception as exc:  # noqa: BLE001
                    print(f"[tinker-gateway] could not stop Ray job: {exc!r}")
                try:
                    checkpoints_volume.commit()
                except Exception as exc:  # noqa: BLE001
                    print(f"[tinker-gateway] final checkpoint commit failed: {exc!r}")
            if self.cluster.is_head and training_run_id:
                flush_status_reporter()
            subprocess.run(["ray", "stop", "--force"], check=False)

    setattr(app, TINKER_SERVER_CLASS, TinkerGatewayServer)
    return app


class TinkerGateway(BaseModel):
    """Handle for a deployed Miles multi-LoRA Tinker gateway.

    Attributes:
        app_name: Modal app name.
        model: Frozen base model the gateway serves.
        recipe: Miles recipe the gateway was launched with.
        base_model: Name clients pass as ``base_model``.
        n_slots: Concurrent LoRA adapter slots.
        checkpoint_root: Volume path holding adapter state and sampler snapshots.
        checkpoints_volume_name: Modal Volume backing ``checkpoint_root``.
        unauthenticated: Whether the endpoint accepts requests without proxy auth.
        modal_app_id: Modal app ID.
        modal_app_url: Dashboard URL.
        url: Gateway base URL (the Tinker SDK's ``base_url``).
        training_run_id: Training Gym dashboard run recording this launch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str
    model: ModelConfig
    recipe: MilesRecipe
    base_model: str
    n_slots: int
    checkpoint_root: str
    checkpoints_volume_name: str
    unauthenticated: bool = False
    modal_app_id: str = ""
    modal_app_url: str = ""
    url: str
    training_run_id: str = ""

    @property
    def health_url(self) -> str:
        return f"{self.url.rstrip('/')}{TINKER_HEALTH_PATH}"

    @property
    def training_run(self) -> TrainingRun | None:
        """The dashboard ``TrainingRun`` for this gateway, if one was recorded."""
        if not self.training_run_id:
            return None
        return TrainingRun.from_id(self.training_run_id)

    def service_client(self, tenant: str, **kwargs: object) -> "tinker.ServiceClient":
        """``tinker.ServiceClient`` for ``tenant`` with proxy auth attached.

        See :func:`modal_training_gym.tinker.service_client`; requires the
        ``tinker`` SDK (``pip install tinker==0.26.2``).
        """
        from modal_training_gym.tinker import service_client

        return service_client(self, tenant=tenant, **kwargs)

    def lora_config_kwargs(self, rank: int | None = None) -> dict[str, int | bool]:
        """``create_lora_training_client`` arguments matching this gateway's layout.

        See :func:`modal_training_gym.tinker.lora_config_kwargs`.
        """
        from modal_training_gym.tinker import lora_config_kwargs

        return lora_config_kwargs(self.recipe, rank=rank)

    @classmethod
    def launch(
        cls,
        recipe: MilesRecipe,
        model: ModelConfig | None = None,
        *,
        app_name: str | None = None,
        unauthenticated: bool = False,
        startup_timeout: int = _DEFAULT_STARTUP_TIMEOUT,
        environment_name: str | None = None,
    ) -> "TinkerGateway":
        """Deploy ``serve_tinker.py`` for ``recipe`` and return its handle.

        Args:
            recipe:
                Miles recipe with ``multi_lora_n_adapters`` set (for example
                ``Qwen3_30B_A3B_Tinker_Recipe``).
            model:
                Frozen base model; defaults to the recipe's
                ``model_config_class``.
            app_name:
                Modal app name; defaults to ``<recipe name or model slug>-tinker``.
            unauthenticated:
                Serve without Modal proxy auth. The gateway accepts training
                jobs, so it is authenticated by default: clients send the
                ``MODAL_KEY`` / ``MODAL_SECRET`` pair as ``Modal-Key`` /
                ``Modal-Secret`` headers and their tenant ``X-API-Key``.
            startup_timeout:
                Seconds Modal allows the containers to load the base model and
                answer ``/api/v1/healthz``.
            environment_name:
                Modal environment to deploy into.

        Returns:
            The deployed gateway.
        """
        require_tinker_gateway(recipe)
        if model is None:
            model_cls = getattr(type(recipe), "model_config_class", None)
            if model_cls is None:
                raise TrainingGymConfigError(
                    f"{type(recipe).__name__} has no model_config_class; pass model="
                )
            model = model_cls()
        if not model.model_name:
            raise TrainingGymConfigError(
                f"{type(model).__name__}.model_name is required for the Tinker gateway"
            )

        slug = model.model_name.rstrip("/").split("/")[-1].replace("_", "-").lower()
        app_name = app_name or f"{recipe.name or slug}-tinker"

        storage = resolve_gateway_storage(recipe, app_name=app_name)
        run, run_record = create_gateway_training_run(
            recipe,
            model,
            app_name=app_name,
            checkpoint_root=storage.checkpoint_root,
            checkpoints_volume_name=storage.volume_name,
            unauthenticated=unauthenticated,
        )

        try:
            app = build_tinker_gateway_app(
                miles=recipe,
                model=model,
                app_name=app_name,
                unauthenticated=unauthenticated,
                startup_timeout=startup_timeout,
                run_record=run_record,
            )
            app.deploy(environment_name=environment_name)
        except BaseException as exc:
            _mark_gateway_run_failed(run, exc)
            raise

        try:
            server = getattr(app, TINKER_SERVER_CLASS)
            url = _resolve(server.get_url())
            if not url:
                raise RuntimeError(
                    f"Deployed {app_name!r} but no web URL was returned."
                )
            modal_app_id = app.app_id
            if not modal_app_id:
                raise RuntimeError(
                    f"Deployed {app_name!r} but no Modal app id was returned."
                )
            url = url.rstrip("/")
            bind_gateway_run_app(
                run.training_run_id, modal_app_id=modal_app_id, url=url
            )
        except BaseException as exc:
            # The app is live but the caller gets no handle to stop it.
            _stop_deployed_app(app.app_id or app_name, environment_name)
            _mark_gateway_run_failed(run, exc)
            raise
        return cls(
            app_name=app_name,
            model=model,
            recipe=recipe,
            base_model=recipe.tinker_base_model or model.model_name,
            n_slots=recipe.multi_lora_n_adapters or 0,
            checkpoint_root=storage.checkpoint_root,
            checkpoints_volume_name=storage.volume_name,
            unauthenticated=unauthenticated,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_dashboard_url(modal_app_id),
            url=url,
            training_run_id=run.training_run_id,
        )

    def wait_until_ready(self, timeout: int = _DEFAULT_STARTUP_TIMEOUT) -> None:
        """Block until ``/api/v1/healthz`` answers through the Modal proxy."""
        import requests

        from modal_training_gym.common.deployment import (
            _modal_proxy_auth_headers,
            _raise_for_proxy_auth,
        )

        headers = {} if self.unauthenticated else _modal_proxy_auth_headers()
        deadline = time.time() + timeout
        last_probe = "no response yet"
        print(f"Waiting for {self.app_name!r} — {self.modal_app_url}")
        while time.time() < deadline:
            probe = last_probe
            try:
                resp = requests.get(self.health_url, timeout=15, headers=headers)
                if resp.ok:
                    return
                _raise_for_proxy_auth(resp.status_code, self.url)
                probe = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:
                probe = f"{type(exc).__name__}"
            if probe != last_probe:
                last_probe = probe
                print(f"[tinker-gateway] probe {self.health_url}: {probe}")
            time.sleep(10)
        raise TimeoutError(
            f"{self.health_url} not ready after {timeout}s (last probe: {last_probe}). "
            f"Inspect logs with `modal app logs {self.modal_app_id}` or open "
            f"{self.modal_app_url}."
        )

    def stop(self) -> None:
        """Stop the gateway's Modal app and mark its dashboard run ``STOPPED``.

        Adapter state stays on the Volume.
        """
        subprocess.run(
            ["modal", "app", "stop", "-y", self.modal_app_id or self.app_name],
            check=True,
        )
        if self.training_run_id:
            mark_gateway_run_stopped(self.training_run_id)


def _resolve(value: Any) -> Any:
    """``_Server.get_url()`` is sync in some Modal versions and async in others."""
    from modal_training_gym.common.deployment import _run_coro

    return _run_coro(value)


def _stop_deployed_app(app_ref: str, environment_name: str | None) -> None:
    cmd = ["modal", "app", "stop", "-y"]
    if environment_name:
        cmd += ["--env", environment_name]
    try:
        subprocess.run([*cmd, app_ref], check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[tinker-gateway] could not stop {app_ref!r}: {exc!r}")


def _mark_gateway_run_failed(run: TrainingRun, exc: BaseException) -> None:
    """Terminalize a launch that failed locally; never masks ``exc``."""
    status = (
        TrainingRunStatus.STOPPED
        if isinstance(exc, KeyboardInterrupt)
        else TrainingRunStatus.FAILED
    )
    try:
        stored = TrainingRun.from_id(run.training_run_id)
    except Exception:  # noqa: BLE001
        stored = run
    try:
        _terminalize_gateway_run(stored, status, f"{type(exc).__name__}: {exc}")
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "TINKER_HEALTH_PATH",
    "TINKER_PORT",
    "TINKER_SDK_VERSION",
    "GatewayRunRecord",
    "TinkerGateway",
    "bind_gateway_run_app",
    "build_tinker_gateway_app",
    "build_tinker_gateway_cmd",
    "create_gateway_training_run",
    "gateway_checkpoint_root",
    "gateway_recipe",
    "gateway_run_config",
    "gateway_run_metadata",
    "mark_gateway_run_failed",
    "mark_gateway_run_stopped",
]
