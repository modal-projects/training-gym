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
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from modal_training_gym.common import hf_secrets
from modal_training_gym.common.checkpoint import require_within_volume_mount
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.launcher_helpers import resolve_checkpoint_volumes
from modal_training_gym.common.launcher_utils import resolve_checkpoint_ref
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.frameworks.miles.modal_helpers.utils import build_train_cmd
from modal_training_gym.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    MilesRecipe,
)

if TYPE_CHECKING:
    import tinker

TINKER_PORT = 10613
TINKER_HEALTH_PATH = "/api/v1/healthz"
TINKER_SDK_VERSION = "0.26.2"
TINKER_SERVE_SCRIPT = "serve_tinker.py"
TINKER_SERVER_CLASS = "TinkerGatewayServer"

_DEFAULT_STARTUP_TIMEOUT = 60 * 60
_CHECKPOINT_COMMIT_INTERVAL_S = 60.0


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
    """Add the client SDK the gateway imports (``tinker.types``) at runtime."""
    return image.uv_pip_install(f"tinker=={TINKER_SDK_VERSION}")


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
) -> Any:
    """Build the Modal app that serves ``serve_tinker.py`` behind ``@app.server``.

    The returned app exposes the ``_Server`` handle as
    ``app.TinkerGatewayServer`` (mirroring ``build_sglang_serve_app``) so the
    launcher can resolve its URL after ``deploy()``.
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

            print(f"[tinker-gateway] downloading {model.model_name}")
            model.download()
            miles.download_model()
            hf_path = resolve_checkpoint_ref(model.model_path or model.model_name)
            hf_cache_volume.commit()
            os.makedirs(checkpoint_root, exist_ok=True)

            cluster.start_ray()

            served = gateway_recipe(
                miles, model=model, hf_path=hf_path, checkpoint_root=checkpoint_root
            )
            cmd = build_tinker_gateway_cmd(served, model=model, miles_root=MILES_ROOT)
            runtime_env = build_ray_runtime_env(
                head_addr=cluster.head_addr,
                metric_env={},
                environment=served.environment,
                substep_timing="off",
                extra_env={
                    "TRAINING_GYM_APP_NAME": app_name,
                    "TRAINING_GYM_CHECKPOINTS_VOLUME_NAME": checkpoints_volume_name,
                },
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

    @property
    def health_url(self) -> str:
        return f"{self.url.rstrip('/')}{TINKER_HEALTH_PATH}"

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

        app = build_tinker_gateway_app(
            miles=recipe,
            model=model,
            app_name=app_name,
            unauthenticated=unauthenticated,
            startup_timeout=startup_timeout,
        )
        app.deploy(environment_name=environment_name)

        server = getattr(app, TINKER_SERVER_CLASS)
        url = _resolve(server.get_url())
        if not url:
            raise RuntimeError(f"Deployed {app_name!r} but no web URL was returned.")
        modal_app_id = app.app_id
        if not modal_app_id:
            raise RuntimeError(
                f"Deployed {app_name!r} but no Modal app id was returned."
            )

        storage = resolve_gateway_storage(recipe, app_name=app_name)
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
            url=url.rstrip("/"),
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
        """Stop the gateway's Modal app. Adapter state stays on the Volume."""
        subprocess.run(
            ["modal", "app", "stop", self.modal_app_id or self.app_name], check=True
        )


def _resolve(value: Any) -> Any:
    """``_Server.get_url()`` is sync in some Modal versions and async in others."""
    from modal_training_gym.common.deployment import _run_coro

    return _run_coro(value)


__all__ = [
    "TINKER_HEALTH_PATH",
    "TINKER_PORT",
    "TINKER_SDK_VERSION",
    "TinkerGateway",
    "build_tinker_gateway_app",
    "build_tinker_gateway_cmd",
    "gateway_checkpoint_root",
    "gateway_recipe",
]
