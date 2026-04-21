"""vLLM serving helper — builds a Modal app that hosts a trained checkpoint.

Pairs with any training framework that produces HuggingFace-format weights
(SLIME bridge mode, ms-swift merged LoRA, etc.). The returned `App` has a
single `VllmServer` class that mounts the checkpoints volume, starts vLLM
pointed at `model_path`, and exposes it through a Modal Flash endpoint so
the trained model is reachable from anywhere.

Deploy separately from the training app:

    modal deploy path/to/tutorial.py::serve_app

Query the Flash URL at `/v1/chat/completions` — it speaks the standard
OpenAI chat API, so any OpenAI-compatible client works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, GPUType

if TYPE_CHECKING:
    from modal import App


def build_vllm_serve_app(
    *,
    app_name: str,
    model_path: str,
    served_model_name: str,
    checkpoints_volume_name: str = "slime-checkpoints",
    checkpoints_mount_path: str = "/checkpoints",
    gpu: GPUType = "H100",
    n_gpu: int = 1,
    vllm_port: int = 8000,
    vllm_version: str = "0.11.2",
    flash_region: str | None = None,
    min_containers: int = 1,
    scaledown_window_seconds: int = 15 * 60,
    startup_timeout_seconds: int = 10 * 60,
    extra_vllm_args: list[str] | None = None,
) -> "App":
    """Build a Modal app that serves `model_path` via vLLM + Flash.

    Args:
        app_name: Modal app name. The derived Flash URL is
            `<workspace>--<app_name>-vllmserver.<region>.modal.direct`.
        model_path: Absolute path inside the checkpoints volume to an
            HF-format checkpoint directory (must contain `config.json`).
        served_model_name: The `--served-model-name` vLLM advertises to
            chat-completion clients.
        checkpoints_volume_name: Name of the Modal Volume holding the
            training checkpoints. Defaults to `slime-checkpoints`, matching
            the SLIME launcher.
        checkpoints_mount_path: Where to mount that volume inside the
            serving container. Should match whatever prefix `model_path`
            uses.
        gpu / n_gpu: Serving GPU type and count; drives both Modal
            allocation and vLLM's `--tensor-parallel-size`.
        flash_region: Region for the Flash endpoint (e.g. `"us-east"`).
            When unset, the app runs without Flash forwarding.
        extra_vllm_args: Additional tokens appended to the `vllm serve`
            command line — use for `--reasoning-parser qwen3`,
            `--enforce-eager`, etc.
    """
    import modal
    import modal.experimental
    from modal import App, Image, Secret, Volume

    image = (
        Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
        )
        .entrypoint([])
        .uv_pip_install(
            f"vllm=={vllm_version}",
            "huggingface-hub==0.36.0",
            "flashinfer-python==0.5.2",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )

    checkpoints_vol = Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache_vol = Volume.from_name("vllm-cache", create_if_missing=True)

    tags = {**COMMON_TRAINING_GYM_TAGS, "framework": "vllm-serve"}
    app = App(app_name, tags=tags)

    cls_kwargs: dict = dict(
        image=image,
        gpu=f"{gpu}:{n_gpu}",
        scaledown_window=scaledown_window_seconds,
        startup_timeout=startup_timeout_seconds,
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/root/.cache/vllm": vllm_cache_vol,
            checkpoints_mount_path: checkpoints_vol,
        },
        secrets=[Secret.from_name("huggingface-secret")],
        min_containers=min_containers,
        serialized=True,
    )
    if flash_region is not None:
        cls_kwargs["experimental_options"] = {"flash": flash_region}
        cls_kwargs["region"] = flash_region

    _extra = list(extra_vllm_args or [])

    @app.cls(**cls_kwargs)
    class VllmServer:
        @modal.enter()
        def setup(self) -> None:
            import socket
            import subprocess
            import time

            cmd = [
                "vllm",
                "serve",
                model_path,
                "--served-model-name",
                served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(vllm_port),
                "--tensor-parallel-size",
                str(n_gpu),
                *_extra,
            ]
            print(" ".join(cmd))
            self._vllm_process = subprocess.Popen(" ".join(cmd), shell=True)

            for _ in range(startup_timeout_seconds):
                try:
                    socket.create_connection(
                        ("localhost", vllm_port), timeout=1
                    ).close()
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError(
                    f"vLLM failed to start on port {vllm_port} "
                    f"within {startup_timeout_seconds}s"
                )
            print(f"vLLM ready on port {vllm_port}")

            if flash_region is not None:
                self.flash_manager = modal.experimental.flash_forward(vllm_port)
                print(f"Flash endpoint forwarded (region={flash_region})")

        @modal.method()
        def keepalive(self) -> None:
            """No-op — call from a client to keep a container warm."""

        @modal.exit()
        def cleanup(self) -> None:
            if hasattr(self, "flash_manager"):
                self.flash_manager.stop()
                self.flash_manager.close()
            if hasattr(self, "_vllm_process"):
                self._vllm_process.terminate()
                self._vllm_process.wait(timeout=10)

    app.VllmServer = VllmServer  # type: ignore[attr-defined]
    return app
