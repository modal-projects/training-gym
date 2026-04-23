"""vLLM serving helper — builds a Modal app that hosts a model via vLLM.

Thin wrapper around the canonical
[Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference):
one `@modal.web_server`-decorated function that runs `vllm serve` and
exposes the OpenAI-compatible chat-completions API at
`https://<workspace>--<app_name>-serve.modal.run/v1/chat/completions`.

`model_path` accepts either:
  - a **HuggingFace repo id** (e.g. `"Qwen/Qwen3-4B"`) — vLLM downloads
    the weights from HF into the cache volume on first boot, or
  - an **absolute container path** to an HF-format checkpoint directory
    (must contain `config.json`) — typically a subdirectory of a
    training-checkpoints volume you pass via `checkpoints_volume`.

Deploy separately from the training app:

    modal deploy path/to/tutorial.py::serve_app
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, GPUType

if TYPE_CHECKING:
    from modal import App, Volume


def build_vllm_serve_app(
    *,
    app_name: str,
    model_path: str,
    served_model_name: str,
    checkpoints_volume: "Volume | str | None" = None,
    checkpoints_mount_path: str = "/checkpoints",
    gpu: GPUType = "H100",
    n_gpu: int = 1,
    vllm_port: int = 8000,
    vllm_version: str = "0.13.0",
    max_concurrent_requests: int = 32,
    scaledown_window_seconds: int = 15 * 60,
    startup_timeout_seconds: int = 10 * 60,
    extra_vllm_args: list[str] | None = None,
) -> "App":
    """Build a Modal app that hosts a model via vLLM.

    Args:
        app_name: Modal app name. Deployed URL:
            `https://<workspace>--<app_name>-serve.modal.run`.
        model_path: A HuggingFace repo id (downloaded on first boot into
            the HF cache volume) **or** an absolute container path to an
            HF-format checkpoint (must contain `config.json`). For the
            latter, pass the backing volume via `checkpoints_volume`.
        served_model_name: The `model` field chat-completion clients
            pass in their requests (`--served-model-name`).
        checkpoints_volume: Volume holding a trained checkpoint — either
            a `modal.Volume` or a volume name string. `None` (default)
            skips mounting a training-checkpoints volume entirely,
            which is the right choice when `model_path` is an HF repo
            id and vLLM will download weights itself.
        checkpoints_mount_path: Where `checkpoints_volume` is mounted
            inside the container. `model_path` should live under this
            when serving a local checkpoint.
        gpu / n_gpu: Serving GPU type and count. `n_gpu` also drives
            `--tensor-parallel-size`.
        vllm_version: `pip install vllm==<version>`. Pinned to the
            current canonical Modal vLLM example (0.13.0); bump if you
            need a newer release.
        max_concurrent_requests: How many in-flight requests one replica
            serves before Modal scales out another.
        scaledown_window_seconds: Idle window before the replica scales
            to zero.
        startup_timeout_seconds: How long `@modal.web_server` waits for
            vLLM to start responding on `vllm_port`.
        extra_vllm_args: Additional tokens appended to the
            `vllm serve` command line. Use for model-specific knobs like
            `["--reasoning-parser", "qwen3"]` (Qwen3) or
            `["--enforce-eager"]` (skip CUDA graph capture — faster
            cold starts, slightly slower steady-state).
    """
    import modal
    from modal import App, Image, Secret, Volume

    image = (
        Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
        )
        .entrypoint([])
        .uv_pip_install(
            f"vllm=={vllm_version}",
            "huggingface-hub==0.36.0",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )

    # HF + vLLM caches are always mounted — even when serving a local
    # checkpoint, vLLM writes JIT artifacts under /root/.cache/vllm.
    hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache_vol = Volume.from_name("vllm-cache", create_if_missing=True)
    volumes = {
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }

    # Only mount a training-checkpoints volume when the caller actually
    # needs one — skipping it lets the helper serve HF-Hub models with
    # no assumptions about the filesystem layout.
    if checkpoints_volume is not None:
        if isinstance(checkpoints_volume, str):
            checkpoints_volume = Volume.from_name(
                checkpoints_volume, create_if_missing=True
            )
        volumes[checkpoints_mount_path] = checkpoints_volume

    tags = {**COMMON_TRAINING_GYM_TAGS, "_modal_framework": "vllm-serve"}
    app = App(app_name, tags=tags)

    _extra = list(extra_vllm_args or [])

    @app.function(
        image=image,
        gpu=f"{gpu}:{n_gpu}",
        scaledown_window=scaledown_window_seconds,
        timeout=24 * 60 * 60,
        volumes=volumes,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="serve",
    )
    @modal.concurrent(max_inputs=max_concurrent_requests)
    @modal.web_server(port=vllm_port, startup_timeout=startup_timeout_seconds)
    def serve():
        import subprocess

        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
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
        print(*cmd)
        subprocess.Popen(" ".join(cmd), shell=True)

    app.serve = serve  # type: ignore[attr-defined]
    return app
