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

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal import App, Volume
    from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe


def build_vllm_serve_app(
    *,
    recipe: "VllmRecipe",
    app_name: str,
    model_path: str,
    served_model_name: str,
    checkpoints_volume: "Volume | str | None" = None,
    checkpoints_mount_path: str | None = None,
) -> "App":
    import modal
    from modal import App, Image, Secret, Volume

    gpu = recipe.gpu or "H100"
    n_gpu = recipe.n_gpu or 1
    vllm_port = 8000
    vllm_version = "0.13.0"

    image = (
        Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install(
            f"vllm=={vllm_version}",
            "huggingface-hub==0.36.0",
        )
        .run_commands("rm -rf /root/.cache/huggingface")
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )

    hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)
    vllm_cache_vol = Volume.from_name("vllm-cache", create_if_missing=True)
    volumes: dict[str | PurePosixPath, Any] = {
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }

    if checkpoints_volume is not None:
        mount = checkpoints_mount_path or "/checkpoints"
        if isinstance(checkpoints_volume, str):
            checkpoints_volume = Volume.from_name(
                checkpoints_volume, create_if_missing=True
            )
        volumes[mount] = checkpoints_volume

    tags = {
        "_modal_source": "training-gym",
        "_modal_job_type": "serving",
        "_modal_framework": "vllm-serve",
    }
    app = App(app_name, tags=tags)

    _extra = list(recipe.extra_vllm_args or [])

    @app.function(
        image=image,
        gpu=f"{gpu}:{n_gpu}",
        scaledown_window=10 * 60,
        timeout=24 * 60 * 60,
        volumes=volumes,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="serve",
    )
    @modal.concurrent(max_inputs=32)
    @modal.web_server(port=vllm_port, startup_timeout=10 * 60)
    def serve():
        import os
        import subprocess

        if model_path.startswith("/") and os.path.isdir(model_path):
            hf_config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(hf_config_path):
                print("[training-gym] Converting checkpoint from mt to hf for serving.")

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

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)
    return app
