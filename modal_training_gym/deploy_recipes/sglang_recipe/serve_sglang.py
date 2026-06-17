"""SGLang serving helper — builds a Modal app that hosts a model via SGLang.

The model is served by ``SGLangEndpoint``, a Modal *server* class registered
with ``@app._experimental_server`` (Modal's low-latency routing service for
inference workloads). The endpoint is the Modal class itself — its
``@modal.enter()`` starts the ``sglang.launch_server`` subprocess, waits for
the health endpoint, and runs a couple of warmup requests; its
``@modal.exit()`` tears the subprocess down. Modal proxies HTTP traffic
straight to the SGLang port, so there's no separate wrapper function.

``model_path`` accepts either:
  - a **HuggingFace repo id** (e.g. ``"Qwen/Qwen3-4B"``) — SGLang
    downloads the weights from HF into the cache volume on first boot, or
  - an **absolute container path** to an HF-format checkpoint directory.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal import App, Volume
    from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe


def build_sglang_serve_app(
    *,
    recipe: "SglangRecipe",
    app_name: str,
    model_path: str,
    served_model_name: str,
    checkpoints_volume: "Volume | str | None" = None,
    checkpoints_mount_path: str | None = None,
    deployment_id: str | None = None,
) -> "App":
    import modal
    from modal import App, Image, Volume

    from modal_training_gym.common import hf_secrets

    sglang_port = 8000

    image = (
        Image.from_registry(recipe.sglang_image)
        .entrypoint([])
        .run_commands(
            "sed -i 's/timeout_keep_alive=5/timeout_keep_alive=300/g'"
            " /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py",
            "rm -rf /root/.cache/huggingface",
        )
        # pydantic_core in the nightly image needs typing_extensions>=4.13 (Sentinel)
        .uv_pip_install("typing_extensions>=4.13")
        # Brand-new model architectures (e.g. deepseek_v4) aren't in released
        # transformers yet; install from git source so AutoConfig recognizes them.
        .run_commands(
            "uv pip install --system --no-build-isolation "
            "'transformers @ git+https://github.com/huggingface/transformers.git'"
        )
        .env(
            {
                "HF_HUB_CACHE": "/root/.cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_python_source("modal_training_gym", copy=True)
    )

    hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)
    volumes: dict[str | PurePosixPath, Any] = {
        "/root/.cache/huggingface": hf_cache_vol,
    }

    if checkpoints_volume is not None:
        mount = checkpoints_mount_path or "/checkpoints"
        if isinstance(checkpoints_volume, str):
            checkpoints_volume = Volume.from_name(
                checkpoints_volume, create_if_missing=True
            )
        volumes[mount] = checkpoints_volume

    n_gpu = recipe.tp or 1
    gpu_spec = f"{recipe.gpu}:{n_gpu}" if n_gpu > 1 else str(recipe.gpu)
    # Gates both Modal's container startup_timeout and the SGLang health poll.
    # Large models (GLM-4.7 355B, Kimi-K2.5 ~1T) need more than the 20-min
    # default to finish loading weights — bump via SglangRecipe.startup_timeout.
    startup_timeout = recipe.startup_timeout

    tags = {
        "_modal_source": "training-gym",
        "_modal_job_type": "serving",
        "_modal_framework": "sglang-serve",
    }
    app = App(app_name, tags=tags)

    server_args = recipe.server_args(served_model_name=served_model_name)
    _tp = recipe.tp
    _dp = recipe.dp
    _deployment_id = deployment_id

    @app._experimental_server(
        image=image,
        gpu=gpu_spec,
        scaledown_window=10 * 60,
        startup_timeout=startup_timeout,
        volumes=volumes,
        secrets=hf_secrets(),
        serialized=True,
        include_source=False,
        port=sglang_port,
        exit_grace_period=25,
        routing_region="us-east",
        target_concurrency=8,
    )
    class SGLangEndpoint:
        @modal.enter()
        def start(self):
            from modal_training_gym.deploy_recipes.sglang_recipe._sglang_endpoint import (
                build_server_cmd,
                rewrite_chat_template_kwargs,
                start_server,
                wait_for_server_ready,
                warmup_chat_completions,
            )

            args = rewrite_chat_template_kwargs(server_args, model_path=model_path)
            cmd = build_server_cmd(
                model_path=model_path,
                port=sglang_port,
                tp=_tp,
                dp=_dp,
                extra_server_args=args,
            )
            self.proc = start_server(cmd)
            wait_for_server_ready(
                self.proc,
                port=sglang_port,
                timeout=float(startup_timeout),
            )
            warmup_chat_completions(
                port=sglang_port,
                payload={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
                successful_requests=2,
                request_timeout=60.0,
            )
            if _deployment_id:
                from modal_training_gym.common.deployment import (
                    update_deployment_status,
                )

                update_deployment_status(_deployment_id, "running")
            print(f"[training-gym] SGLang serving {served_model_name} ready.")

        @modal.exit()
        def stop(self):
            from modal_training_gym.deploy_recipes.sglang_recipe._sglang_endpoint import (
                stop_server,
            )

            if _deployment_id:
                from modal_training_gym.common.deployment import (
                    update_deployment_status,
                )

                update_deployment_status(_deployment_id, "stopped")
            stop_server(getattr(self, "proc", None))

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)
    for tag, cls in app.registered_classes.items():
        setattr(app, tag, cls)
    # The decorator returns the `_Server` handle (carries `get_urls()`); the
    # entry in `registered_functions` is only its underlying Function. Bind the
    # server itself so callers (e.g. deployment URL resolution) get `get_urls`.
    setattr(app, "SGLangEndpoint", SGLangEndpoint)
    return app
