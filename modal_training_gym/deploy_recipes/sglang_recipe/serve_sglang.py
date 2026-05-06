"""SGLang serving helper — builds a Modal app that hosts a model via SGLang.

Uses the ``SGLangEndpoint`` subprocess pattern from autoinference:
the Modal function starts an SGLang server process, polls its health
endpoint until ready, runs a warmup request, then proxies traffic.

``model_path`` accepts either:
  - a **HuggingFace repo id** (e.g. ``"Qwen/Qwen3-4B"``) — SGLang
    downloads the weights from HF into the cache volume on first boot, or
  - an **absolute container path** to an HF-format checkpoint directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
) -> "App":
    import modal
    from modal import App, Image, Secret, Volume

    sglang_port = 8000

    image = (
        Image.from_registry(recipe.sglang_image)
        .entrypoint([])
        .run_commands(
            "sed -i 's/timeout_keep_alive=5/timeout_keep_alive=300/g'"
            " /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py",
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
    volumes: dict[str, Volume] = {
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
    startup_timeout = 20 * 60

    tags = {
        "_modal_source": "training-gym",
        "_modal_job_type": "serving",
        "_modal_framework": "sglang-serve",
    }
    app = App(app_name, tags=tags)

    server_args = recipe.server_args(served_model_name=served_model_name)
    _tp = recipe.tp
    _dp = recipe.dp

    @app.cls(
        image=image,
        gpu=gpu_spec,
        scaledown_window=10 * 60,
        timeout=24 * 60 * 60,
        volumes=volumes,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        include_source=False,
    )
    @modal.experimental.http_server(
        port=sglang_port,
        exit_grace_period=25,
        startup_timeout=startup_timeout,
        proxy_regions=["us-east"],
    )
    @modal.concurrent(target_inputs=8)
    class Server:
        @modal.enter()
        def startup(self):
            from modal_training_gym.deploy_recipes.sglang_recipe._sglang_endpoint import (
                SGLangEndpoint,
                warmup_chat_completions,
            )

            self.endpoint = SGLangEndpoint(
                model_path=model_path,
                worker_port=sglang_port,
                tp=_tp,
                dp=_dp,
                extra_server_args=server_args,
                health_timeout=float(startup_timeout),
            )
            self.endpoint.start()
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
            print(f"[training-gym] SGLang serving {served_model_name} ready.")

        @modal.exit()
        def stop(self):
            if hasattr(self, "endpoint"):
                self.endpoint.stop()

    app.serve = Server  # type: ignore[attr-defined]
    return app
