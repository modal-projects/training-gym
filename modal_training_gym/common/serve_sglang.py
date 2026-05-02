"""SGLang serving helper — builds a Modal app that hosts a model via SGLang.

Mirrors ``serve_vllm.py`` but launches an SGLang OpenAI-compatible server
instead.  One ``@modal.web_server``-decorated function runs
``python -m sglang.launch_server`` and exposes the chat-completions API at
``https://<workspace>--<app_name>-serve.modal.run/v1/chat/completions``.

``model_path`` accepts either:
  - a **HuggingFace repo id** (e.g. ``"Qwen/Qwen3-4B"``) — SGLang downloads
    the weights from HF into the cache volume on first boot, or
  - an **absolute container path** to an HF-format checkpoint directory
    (must contain ``config.json``) — typically a subdirectory of a
    training-checkpoints volume you pass via ``checkpoints_volume``.

Deploy separately from the training app::

    modal deploy path/to/tutorial.py::serve_app
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modal_training_gym.common import GPUType

if TYPE_CHECKING:
    from modal import App, Volume


def build_sglang_serve_app(
    *,
    app_name: str,
    model_path: str,
    served_model_name: str,
    checkpoints_volume: "Volume | str | None" = None,
    checkpoints_mount_path: str = "/checkpoints",
    gpu: GPUType = "H100",
    n_gpu: int = 1,
    sglang_port: int = 8000,
    sglang_image_tag: str = "lmsysorg/sglang:v0.4.7-cu124",
    max_concurrent_requests: int = 32,
    scaledown_window_seconds: int = 60,
    startup_timeout_seconds: int = 10 * 60,
    extra_sglang_args: list[str] | None = None,
) -> "App":
    """Build a Modal app that hosts a model via SGLang.

    Args:
        app_name: Modal app name. Deployed URL:
            ``https://<workspace>--<app_name>-serve.modal.run``.
        model_path: A HuggingFace repo id (downloaded on first boot into
            the HF cache volume) **or** an absolute container path to an
            HF-format checkpoint (must contain ``config.json``). For the
            latter, pass the backing volume via ``checkpoints_volume``.
        served_model_name: The ``model`` field chat-completion clients
            pass in their requests (``--served-model-name``).
        checkpoints_volume: Volume holding a trained checkpoint — either
            a ``modal.Volume`` or a volume name string. ``None`` (default)
            skips mounting a training-checkpoints volume entirely,
            which is the right choice when ``model_path`` is an HF repo
            id and SGLang will download weights itself.
        checkpoints_mount_path: Where ``checkpoints_volume`` is mounted
            inside the container. ``model_path`` should live under this
            when serving a local checkpoint.
        gpu / n_gpu: Serving GPU type and count. ``n_gpu`` drives
            ``--tp`` (tensor parallelism).
        sglang_image_tag: Docker image for SGLang. Defaults to a stable
            release tag. Override with a nightly tag for newer models.
        max_concurrent_requests: How many in-flight requests one replica
            serves before Modal scales out another.
        scaledown_window_seconds: Idle window before the replica scales
            to zero. Defaults to 60 seconds so tutorial deployments do
            not keep GPU containers warm after use.
        startup_timeout_seconds: How long ``@modal.web_server`` waits for
            SGLang to start responding on ``sglang_port``.
        extra_sglang_args: Additional tokens appended to the
            ``sglang.launch_server`` command line. Use for model-specific
            knobs like ``["--reasoning-parser", "qwen3"]`` or
            ``["--context-length", "8192"]``.
    """
    import modal
    from modal import App, Image, Secret, Volume

    image = Image.from_registry(sglang_image_tag).env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )

    hf_cache_vol = Volume.from_name("huggingface-cache", create_if_missing=True)
    volumes: dict[str, Volume] = {
        "/root/.cache/huggingface": hf_cache_vol,
    }

    if checkpoints_volume is not None:
        if isinstance(checkpoints_volume, str):
            checkpoints_volume = Volume.from_name(
                checkpoints_volume, create_if_missing=True
            )
        volumes[checkpoints_mount_path] = checkpoints_volume

    tags = {
        "_modal_source": "training-gym",
        "_modal_job_type": "serving",
        "_modal_framework": "sglang-serve",
    }
    app = App(app_name, tags=tags)

    _extra = list(extra_sglang_args or [])

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
    @modal.web_server(port=sglang_port, startup_timeout=startup_timeout_seconds)
    def serve():
        import subprocess

        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--served-model-name",
            served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(sglang_port),
            "--tp",
            str(n_gpu),
            *_extra,
        ]
        print(*cmd)
        subprocess.Popen(" ".join(cmd), shell=True)

    app.serve = serve  # type: ignore[attr-defined]
    return app
