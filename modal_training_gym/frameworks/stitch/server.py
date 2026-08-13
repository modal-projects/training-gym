"""Rollout-Server bring-up and teardown: SGLang plus the stitch sidecar.

Vendored from the stitch cookbook (``cookbook/common/server.py``). The Modal
``Server`` class in :mod:`launcher` is a thin shell whose ``@enter``/``@exit``
delegate here. The container's public port is the sidecar; it fronts the private
SGLang server on ``sglang_port``.
"""

from __future__ import annotations

import os
from typing import Any

from huggingface_hub import snapshot_download

from modal_training_gym.frameworks.stitch import sidecar_process

# What the engine needs to seed a delta from the base checkpoint: weights plus
# the config/tokenizer files beside them. Restricting the resolve to these keeps
# it from failing on a cache the SGLang server populated itself (it fetches no
# README/figures, and a snapshot missing *any* file is "incomplete").
_CHECKPOINT_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
]


def base_checkpoint_dir(model_name: str) -> str:
    """The local checkpoint directory the SGLang server loaded.

    A prepared baseline (the NVFP4 conversion on the checkpoints Volume) is
    already a path and is used as-is; a repo id is resolved to the snapshot the
    engine booted from.
    """
    if str(model_name).startswith("/"):
        return str(model_name)
    return snapshot_download(
        model_name, local_files_only=True, allow_patterns=_CHECKPOINT_PATTERNS
    )


def serve_startup(
    replica: Any,
    *,
    model_name: str,
    sglang_args: dict,
    sidecar_flags: list[str],
    delta_update_mode: str,
    tp: int,
    concurrency: int,
    sidecar_port: int,
    sglang_port: int,
    log_dir: str | None,
    startup_timeout: int,
) -> None:
    """Start SGLang + the versioned-proxy sidecar on one replica.

    SGLang serves from the immutable base checkpoint; the sidecar initializes the
    selected weight-update destination in the background once serving is up.
    """
    import modal.experimental
    from autoinference_utils.endpoint import (
        SGLangEndpoint,
        start_heartbeat_thread,
        warmup_chat_completions,
    )
    from stitch.service import sync_in_progress

    if delta_update_mode not in {"disk", "cpu"}:
        raise ValueError(f"unsupported delta update mode: {delta_update_mode!r}")
    cpu_cache_enabled = "--enable-cpu-weight-cache" in sglang_args
    if cpu_cache_enabled != (delta_update_mode == "cpu"):
        raise ValueError(
            "sglang_delta_update_mode must be 'cpu' exactly when "
            "--enable-cpu-weight-cache is present in sglang_server_args"
        )

    replica.endpoint = SGLangEndpoint(
        model_path=model_name,
        worker_port=sglang_port,
        tp=tp,
        extra_server_args=sglang_args,
        health_timeout=startup_timeout,
        health_poll_interval=10.0,
        log_requests_level=-1,
    )
    replica.endpoint.start()
    warmup_chat_completions(
        port=sglang_port,
        payload={
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        successful_requests=2,
        request_timeout=120.0,
        max_attempts_per_request=3,
    )
    replica.sidecar = sidecar_process.start_sidecar(
        [
            *sidecar_flags,
            "--base-checkpoint-dir",
            base_checkpoint_dir(model_name),
        ],
        sidecar_port=sidecar_port,
        sglang_port=sglang_port,
        log_path=(
            f"{log_dir}/sidecar-{os.environ.get('MODAL_TASK_ID', 'local')}.log"
            if log_dir
            else None
        ),
    )
    # Modal admits the container to Flash routing when @enter returns and never
    # re-polls /health, so blocking here (503 until the reconciler's first
    # catch-up) is the only thing keeping a not-yet-synced replica out of
    # rotation. A fresh boot clears at once; a mid-run joiner waits until it has
    # applied the live version, bounded by startup_timeout.
    sidecar_process.wait_http(
        f"http://127.0.0.1:{sidecar_port}/health", replica.sidecar, startup_timeout
    )

    def engine_health() -> str | None:
        # Weight staging can make the engine's health endpoint briefly stale;
        # suppress that expected blip only while the reconciler reports work.
        error = replica.endpoint.health_check()
        if error is None:
            return None
        server_info = f"http://127.0.0.1:{sidecar_port}/server_info"
        return None if sync_in_progress(server_info) else error

    start_heartbeat_thread(
        engine_health,
        on_failure=lambda: modal.experimental.stop_fetching_inputs(),
        max_consecutive_failures=12,  # ~1 min of sustained idle-state failures
    )
    print(f"Rollout server ready: model={model_name}, target_inputs={concurrency}")


def serve_stop(replica: Any) -> None:
    """Tear down the sidecar + SGLang (from ``@modal.exit``)."""
    sidecar_process.terminate_process(getattr(replica, "sidecar", None))
    endpoint = getattr(replica, "endpoint", None)
    if endpoint is not None:
        endpoint.stop()
