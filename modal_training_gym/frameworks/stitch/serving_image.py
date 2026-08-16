"""The weight-sync SGLang image the rollout pool serves on.

Modeled on the stitch cookbook (``cookbook/common/serving_image.py``); it stays
local because a ``modal.Image`` is built client-side, before any container the
cookbook could be imported from exists. The pool runs a forked SGLang whose
``/stage_weight_update`` prepares and checksums the next version while the engine
keeps serving, so it is a *different* image from the miles trainer image: no
trainer package is installed, and precision comes from the served checkpoint
rather than a ``--quantization`` flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import modal

from modal_training_gym.train_recipes.stitch_recipe.pins import (
    DEFAULT_SGLANG_RUNTIME,
    SGLangRuntime,
    stitch_install_commands,
)

SGLANG_CACHE_PATH = "/root/.cache/sglang"  # kernel/JIT cache; survives cold starts

_SERVING_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_MODULES_CACHE": "/tmp/huggingface/modules",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
}


def build_serving_image(
    *,
    hf_cache_path: str,
    delta_volume_name: str,
    bulletin_root: str,
    extra_packages: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
    runtime: SGLangRuntime = DEFAULT_SGLANG_RUNTIME,
) -> modal.Image:
    """The rollout-pool image: the pinned SGLang fork overlaid on its base image,
    the sidecar's runtime deps, and the stitch checkout (the sidecar runs as
    ``python3 -m cookbook.common.sidecar``)."""
    return (
        modal.Image.from_registry(runtime.image)
        .run_commands(
            "rm -rf /tmp/stitch-sglang-overlay"
            f" && git clone --filter=blob:none --single-branch --branch {runtime.branch}"
            f" {runtime.repository} /tmp/stitch-sglang-overlay"
            f" && git -C /tmp/stitch-sglang-overlay checkout --detach {runtime.commit}"
            " && rm -rf /sgl-workspace/sglang/python/sglang"
            " && cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
            " && rm -rf /tmp/stitch-sglang-overlay"
        )
        # A baked HF cache must not shadow the mounted volume.
        .run_commands(f"rm -rf {hf_cache_path}")
        .pip_install(
            "autoinference-utils==0.2.3",  # SGLang server lifecycle + health heartbeat
            "fastapi",
            "httpx",
            "uvicorn",  # the stitch sidecar
            "zstandard",
            "xxhash",
            "blake3",  # engine-side weight-staging checksums
            "fastsafetensors",
            *extra_packages,
        )
        .run_commands(*stitch_install_commands())
        .env(
            {
                **_SERVING_ENV,
                **(extra_env or {}),
                "DELTA_VOLUME_NAME": delta_volume_name,
                "DELTA_BULLETIN_ROOT": bulletin_root,
            }
        )
        # The kernel-cache volume can't mount over a non-empty path — clear it as the
        # final filesystem step (it repopulates on boot).
        .run_commands(f"rm -rf {SGLANG_CACHE_PATH}")
        # The serialized Server class carries recipe objects in its closure, so the
        # package has to be importable when Modal deserializes it.
        .add_local_python_source("modal_training_gym", copy=True)
    )
