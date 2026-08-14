"""Upstream versions the disaggregated stitch flow is pinned to.

Three moving parts have to agree: the ``stitch`` control plane, the miles fork
that publishes deltas into it, and the SGLang fork that applies them behind
``/stage_weight_update``. Each is pinned to an exact commit (not a branch tip)
because the fetch/checkout is a cached image layer — a moving tip would leave a
stale build silently in place.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── stitch ─────────────────────────────────────────────────────────────────────
# The control plane: Store/Engine/Pool ports plus the trainer publish helpers and
# the rollout-service runtime the sidecar runs. modal-projects repos are public,
# so a build-time clone needs no token.
#
# The repo is cloned at this exact SHA rather than pip-installed, because the
# integration also runs stitch's ``cookbook`` package (the trainer hooks, the
# sidecar entrypoint, the Server bring-up, the checkpoint prep and the Megatron
# patches), which ships in the repo but not in the wheel.
STITCH_REPO_URL = "https://github.com/modal-projects/stitch.git"
STITCH_REPO_REF = "697cda79666fad8cfa7ab4a98b9f9f4f11cce1da"
STITCH_ROOT = "/root/stitch"

# ── miles fork ─────────────────────────────────────────────────────────────────
# Carries the disk-delta weight-sync path this flow drives: sparse export with a
# post-write publish hook, the generic HTTP rollout endpoint (``use_miles_router``),
# and TE-direct NVFP4 quantization shared by the trainer's export and the served
# base's conversion. Dated image tag, never ``latest``: Modal caches
# ``from_registry`` per tag string and will not re-pull a moved mutable tag.
MILES_IMAGE_TAG = "radixark/miles:dev-202607290235"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "1eb7520018446cb94b7406715f66dff1a271b53b"  # stitch-weight-sync-v0516
MILES_ROOT = "/root/miles"
# Source-only ``megatron.training`` must be on PYTHONPATH; the image installs
# megatron-core but the trainer imports the source tree.
MEGATRON_PATH = "/root/Megatron-LM"

# ── Megatron runtime patches ───────────────────────────────────────────────────
# Read out of the stitch clone. They are applied to the Megatron source tree at
# container start (not at build time) so an image layer is shared across recipes
# that need different subsets. R3 routing replay needs the dropless-dispatch fix;
# the reshardable-step patch lets a CPU-offloaded optimizer resume across a
# changed DP layout.
MEGATRON_PATCH_DIR = f"{STITCH_ROOT}/cookbook/miles_disagg/patches"
MEGATRON_R3_DISPATCH_PATCH = f"{MEGATRON_PATCH_DIR}/megatron-r3-dispatch.patch"
MEGATRON_RESHARDABLE_STEP_PATCH = (
    f"{MEGATRON_PATCH_DIR}/megatron-hdo-dp-reshardable-step.patch"
)


def stitch_install_commands() -> list[str]:
    """Image commands that install stitch *and* make its ``cookbook`` importable.

    ``pyproject.toml`` ships only ``src/stitch``, so the cookbook has to come from
    a checkout; a ``.pth`` file puts it on ``sys.path`` for every interpreter in
    the container, including the Ray actors and the sidecar subprocess, neither of
    which inherits an exported ``PYTHONPATH``.
    """
    return [
        f"rm -rf {STITCH_ROOT}"
        f" && git clone --filter=blob:none {STITCH_REPO_URL} {STITCH_ROOT}"
        f" && git -C {STITCH_ROOT} fetch --depth 1 origin {STITCH_REPO_REF}"
        f" && git -C {STITCH_ROOT} checkout --detach FETCH_HEAD"
        f" && python3 -m pip install {STITCH_ROOT}",
        'python3 -c "import pathlib, site;'
        f" pathlib.Path(site.getsitepackages()[0], 'stitch-cookbook.pth')"
        f".write_text('{STITCH_ROOT}')\"",
    ]


@dataclass(frozen=True)
class SGLangRuntime:
    """An SGLang source overlay and the ABI-compatible base image it is copied over."""

    image: str
    repository: str
    branch: str
    commit: str


# ── SGLang fork ────────────────────────────────────────────────────────────────
# Asynchronous weight staging (``/stage_weight_update``), correct quantized weight
# loading, and the optional CPU delta cache. See the fork's SGLANG_FORK.md for the
# patch stack and how to re-port onto a newer SGLang release.
DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.16",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.16",
    commit="1051a95a6ab16773037f8795a51aa03a1664a3b2",
)
