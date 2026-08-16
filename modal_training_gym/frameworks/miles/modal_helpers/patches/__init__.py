"""Build-time patch commands for miles' in-container dashboard reporting.

The patch scripts rewrite the miles checkout in place (``/root/miles``), so any
image that clones miles — the miles launcher's, and the stitch trainer's, which
runs miles too — applies the same commands after the checkout is in place.
"""

from pathlib import Path

from modal_training_gym.common.patches import encode_patch

PATCHES_DIR = Path(__file__).parent

_SGLANG_ABORT_B64 = encode_patch("patch_sglang_abort", PATCHES_DIR)
_ROLLOUT_STATUS_B64 = encode_patch("patch_rollout_status_reporting", PATCHES_DIR)
_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", PATCHES_DIR)

# Router-abort resilience: transient rollout-cleanup failures otherwise crash the
# run. Not reporting, so it is applied separately from the pair below.
SGLANG_ABORT_PATCH_COMMAND = f"echo {_SGLANG_ABORT_B64} | base64 -d | python3"

# Phase/step-event + advantage-distribution reporting, which is what the
# dashboard's per-rollout progress reads.
REPORTING_PATCH_COMMANDS = (
    f"echo {_ROLLOUT_STATUS_B64} | base64 -d | python3",
    f"echo {_ADVANTAGE_DIST_B64} | base64 -d | python3",
)

__all__ = [
    "PATCHES_DIR",
    "REPORTING_PATCH_COMMANDS",
    "SGLANG_ABORT_PATCH_COMMAND",
]
