from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modal import Image


# Canonical mount point for `modal_training_gym/tools/` inside every framework
# image. Framework-agnostic helpers (e.g. `Kimi_K2_5.download`) shell out
# to scripts at this path, so it must exist on every launcher's image.
TOOLS_REMOTE_PATH = "/opt/training-gym/tools"

# Local path to the tools directory. Shipped with the package so
# `mount_tools_dir` works from an editable install or a `pip install`ed copy.
TOOLS_LOCAL_PATH = Path(__file__).resolve().parent.parent / "tools"


# TODO: use this
def mount_tools_dir(image: "Image") -> "Image":
    """Attach `modal_training_gym/tools/` to a Modal image at `TOOLS_REMOTE_PATH`.

    Every framework launcher should call this on its training image(s) so
    cross-framework scripts (e.g. `convert_kimi_int4_to_bf16.py`) resolve at
    a predictable remote path regardless of the framework image's layout.
    """
    return image.add_local_dir(
        TOOLS_LOCAL_PATH,
        remote_path=TOOLS_REMOTE_PATH,
        copy=True,
    )
