from __future__ import annotations

import functools
import os
from pathlib import Path

MARKER = "PATCHED_VOLUME_COMMIT_ON_SAVE"
TARGET = Path("/root/Megatron-LM/megatron/training/checkpointing.py")


def commit_after_save(save_checkpoint):
    @functools.wraps(save_checkpoint)
    def wrapped(*args, **kwargs):
        result = save_checkpoint(*args, **kwargs)
        volume_name = os.environ.get("TRAINING_GYM_CHECKPOINTS_VOLUME", "")
        if volume_name == "":
            return result
        import modal
        import torch.distributed as dist

        if dist.is_initialized() is False:
            raise RuntimeError("distributed checkpoint save returned uninitialized")

        dist.barrier()
        try:
            if int(os.environ["LOCAL_RANK"]) == 0:
                modal.Volume.from_name(volume_name).commit()
                print(
                    f"[training-gym] committed checkpoint volume {volume_name} "
                    f"from rank {dist.get_rank()}"
                )
        finally:
            dist.barrier()
        return result

    return wrapped


WRAPPER = f"""

from modal_training_gym.frameworks.slime.modal_helpers.patches.patch_volume_commit_on_save import (  # {MARKER}
    commit_after_save,
)

save_checkpoint = commit_after_save(save_checkpoint)
"""


def main() -> None:
    if TARGET.exists() is False:
        raise FileNotFoundError(f"Megatron checkpointing module missing: {TARGET}")
    source = TARGET.read_text()
    if MARKER in source:
        print(f"{TARGET.name} already commits checkpoint shards")
        return
    if source.find("\ndef save_checkpoint(") == -1:
        raise RuntimeError(f"save_checkpoint missing from {TARGET}")
    TARGET.write_text(source + WRAPPER)
    print(f"Patched {TARGET} to commit checkpoint shards")


if __name__ == "__main__":
    main()
