"""Keep a resumed policy active after Slime snapshots its reference model.

Slime loads a reference checkpoint by temporarily writing it into the one
Megatron model instance, then snapshots those tensors under the ``ref`` tag.
On a resumed run the policy has already been snapshotted under ``actor``.  The
upstream initialization path did not switch back to that actor snapshot, so
the first rollout-weight sync served the reference/base model instead of the
restored policy.

This is deliberately a narrow, idempotent source patch: later training code
already switches between the actor and ref snapshots correctly.
"""

import os
from pathlib import Path


ROOT = Path(os.environ.get("SLIME_ROOT", "/root/slime"))
ACTOR = ROOT / "slime/backends/megatron_utils/actor.py"
MARKER = "TRAINING_GYM_RESTORE_ACTOR_AFTER_REF_LOAD"

source = ACTOR.read_text()
if MARKER in source:
    print("[patch_restore_actor_after_ref_load] actor.py already patched")
    raise SystemExit(0)

old = """\
        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)
"""
new = """\
        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)
            # TRAINING_GYM_RESTORE_ACTOR_AFTER_REF_LOAD: load_other_checkpoint
            # leaves self.model containing the reference snapshot.  Rollout
            # engines must receive the resumed/trainable actor instead.
            self._switch_model("actor")
            logger.info("Restored actor weights after taking frozen ref snapshot")
"""

if old not in source:
    raise RuntimeError(
        "[patch_restore_actor_after_ref_load] expected actor reference-load block not found"
    )

ACTOR.write_text(source.replace(old, new, 1))
print("[patch_restore_actor_after_ref_load] restored actor after ref snapshot")
