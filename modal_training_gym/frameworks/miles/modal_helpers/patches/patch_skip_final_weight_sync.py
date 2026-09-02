"""Skip miles' post-save weight sync on the final rollout.

``train.py``'s loop body ends every iteration with

    await offload_train()
    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()
    await actor_model.update_weights(rollout_id=rollout_id)
    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

and it runs that on the **last** rollout too, right after the final checkpoint
save, before falling out of the loop into ``dispose()``. On the last iteration
the sync pushes weights into engines that will never generate again: pure
teardown-time work whose output nothing reads.

At 550 B that redundant sync costs ~4 minutes of heavy NCCL and host-memory
activity immediately after a TB-scale save, and it is where runs die. Two
16-node runs (``achromatic-tint-51afd20a17fa``, ``timid-borzoi-e87a33779b29``)
completed every requested rollout, wrote a complete and valid checkpoint, and
were then recorded as failures because a node dropped out during this sync —
`Connection reset by peer` / `ActorUnavailableError` across several nodes, with
no Python or NCCL error and no OOM in the logs. Both would have been recorded
as succeeded had the loop simply stopped after the save.

The guard breaks out before the sync when this is the final rollout, unless an
eval is due for it — an eval generates, so it needs the fresh weights onloaded.
Runs longer than one rollout are unaffected: every non-final iteration still
syncs exactly as before, so this removes only the redundant terminal sync, not
the real ones.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_SKIP_FINAL_WEIGHT_SYNC"

TARGET = pathlib.Path("/root/miles/train.py")

OLD = """        await offload_train()
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
"""

NEW = """        # {marker}: the last rollout's sync feeds engines that will never
        # generate again, and at TB scale that redundant post-save work is
        # where 16-node runs lose a node. An eval still needs fresh weights.
        if rollout_id + 1 >= args.num_rollout and not should_run_periodic_action(
            rollout_id, args.eval_interval, num_rollout_per_epoch
        ):
            logger.info(
                "rollout %d is the final rollout and no eval is due; "
                "skipping the post-save weight sync",
                rollout_id,
            )
            break

        await offload_train()
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
""".format(marker=MARKER)


def _patch_file(target: pathlib.Path) -> None:
    if not target.exists():
        raise SystemExit(
            f"{target} not found; miles layout changed — re-check the patch."
        )

    src = target.read_text()
    if MARKER in src:
        print("final weight sync patch already applied")
        return

    if src.count(OLD) != 1:
        raise SystemExit(
            "final weight sync patch did not match; miles' train.py rollout loop "
            f"has changed (found {src.count(OLD)} candidate sites, expected 1). "
            "Re-check the loop tail before shipping."
        )

    target.write_text(src.replace(OLD, NEW, 1))
    print("Patched train.py to skip the redundant final-rollout weight sync")


if __name__ == "__main__":
    _patch_file(TARGET)
