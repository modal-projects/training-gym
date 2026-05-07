"""Backfill training run status on the metadata volume.

Sets status='stopped' for runs with no train result, and 'completed' for
runs that do have one.

Usage:
    uv run python scripts/backfill_runs.py          # dry-run (default)
    uv run python scripts/backfill_runs.py --apply   # write changes
"""

from __future__ import annotations

import argparse
import time

from modal_training_gym.utils.metadata import MetadataStore, vol_list, vol_put


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to the volume")
    args = parser.parse_args()

    runs = vol_list(MetadataStore.TRAINING_RUNS)
    results = vol_list(MetadataStore.TRAIN_RESULTS)

    result_run_ids = {r.get("training_run_id", "") for r in results}

    print(f"Found {len(runs)} training runs, {len(results)} train results")

    updates = []
    for run in runs:
        run_id = run.get("run_id", "")
        if not run_id:
            continue

        old_status = run.get("status", "(none)")
        has_result = run_id in result_run_ids
        status = run.get("status", "")

        if has_result and status != "completed":
            run["status"] = "completed"
        elif not has_result and status not in ("stopped", "failed"):
            run["status"] = "stopped"
        else:
            continue

        updates.append((run_id, run, old_status))

    if not updates:
        print("Nothing to backfill.")
        return

    print(f"\n{len(updates)} runs to update{'  (dry-run)' if not args.apply else ''}:\n")
    for run_id, run, old_status in updates:
        print(f"  {run_id}: {old_status} -> {run['status']}")

    if args.apply:
        print("\nApplying...")
        for run_id, run, _ in updates:
            if not run.get("ended_at"):
                run["ended_at"] = run.get("started_at") or int(time.time())
            if run.get("status") == "completed" and not run.get("completed_at"):
                run["completed_at"] = run.get("ended_at") or run.get("started_at") or int(time.time())
            if run.get("duration_seconds") is None:
                started = run.get("started_at", 0)
                ended = run.get("ended_at", 0)
                if started and ended:
                    run["duration_seconds"] = max(0, ended - started)
            vol_put(MetadataStore.TRAINING_RUNS, run_id, run)
        print(f"Done. Updated {len(updates)} runs.")
    else:
        print("\nDry run. Pass --apply to write changes.")


if __name__ == "__main__":
    main()
