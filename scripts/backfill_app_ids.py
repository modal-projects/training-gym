"""Backfill modal_app_id on training runs by matching timestamps against Modal's app list.

The `modal app list --json` CLI returns recently stopped apps. Each training run's
created_at is matched to the Modal app whose creation time is closest (within 120s).

Usage:
    uv run python scripts/backfill_app_ids.py          # dry-run
    uv run python scripts/backfill_app_ids.py --apply   # write changes
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

from modal_training_gym.utils.metadata import MetadataStore, vol_list, vol_put


def _get_modal_apps(app_name: str) -> list[dict]:
    result = subprocess.run(
        ["modal", "app", "list", "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  modal app list failed: {result.stderr}")
        return []
    apps = json.loads(result.stdout)
    return [a for a in apps if a.get("Description") == app_name]


def _parse_modal_timestamp(ts: str) -> int:
    ts = ts.strip()
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    runs = vol_list(MetadataStore.TRAINING_RUNS)
    needs_backfill = [r for r in runs if not r.get("modal_app_id")]
    print(f"Found {len(runs)} runs, {len(needs_backfill)} missing modal_app_id")

    if not needs_backfill:
        print("Nothing to backfill.")
        return

    app_names = {r.get("run_id", "").rsplit("-", 1)[0] for r in needs_backfill}
    modal_apps = []
    for name in app_names:
        print(f"Fetching Modal apps named '{name}'...")
        modal_apps.extend(_get_modal_apps(name))
    print(f"Found {len(modal_apps)} matching Modal apps")

    app_timestamps = []
    for a in modal_apps:
        ts = _parse_modal_timestamp(a.get("Created at", ""))
        app_timestamps.append((ts, a["App ID"]))
    app_timestamps.sort()

    updates = []
    for run in needs_backfill:
        run_id = run.get("run_id", "")
        created_at = run.get("created_at", 0)
        if not created_at:
            continue

        best_app_id = None
        best_delta = float("inf")
        for app_ts, app_id in app_timestamps:
            delta = abs(created_at - app_ts)
            if delta < best_delta:
                best_delta = delta
                best_app_id = app_id

        if best_app_id and best_delta <= 120:
            run["modal_app_id"] = best_app_id
            updates.append((run_id, run, best_app_id, best_delta))

    if not updates:
        print("No matches found within 120s window.")
        return

    print(f"\n{len(updates)} runs to update{'  (dry-run)' if not args.apply else ''}:\n")
    for run_id, _, app_id, delta in updates:
        print(f"  {run_id} -> {app_id}  (delta: {delta}s)")

    if args.apply:
        print("\nApplying...")
        for run_id, run, _, _ in updates:
            vol_put(MetadataStore.TRAINING_RUNS, run_id, run)
        print(f"Done. Updated {len(updates)} runs.")
    else:
        print("\nDry run. Pass --apply to write changes.")


if __name__ == "__main__":
    main()
