"""Backfill dashboard summary blobs on the metadata volume.

Writes single-blob summary payloads used by the dashboard for faster reads.

Usage:
    uv run python scripts/backfill_dashboard_summaries.py
    uv run python scripts/backfill_dashboard_summaries.py --apply
"""

from __future__ import annotations

import argparse

from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_list,
    vol_put,
    vol_put_summary_items,
)


def _sorted_training_runs() -> list[dict]:
    runs = vol_list(MetadataStore.TRAINING_RUNS)
    runs.sort(
        key=lambda item: (
            int(item.get("created_at", 0) or 0),
            str(item.get("run_id", "")),
        ),
        reverse=True,
    )
    return runs


def _sorted_train_results() -> list[dict]:
    results = vol_list(MetadataStore.TRAIN_RESULTS)
    results.sort(
        key=lambda item: str(item.get("training_run_id", "")),
        reverse=True,
    )
    return results


def _sorted_deployments() -> list[dict]:
    deployments = vol_list(MetadataStore.DEPLOYMENTS)
    deployments.sort(
        key=lambda item: (
            str(item.get("deployment_config", {}).get("app_name", "")),
            str(item.get("deployment_id", "")),
        ),
        reverse=True,
    )
    return deployments


def _eval_summaries() -> list[dict]:
    eval_results = vol_list(MetadataStore.EVAL_RESULTS)
    summaries: list[dict] = []
    for result in eval_results:
        rows = result.get("rows", [])
        total = len(rows) if isinstance(rows, list) else 0
        mean = (
            sum(row.get("score", 0) for row in rows) / total
            if total
            else 0.0
        )
        summary = {key: value for key, value in result.items() if key != "rows"}
        summary["total"] = total
        summary["mean"] = mean
        summaries.append(summary)

    summaries.sort(
        key=lambda item: (
            str(item.get("created_at", "")),
            str(item.get("eval_id", "")),
        ),
        reverse=True,
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write summary blobs")
    args = parser.parse_args()

    runs = _sorted_training_runs()
    train_results = _sorted_train_results()
    deployments = _sorted_deployments()
    eval_summaries = _eval_summaries()

    print("Dashboard summary backfill")
    print(f"  training-runs-summary: {len(runs)} items")
    print(f"  train-results-summary: {len(train_results)} items")
    print(f"  deployments-summary:   {len(deployments)} items")
    print(f"  evals summary:         {len(eval_summaries)} items")

    if not args.apply:
        print("\nDry run. Pass --apply to write summaries.")
        return

    vol_put_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY, runs)
    vol_put_summary_items(MetadataStore.TRAIN_RESULTS_SUMMARY, train_results)
    vol_put_summary_items(MetadataStore.DEPLOYMENTS_SUMMARY, deployments)
    vol_put(
        MetadataStore.EVALS,
        "summary",
        {"summaries": eval_summaries},
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
