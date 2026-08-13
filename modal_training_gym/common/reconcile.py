"""Orchestrate orphan cleanup across training runs and deployments."""

from __future__ import annotations

from dataclasses import dataclass

from modal_training_gym.common.deployment_reconciler import (
    DeployReconcileResult,
    reconcile_orphan_deployments,
)
from modal_training_gym.common.run_reconciler import (
    ReconcileResult,
    reconcile_orphan_runs,
)


@dataclass(frozen=True)
class ReconcileOutcome:
    runs: list[ReconcileResult]
    deployments: list[DeployReconcileResult]
    run_error: str | None = None
    deployment_error: str | None = None


def reconcile(*, dry_run: bool = False) -> ReconcileOutcome:
    runs: list[ReconcileResult] = []
    deployments: list[DeployReconcileResult] = []
    run_error: str | None = None
    deployment_error: str | None = None

    try:
        runs = reconcile_orphan_runs(dry_run=dry_run)
    except Exception as exc:
        run_error = str(exc)
        print(f"WARNING: run reconciliation failed: {exc}")

    try:
        deployments = reconcile_orphan_deployments(dry_run=dry_run)
    except Exception as exc:
        deployment_error = str(exc)
        print(f"WARNING: deployment reconciliation failed: {exc}")

    return ReconcileOutcome(
        runs=runs,
        deployments=deployments,
        run_error=run_error,
        deployment_error=deployment_error,
    )
