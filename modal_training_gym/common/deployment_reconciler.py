"""Reconcile orphaned deployments stuck in non-terminal status."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modal_training_gym.common.deployment import (
    DeploymentStatus,
    update_deployment_status,
)
from modal_training_gym.common.modal_lifecycle import resolve_app_liveness
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items_healed,
    vol_list,
)

ACTIVE_DEPLOYMENT_STATUSES = frozenset(
    {
        "",
        "pending",
        DeploymentStatus.INITIALIZING.value,
        DeploymentStatus.READY.value,
        DeploymentStatus.RUNNING.value,
    }
)


@dataclass(frozen=True)
class DeployReconcileDecision:
    should_terminalize: bool
    reason: str | None = None
    modal_app_state: int | None = None


@dataclass(frozen=True)
class DeployReconcileResult:
    deployment_id: str
    reason: str
    previous_status: str


def _deployment_status(deployment: dict[str, Any]) -> str:
    return str(deployment.get("status") or "").strip().lower()


def _deployment_id(deployment: dict[str, Any]) -> str:
    return str(deployment.get("deployment_id") or "").strip()


def reconcile_decision(
    deployment: dict[str, Any],
    *,
    app_live: bool | None,
    modal_app_state: int | None = None,
) -> DeployReconcileDecision:
    if _deployment_status(deployment) not in ACTIVE_DEPLOYMENT_STATUSES:
        return DeployReconcileDecision(False)

    app_id = str(deployment.get("modal_app_id") or "").strip()
    if app_id and app_live is False:
        return DeployReconcileDecision(
            True,
            reason="stale_modal_app_terminated",
            modal_app_state=modal_app_state,
        )

    return DeployReconcileDecision(False)


def _load_active_deployments() -> list[dict[str, Any]]:
    deployments_by_id: dict[str, dict[str, Any]] = {}

    for raw in vol_list(MetadataStore.DEPLOYMENTS):
        if not isinstance(raw, dict):
            continue
        deployment_id = _deployment_id(raw)
        if deployment_id:
            deployments_by_id[deployment_id] = raw

    for raw in vol_get_summary_items_healed(MetadataStore.DEPLOYMENTS_SUMMARY) or []:
        if not isinstance(raw, dict):
            continue
        deployment_id = _deployment_id(raw)
        if deployment_id:
            deployments_by_id.setdefault(deployment_id, raw)

    return [
        deployment
        for deployment in deployments_by_id.values()
        if _deployment_status(deployment) in ACTIVE_DEPLOYMENT_STATUSES
    ]


def reconcile_orphan_deployments(
    *,
    dry_run: bool = False,
    get_lifecycle_state: Callable[[str], int | None] | None = None,
) -> list[DeployReconcileResult]:
    """Terminalize orphaned deployments. Returns reconciled deployment summaries."""
    results: list[DeployReconcileResult] = []
    for deployment in _load_active_deployments():
        app_id = str(deployment.get("modal_app_id") or "").strip()
        modal_app_state, app_live = resolve_app_liveness(
            app_id,
            get_lifecycle_state=get_lifecycle_state,
        )

        decision = reconcile_decision(
            deployment,
            app_live=app_live,
            modal_app_state=modal_app_state,
        )
        if not decision.should_terminalize or not decision.reason:
            continue

        deployment_id = _deployment_id(deployment)
        previous_status = _deployment_status(deployment)
        result = DeployReconcileResult(
            deployment_id=deployment_id,
            reason=decision.reason,
            previous_status=previous_status,
        )
        if dry_run:
            results.append(result)
            continue

        try:
            wrote = update_deployment_status(
                deployment_id,
                DeploymentStatus.STOPPED.value,
                seed=deployment,
            )
        except Exception as exc:
            print(f"WARNING: failed to reconcile {deployment_id}: {exc}")
            continue

        if not wrote:
            print(
                f"WARNING: failed to reconcile {deployment_id}: "
                "status write did not persist"
            )
            continue

        results.append(result)

    return results
