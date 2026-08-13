"""Tests for orphaned deployment reconciliation."""

from __future__ import annotations

from modal_training_gym.common.deployment import DeploymentStatus
from modal_training_gym.common.deployment_reconciler import (
    reconcile_decision,
    reconcile_orphan_deployments,
)
from modal_training_gym.common.modal_lifecycle import resolve_app_liveness
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_get_summary_items,
    vol_put,
    vol_put_summary_items,
)

_DEAD_APP_STATE = 99


def _deployment(
    *,
    status: str = DeploymentStatus.RUNNING.value,
    modal_app_id: str = "ap-123",
    deployment_id: str = "deployment-1",
) -> dict[str, object]:
    return {
        "deployment_id": deployment_id,
        "modal_app_id": modal_app_id,
        "status": status,
    }


def test_stopped_deployment_is_ignored():
    decision = reconcile_decision(
        _deployment(status=DeploymentStatus.STOPPED.value),
        app_live=False,
    )

    assert decision.should_terminalize is False


def test_dead_modal_app_is_terminalized():
    decision = reconcile_decision(
        _deployment(),
        app_live=False,
        modal_app_state=_DEAD_APP_STATE,
    )

    assert decision.should_terminalize is True
    assert decision.reason == "stale_modal_app_terminated"
    assert decision.modal_app_state == _DEAD_APP_STATE


def test_live_modal_app_is_ignored():
    decision = reconcile_decision(_deployment(), app_live=True)

    assert decision.should_terminalize is False


def test_unknown_modal_app_state_is_ignored():
    decision = reconcile_decision(_deployment(), app_live=None)

    assert decision.should_terminalize is False


def test_empty_modal_app_id_is_ignored():
    decision = reconcile_decision(
        _deployment(modal_app_id=""),
        app_live=False,
    )

    assert decision.should_terminalize is False


def test_reconcile_orphan_deployments_updates_status(fake_volume):
    deployment = _deployment()
    vol_put(MetadataStore.DEPLOYMENTS, "deployment-1", deployment)

    results = reconcile_orphan_deployments(
        get_lifecycle_state=lambda _app_id: _DEAD_APP_STATE,
    )

    assert len(results) == 1
    assert results[0].deployment_id == "deployment-1"
    assert results[0].reason == "stale_modal_app_terminated"
    assert results[0].previous_status == DeploymentStatus.RUNNING.value
    assert (
        vol_get(MetadataStore.DEPLOYMENTS, "deployment-1")["status"]
        == DeploymentStatus.STOPPED.value
    )


def test_summary_only_orphan_persists_stopped(fake_volume):
    deployment = _deployment()
    vol_put_summary_items(MetadataStore.DEPLOYMENTS_SUMMARY, [deployment])

    results = reconcile_orphan_deployments(
        get_lifecycle_state=lambda _app_id: _DEAD_APP_STATE,
    )

    assert len(results) == 1
    assert results[0].deployment_id == "deployment-1"
    assert (
        vol_get(MetadataStore.DEPLOYMENTS, "deployment-1")["status"]
        == DeploymentStatus.STOPPED.value
    )
    summary_items = vol_get_summary_items(MetadataStore.DEPLOYMENTS_SUMMARY) or []
    assert len(summary_items) == 1
    assert summary_items[0]["status"] == DeploymentStatus.STOPPED.value


def test_dry_run_leaves_status_unchanged(fake_volume):
    deployment = _deployment()
    vol_put(MetadataStore.DEPLOYMENTS, "deployment-1", deployment)
    vol_put_summary_items(MetadataStore.DEPLOYMENTS_SUMMARY, [deployment])

    results = reconcile_orphan_deployments(
        dry_run=True,
        get_lifecycle_state=lambda _app_id: _DEAD_APP_STATE,
    )

    assert len(results) == 1
    assert (
        vol_get(MetadataStore.DEPLOYMENTS, "deployment-1")["status"]
        == DeploymentStatus.RUNNING.value
    )
    summary_items = vol_get_summary_items(MetadataStore.DEPLOYMENTS_SUMMARY) or []
    assert summary_items[0]["status"] == DeploymentStatus.RUNNING.value


def test_resolve_app_liveness_fetches_lifecycle_once():
    calls: list[str] = []

    def get_state(app_id: str) -> int:
        calls.append(app_id)
        return _DEAD_APP_STATE

    state, live = resolve_app_liveness("ap-123", get_lifecycle_state=get_state)

    assert calls == ["ap-123"]
    assert state == _DEAD_APP_STATE
    assert live is False
