"""Tests for the orphan reconciler."""

from __future__ import annotations

from modal_training_gym.common.deployment_reconciler import DeployReconcileResult
from modal_training_gym.common.reconcile import reconcile
from modal_training_gym.common.run_reconciler import ReconcileResult


def test_reconcile_runs_both_cleanups(monkeypatch):
    run_result = ReconcileResult(
        training_run_id="run-1",
        reason="stale_modal_app_terminated",
        previous_status="running",
    )
    deployment_result = DeployReconcileResult(
        deployment_id="deployment-1",
        reason="stale_modal_app_terminated",
        previous_status="running",
    )
    calls: list[str] = []

    def fake_runs(*, dry_run: bool = False):
        calls.append(f"runs:{dry_run}")
        return [run_result]

    def fake_deployments(*, dry_run: bool = False):
        calls.append(f"deployments:{dry_run}")
        return [deployment_result]

    monkeypatch.setattr(
        "modal_training_gym.common.reconcile.reconcile_orphan_runs",
        fake_runs,
    )
    monkeypatch.setattr(
        "modal_training_gym.common.reconcile.reconcile_orphan_deployments",
        fake_deployments,
    )

    outcome = reconcile(dry_run=True)

    assert calls == ["runs:True", "deployments:True"]
    assert outcome.runs == [run_result]
    assert outcome.deployments == [deployment_result]
    assert outcome.run_error is None
    assert outcome.deployment_error is None


def test_reconcile_isolates_side_failures(monkeypatch):
    deployment_result = DeployReconcileResult(
        deployment_id="deployment-1",
        reason="stale_modal_app_terminated",
        previous_status="running",
    )
    calls: list[str] = []

    def fake_runs(*, dry_run: bool = False):
        calls.append("runs")
        raise RuntimeError("runs boom")

    def fake_deployments(*, dry_run: bool = False):
        calls.append("deployments")
        return [deployment_result]

    monkeypatch.setattr(
        "modal_training_gym.common.reconcile.reconcile_orphan_runs",
        fake_runs,
    )
    monkeypatch.setattr(
        "modal_training_gym.common.reconcile.reconcile_orphan_deployments",
        fake_deployments,
    )

    outcome = reconcile()

    assert calls == ["runs", "deployments"]
    assert outcome.runs == []
    assert outcome.deployments == [deployment_result]
    assert outcome.run_error == "runs boom"
    assert outcome.deployment_error is None
