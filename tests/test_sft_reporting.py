import ast
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from modal_training_gym.common.framework import Framework
from modal_training_gym.common import reporting
from modal_training_gym.common.reporting import _step_progress
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.run_summary import build_run_summary
from modal_training_gym.common.training_steps import TrainingStep
from modal_training_gym.frameworks.slime import phase_reporting, sft_reporting as sft
from modal_training_gym.frameworks.slime.modal_helpers.patches.patch_sft_reporting import (
    patch_data,
    patch_model,
)


def test_patch_captures_reduced_metrics_in_pinned_sources():
    for name in ("model.py.input", "model.py.timing.output"):
        source = (Path(__file__).parent / "testdata/slime" / name).read_text()
        patched = patch_model(source)
        ast.parse(patched)
        assert patched.count("report_train_metrics(args, log_dict)") == 1
        assert patch_model(patched) == patched
    with pytest.raises((RuntimeError, StopIteration)):
        patch_model("def train():\n    pass\n")


def test_sft_skips_upstream_rl_statistics():
    source = "def log_rollout_data(rollout_id, args, rollout_data):\n    raise RuntimeError('RL stats')\n"
    patched = patch_data(source)
    namespace = {}
    exec(patched, namespace)
    namespace["log_rollout_data"](0, SimpleNamespace(loss_type="sft_loss"), {})
    with pytest.raises(RuntimeError, match="RL stats"):
        namespace["log_rollout_data"](0, SimpleNamespace(loss_type="policy_loss"), {})
    assert patch_data(patched) == patched


def test_sft_logger_does_not_report_samples(monkeypatch):
    monkeypatch.setattr(
        reporting, "_enqueue_rollout", lambda *a, **k: pytest.fail("Sample reporting")
    )
    monkeypatch.setattr(
        phase_reporting, "_sample_to_dict", lambda *a, **k: pytest.fail("RL extraction")
    )
    assert (
        phase_reporting.log_rollout_data(
            0, SimpleNamespace(loss_type="sft_loss"), [object()], {}, 0
        )
        is True
    )


def test_prepared_sft_batch_is_not_a_completed_step():
    assert (
        _step_progress(SimpleNamespace(loss_type="sft_loss", num_rollout=10), 0)[
            "progress_current"
        ]
        == 0
    )
    assert _step_progress(SimpleNamespace(num_rollout=10), 0)["progress_current"] == 1


def test_sft_timing_omits_noop_rl_phases(monkeypatch):
    from modal_training_gym.common.timing_recorder import RoleRecorder

    monkeypatch.setenv("TRAINING_GYM_TRAINING_TYPE", "sft")
    recorder = RoleRecorder("driver", 0)
    for name in (
        "weight_sync",
        "initial_weight_sync",
        "reward",
        "compute_log_probs",
        "generate_rollouts",
        "forward_backward",
    ):
        with recorder.phase(name):
            pass
    assert set(recorder.phases) == {"prepare_batch", "forward_backward"}


def test_sft_does_not_publish_noop_weight_sync_status(monkeypatch):
    from modal_training_gym.common import reporting

    monkeypatch.setenv("TRAINING_GYM_TRAINING_TYPE", "sft")
    monkeypatch.setattr(
        reporting, "_ensure_worker", lambda: pytest.fail("No SFT weight sync")
    )
    monkeypatch.setenv(
        "TRAINING_GYM_FRAMEWORK_STATUS_URL",
        "https://dashboard.test/api/framework-status",
    )
    reporting._enqueue({"phase": "weight_sync"})


@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_nonfinite_loss_is_rejected(loss):
    with pytest.raises(ValueError):
        TrainingStep(training_run_id="sft-test", step=0, loss=loss, created_at=1)


def test_sft_metrics_use_shared_reporting_queue(monkeypatch):
    queue = Queue()
    monkeypatch.setattr(reporting, "_REPORT_QUEUE", queue)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    monkeypatch.setattr(reporting, "_ensure_worker", lambda: None)
    monkeypatch.setenv(
        "TRAINING_GYM_FRAMEWORK_STATUS_URL",
        "https://dashboard.test/api/framework-status",
    )
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "sft-test")
    sft.report_train_metrics(
        SimpleNamespace(loss_type="sft_loss"),
        {
            "train/step": 0,
            "train/loss": 1.5,
            "train/grad_norm": 0.2,
            "train/lr-pg_0": 1e-5,
        },
    )
    payload = queue.get_nowait()
    assert payload["_url"] == "https://dashboard.test/api/training-steps"
    assert payload["training_run_id"] == "sft-test"
    assert payload["step"] == 0
    assert payload["loss"] == 1.5
    assert payload["grad_norm"] == 0.2
    assert payload["learning_rate"] == 1e-5
    assert queue.empty()


@pytest.mark.parametrize("old_step,new_step", [(0, 1), (0, 0)])
def test_late_status_save_cannot_regress_sft_loss(fake_volume, old_step, new_step):
    from modal_training_gym.common.run import FrameworkStatusUpdate

    initial = TrainingStep(
        training_run_id="stale-sft", step=old_step, loss=2, created_at=1
    ).model_dump()
    TrainingRun(
        training_run_id="stale-sft",
        framework=Framework.SLIME,
        config={"training_type": "sft", "recipe": {"num_rollout": 10}},
        metadata={"latest_training_step": initial},
    ).save()
    stale = TrainingRun.from_id("stale-sft")
    fresh = TrainingRun.from_id("stale-sft")
    fresh.metadata["latest_training_step"] = TrainingStep(
        training_run_id="stale-sft", step=new_step, loss=1, created_at=2
    ).model_dump()
    fresh.save()
    stale.apply_framework_status(
        FrameworkStatusUpdate(
            training_run_id="stale-sft",
            phase="training",
            progress_current=0,
            progress_total=10,
        )
    )
    stale.save()
    saved = TrainingRun.from_id("stale-sft")
    assert saved.metadata["latest_training_step"]["loss"] == 1
    assert saved.metadata["framework_progress"]["current"] == 0
    summary = build_run_summary(saved.model_dump(mode="json"))
    assert summary.framework_progress.current == new_step + 1
    assert summary.framework_progress.total == 10


@pytest.mark.parametrize(
    "training_type,step,expected_current",
    [("rl", None, 8), ("rl", 2, 8), ("sft", None, 0), ("sft", 2, 3)],
)
def test_summary_derives_only_sft_progress(training_type, step, expected_current):
    progress = {
        "current": 8,
        "total": 20,
        "unit": "step",
        "phase": "checkpoint_save",
        "is_active": True,
    }
    metadata = {"framework_progress": progress}
    if step is not None:
        metadata["latest_training_step"] = TrainingStep(
            training_run_id="sft-test", step=step, loss=1, created_at=1
        ).model_dump()
    summary = build_run_summary(
        {
            "config": {"training_type": training_type, "recipe": {"num_rollout": 10}},
            "framework_status": "checkpoint_save",
            "metadata": metadata,
        }
    )
    assert summary.framework_progress.current == expected_current
    assert summary.framework_progress.total == (10 if training_type == "sft" else 20)
    assert summary.framework_progress.phase == "checkpoint_save"
    assert summary.framework_progress.is_active is True
    assert summary.display_stage == "Saving checkpoint"
    assert progress["current"] == 8
