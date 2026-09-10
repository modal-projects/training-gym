from __future__ import annotations

import os
import time

from modal_training_gym.common import reporting
from modal_training_gym.common.training_steps import TrainingStep


def is_sft(args=None) -> bool:
    return (
        getattr(args, "loss_type", None) == "sft_loss"
        or os.environ.get("TRAINING_GYM_TRAINING_TYPE") == "sft"
    )


def report_train_metrics(args, metrics: dict) -> None:
    if not is_sft(args):
        return
    run_id = reporting._run_context(args)["training_run_id"]
    if not run_id:
        return
    step = TrainingStep(
        training_run_id=run_id,
        step=int(metrics["train/step"]),
        loss=metrics["train/loss"],
        grad_norm=metrics.get("train/grad_norm"),
        learning_rate=metrics.get("train/lr-pg_0"),
        created_at=time.time(),
    )
    reporting._enqueue(
        step.model_dump(),
        url=reporting._derive_url("/api/training-steps"),
        timeout_seconds=reporting._STEP_EVENT_TIMEOUT_SECONDS,
    )
