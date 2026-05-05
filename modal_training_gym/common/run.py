"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from modal_training_gym.frameworks.base import Framework

if TYPE_CHECKING:
    from modal_training_gym.common.train_result import TrainResult

TRAINING_RUNS_STORE_NAME = "training-runs"


@dataclass(frozen=True)
class TrainingRunWithResult:
    run: "TrainingRun"
    train_result: "TrainResult | None"

    @property
    def has_train_result(self) -> bool:
        return self.train_result is not None


class TrainingRun(BaseModel):
    run_id: str
    modal_app_id: str
    framework: Framework
    config: Any
    dataset_id: str = ""
    deployment_id: str = ""
    created_at: int = 0
    metadata: dict[str, Any] | None = None

    def save(self) -> None:
        from modal_training_gym.common import vol_put

        vol_put(TRAINING_RUNS_STORE_NAME, self.run_id, self.model_dump(mode="json"))

    @classmethod
    def from_id(cls, run_id: str) -> "TrainingRun":
        from modal_training_gym.common import vol_get

        return cls.model_validate(vol_get(TRAINING_RUNS_STORE_NAME, run_id))

    @classmethod
    def list_runs(cls) -> list["TrainingRun"]:
        from modal_training_gym.common import vol_list

        return [cls.model_validate(r) for r in vol_list(TRAINING_RUNS_STORE_NAME)]

    @classmethod
    def list_runs_with_train_result(cls) -> list[TrainingRunWithResult]:
        from modal_training_gym.common.train_result import (
            TrainResult,
            TRAIN_RESULTS_STORE_NAME,
        )
        from modal_training_gym.common import vol_list

        runs = cls.list_runs()

        train_results_by_run_id: dict[str, TrainResult] = {}
        for record in vol_list(TRAIN_RESULTS_STORE_NAME):
            result_run_id = record.get("training_run_id", "")
            if result_run_id:
                train_results_by_run_id[result_run_id] = TrainResult(**record)

        return [
            TrainingRunWithResult(
                run=run,
                train_result=train_results_by_run_id.get(run.run_id),
            )
            for run in runs
        ]
