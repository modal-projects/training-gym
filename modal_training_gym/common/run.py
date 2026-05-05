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
        from modal_training_gym.common.client import _post

        _post("/run", self.model_dump())

    @classmethod
    def from_id(cls, run_id: str) -> "TrainingRun":
        from modal_training_gym.common.client import _get

        return cls.model_validate(_get(f"/runs/{run_id}"))

    @classmethod
    def list_runs(cls) -> list["TrainingRun"]:
        from modal_training_gym.common.client import _get

        return [cls.model_validate(r) for r in _get("/runs")]

    @classmethod
    def list_runs_with_train_result(cls) -> list[TrainingRunWithResult]:
        from modal_training_gym.common.client import _get
        from modal_training_gym.common.train_result import TrainResult

        runs = cls.list_runs()
        train_result_records: list[dict[str, Any]] = _get("/train-results")

        train_results_by_run_id: dict[str, TrainResult] = {}
        for record in train_result_records:
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
