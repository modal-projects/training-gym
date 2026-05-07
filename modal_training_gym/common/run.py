"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from modal_training_gym.common.framework import Framework
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

TRAINING_RUNS_STORE_NAME = MetadataStore.TRAINING_RUNS.value

class TrainingRun(BaseModel):
    run_id: str
    modal_app_id: str
    framework: Framework
    config: Any
    dataset_id: str = ""
    deployment_id: str = ""
    created_at: int = 0
    started_at: int = 0
    ended_at: int | None = None
    duration_seconds: int | None = None
    metadata: dict[str, Any] | None = None

    def save(self) -> None:
        vol_put(MetadataStore.TRAINING_RUNS, self.run_id, self.model_dump(mode="json"))

    @classmethod
    def from_id(cls, run_id: str) -> "TrainingRun":
        return cls.model_validate(vol_get(MetadataStore.TRAINING_RUNS, run_id))

    @classmethod
    def list_runs(cls) -> list["TrainingRun"]:
        return [cls.model_validate(r) for r in vol_list(MetadataStore.TRAINING_RUNS)]
