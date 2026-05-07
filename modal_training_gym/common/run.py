"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from modal_training_gym.common.framework import Framework
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_list,
    vol_put,
    vol_upsert_summary_item,
)

TRAINING_RUNS_STORE_NAME = MetadataStore.TRAINING_RUNS.value

class TrainingRunStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingRun(BaseModel):
    run_id: str
    modal_app_id: str
    framework: Framework
    config: Any
    dataset_id: str = ""
    deployment_id: str = ""
    status: TrainingRunStatus = TrainingRunStatus.RUNNING
    created_at: int = 0
    started_at: int = 0
    ended_at: int | None = None
    completed_at: int | None = None
    duration_seconds: int | None = None
    metadata: dict[str, Any] | None = None

    def save(self) -> None:
        payload = self.model_dump(mode="json")
        vol_put(MetadataStore.TRAINING_RUNS, self.run_id, payload)
        vol_upsert_summary_item(
            MetadataStore.TRAINING_RUNS_SUMMARY,
            payload,
            item_id_key="run_id",
            sort_key=lambda item: (
                int(item.get("created_at", 0) or 0),
                str(item.get("run_id", "")),
            ),
            reverse=True,
        )

    @classmethod
    def from_id(cls, run_id: str) -> "TrainingRun":
        return cls.model_validate(vol_get(MetadataStore.TRAINING_RUNS, run_id))

    @classmethod
    def list_runs(cls) -> list["TrainingRun"]:
        return [cls.model_validate(r) for r in vol_list(MetadataStore.TRAINING_RUNS)]
