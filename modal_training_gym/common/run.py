"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from dataclasses import asdict, dataclass

from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.frameworks.base import Framework
from modal_training_gym.frameworks.slime.config import SlimeConfig


TrainConfig = SlimeConfig # TODO: add other frameworks like `| Miles | MsSwift`

TRAINING_RUNS_STORE_NAME = "training-runs"


@dataclass
class TrainingRun:
    run_id: str
    modal_app_id: str
    framework: Framework
    config: TrainConfig # TODO: add other frameworks 

    def save(self) -> None:
        from modal import Dict
        store = Dict.from_name(TRAINING_RUNS_STORE_NAME, create_if_missing=True)
        store[self.run_id] = asdict(self)
    
    @classmethod
    def from_id(cls, run_id: str) -> "TrainingRun":
        from modal import Dict
        store = Dict.from_name(TRAINING_RUNS_STORE_NAME, create_if_missing=True)
        return cls(**store[run_id])
    
    @classmethod
    def list_runs(cls) -> list["TrainingRun"]:
        from modal import Dict
        store = Dict.from_name(TRAINING_RUNS_STORE_NAME, create_if_missing=True)
        return [cls(**run) for run in store.values()]