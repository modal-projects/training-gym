from pydantic import BaseModel, ConfigDict, Field

from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put_with_summary,
)


class TrainingStep(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    training_run_id: str
    step: int = Field(ge=0)
    loss: float
    grad_norm: float | None = None
    learning_rate: float | None = None
    created_at: float

    def save(self) -> None:
        vol_put_with_summary(
            MetadataStore.TRAINING_STEPS,
            f"{self.training_run_id}__{self.step:08d}",
            self.model_dump(),
            summary_store=MetadataStore.TRAINING_STEPS_SUMMARY,
            summary_key=self.training_run_id,
            item_id_key="step",
            sort_key=lambda step: step["step"],
        )

    @classmethod
    def list_summaries_for_run(cls, training_run_id: str) -> list[dict]:
        steps = vol_get_summary_items(
            MetadataStore.TRAINING_STEPS_SUMMARY, key=training_run_id
        )
        return sorted(steps or [], key=lambda step: step["step"])
