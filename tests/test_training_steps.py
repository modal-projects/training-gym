from unittest.mock import Mock

from modal_training_gym.common.training_steps import TrainingStep
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


def test_training_steps_read_one_summary(fake_volume, monkeypatch):
    steps = [
        TrainingStep(
            training_run_id="sft", step=step, loss=1.0, created_at=step
        ).model_dump()
        for step in range(100)
    ]
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_STEPS_SUMMARY, steps[::-1], key="sft"
    )
    read = Mock(wraps=fake_volume.read_file)
    monkeypatch.setattr(fake_volume, "read_file", read)

    assert TrainingStep.list_summaries_for_run("sft") == steps
    read.assert_called_once_with("training-steps-summary/sft.json")


def test_training_steps_preserve_existing_history_when_saving(fake_volume):
    old = TrainingStep(training_run_id="sft", step=0, loss=2.0, created_at=1)
    old.save()
    other = TrainingStep(training_run_id="other", step=0, loss=3.0, created_at=1)
    other.save()
    new = TrainingStep(training_run_id="sft", step=1, loss=1.0, created_at=2)
    new.save()

    assert TrainingStep.list_summaries_for_run("sft") == [
        old.model_dump(),
        new.model_dump(),
    ]
    assert TrainingStep.list_summaries_for_run("other") == [other.model_dump()]

    new.loss = 0.5
    new.created_at = 3
    new.save()
    assert TrainingStep.list_summaries_for_run("sft") == [
        old.model_dump(),
        new.model_dump(),
    ]
    assert (
        metadata.vol_get(MetadataStore.TRAINING_STEPS, "sft__00000001")
        == new.model_dump()
    )


def test_training_steps_do_not_backfill_missing_summary(fake_volume):
    old = TrainingStep(training_run_id="sft", step=0, loss=2.0, created_at=1)
    metadata.vol_put(MetadataStore.TRAINING_STEPS, "sft__00000000", old.model_dump())
    files = fake_volume.files.copy()

    assert TrainingStep.list_summaries_for_run("sft") == []
    assert fake_volume.files == files
