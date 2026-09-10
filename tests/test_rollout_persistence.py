from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from modal_training_gym.common.training_rollout import TrainingRolloutResult
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


def _rollout(run_id="run", step=0):
    return TrainingRolloutResult(
        training_run_id=run_id,
        rollout_id=step,
        samples=[{"prompt": "p", "response": "r", "score": 0.5}],
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_data_and_summary_share_one_batch(fake_volume, monkeypatch, is_async):
    batches = []
    upload = fake_volume.batch_upload

    def record_batch(force=False):
        batches.append(force)
        return upload(force=force)

    monkeypatch.setattr(fake_volume, "batch_upload", record_batch)
    result = _rollout()
    if is_async:
        asyncio.run(result.save(is_async=True))
    else:
        result.save()
    assert batches == [True]
    assert set(fake_volume.files) == {
        "training-rollouts/run__00000000.json",
        "training-rollouts-summary/run/run__00000000.json",
    }
    assert TrainingRolloutResult.list_summaries_for_run("run")[0].mean == 0.5


def test_concurrent_steps_keep_all_summaries(fake_volume, monkeypatch):
    # Hold every writer just before committing to reproduce overlapping writes.
    barrier = Barrier(8)
    upload = fake_volume.batch_upload

    def overlapping_batch(force=False):
        batch = upload(force=force)

        class Batch:
            def __enter__(self):
                barrier.wait(timeout=10)
                return batch.__enter__()

            def __exit__(self, *args):
                return batch.__exit__(*args)

        return Batch()

    monkeypatch.setattr(fake_volume, "batch_upload", overlapping_batch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda step: _rollout(step=step).save(), range(8)))

    monkeypatch.setattr(fake_volume, "batch_upload", upload)
    _rollout(run_id="other", step=3).save()
    # A retry replaces its own files; it cannot duplicate or erase another step.
    _rollout(step=3).save()
    assert [
        s.rollout_id for s in TrainingRolloutResult.list_summaries_for_run("run")
    ] == list(range(8))
    assert len(TrainingRolloutResult.list_summaries_for_run("other")) == 1


def test_listing_reads_only_small_summaries(fake_volume, monkeypatch):
    _rollout().save()
    _rollout(run_id="other").save()
    read = fake_volume.read_file._sync_fn
    paths = []

    def read_summary(path):
        paths.append(path)
        return read(path)

    monkeypatch.setattr(fake_volume.read_file, "_sync_fn", read_summary)
    assert len(TrainingRolloutResult.list_summaries_for_run("run")) == 1
    assert paths == ["training-rollouts-summary/run/run__00000000.json"]


def test_failed_batch_is_not_acknowledged(fake_volume, monkeypatch):
    def fail_commit(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(fake_volume, "batch_upload", fail_commit)
    with pytest.raises(OSError, match="storage unavailable"):
        _rollout().save()


def test_listing_skips_invalid_summary(fake_volume):
    _rollout().save()
    metadata.vol_put(
        TrainingRolloutResult.summary_store("run"),
        "invalid",
        {"training_run_id": "run"},
    )
    assert len(TrainingRolloutResult.list_summaries_for_run("run")) == 1


def test_cleanup_removes_individual_summaries(fake_volume, monkeypatch):
    from modal_training_gym.cli.cleanup import cleanup
    from modal_training_gym.common.run import TrainingRun, TrainingRunStatus

    TrainingRun(
        training_run_id="run",
        framework="slime",
        status=TrainingRunStatus.FAILED,
        created_at=1,
        config={},
    ).save()
    _rollout().save()
    _rollout(run_id="other").save()
    remove = fake_volume.remove_file

    def remove_file(path, *, recursive=False):
        if recursive:
            raise FileNotFoundError(path)
        return remove(path)

    monkeypatch.setattr(fake_volume, "remove_file", remove_file)
    cleanup(older_than_days=1)
    assert TrainingRolloutResult.list_summaries_for_run("run") == []
    assert len(TrainingRolloutResult.list_summaries_for_run("other")) == 1
    assert all(
        json.loads(blob)["training_run_id"] == "other"
        for path, blob in fake_volume.files.items()
        if path.startswith(f"{MetadataStore.TRAINING_ROLLOUTS.value}/")
    )
