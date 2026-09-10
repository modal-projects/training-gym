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
def test_data_and_summary_are_persisted(fake_volume, is_async):
    result = _rollout()
    if is_async:
        asyncio.run(result.save(is_async=True))
    else:
        result.save()
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
    assert "training-rollouts-summary/run/run__00000000.json" in paths
    assert all(path.startswith("training-rollouts-summary/") for path in paths)
    assert not any("/other/" in path for path in paths)


@pytest.mark.parametrize("is_async", [False, True])
def test_overwrite_commits_data_and_summary_together(
    fake_volume, monkeypatch, is_async
):
    batches = []
    fail = False

    class Batch:
        def __init__(self, force=False):
            self.files = {}
            batches.append(self.files)

        def __enter__(self):
            return self

        def put_file(self, fileobj, path):
            self.files[path] = fileobj.read()

        def __exit__(self, *args):
            if fail:
                raise OSError("commit failed")
            fake_volume.files.update(self.files)

        async def __aenter__(self):
            return self.__enter__()

        async def __aexit__(self, *args):
            return self.__exit__(*args)

    def save(result):
        if is_async:
            asyncio.run(result.save(is_async=True))
        else:
            result.save()

    monkeypatch.setattr(fake_volume, "batch_upload", Batch)
    result = _rollout()
    save(result)
    assert len(batches) == 1
    assert len(batches[0]) == 2
    before = dict(fake_volume.files)
    result.samples[0].score = 1.0
    fail = True
    with pytest.raises(OSError, match="commit failed"):
        save(result)
    assert fake_volume.files == before
    fail = False
    save(result)
    assert len(batches) == 3
    assert TrainingRolloutResult.list_summaries_for_run("run")[0].mean == 1.0
    assert (
        metadata.vol_get(MetadataStore.TRAINING_ROLLOUTS, result.storage_key)[
            "samples"
        ][0]["score"]
        == 1.0
    )


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
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
        [_rollout().to_summary(), _rollout(run_id="other").to_summary()],
    )

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


def test_legacy_summary_and_canonical_gaps_are_recovered_once(fake_volume, monkeypatch):
    known, hidden = _rollout(step=12), _rollout(step=13)
    for result in (known, hidden):
        metadata.vol_put(
            MetadataStore.TRAINING_ROLLOUTS,
            result.storage_key,
            result.model_dump(mode="json"),
        )
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
        [known.to_summary(), _rollout(run_id="other").to_summary()],
    )
    read = fake_volume.read_file._sync_fn
    paths = []

    def track_read(path):
        paths.append(path)
        return read(path)

    monkeypatch.setattr(fake_volume.read_file, "_sync_fn", track_read)
    summaries = TrainingRolloutResult.list_summaries_for_run("run")
    assert [s.rollout_id for s in summaries] == [12, 13]
    assert summaries[1].export_size_bytes > 0
    assert [p for p in paths if p.startswith("training-rollouts/")] == [
        "training-rollouts/run__00000013.json"
    ]
    paths.clear()
    assert TrainingRolloutResult.list_summaries_for_run("run") == summaries
    assert not any(p.startswith("training-rollouts/") for p in paths)
    assert "training-rollouts-summary/summary.json" not in paths


def test_new_summaries_win_over_legacy_and_legacy_rows_remain_visible(fake_volume):
    result = _rollout(step=13)
    result.save()
    older = _rollout(step=12)
    metadata.vol_put(
        MetadataStore.TRAINING_ROLLOUTS,
        older.storage_key,
        older.model_dump(mode="json"),
    )
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
        [
            {**result.to_summary(), "mean": 99},
            _rollout(step=12).to_summary(),
            {"training_run_id": "run"},
        ],
    )
    summaries = TrainingRolloutResult.list_summaries_for_run("run")
    assert [s.rollout_id for s in summaries] == [12, 13]
    assert summaries[1].mean == 0.5


def test_missing_legacy_summary_recovers_canonical_and_discovers_later_steps(
    fake_volume,
):
    result = _rollout(step=13)
    metadata.vol_put(
        MetadataStore.TRAINING_ROLLOUTS,
        result.storage_key,
        result.model_dump(mode="json"),
    )
    assert [
        s.rollout_id for s in TrainingRolloutResult.list_summaries_for_run("run")
    ] == [13]
    later = _rollout(step=18)
    metadata.vol_put(
        MetadataStore.TRAINING_ROLLOUTS,
        later.storage_key,
        later.model_dump(mode="json"),
    )
    assert [
        s.rollout_id for s in TrainingRolloutResult.list_summaries_for_run("run")
    ] == [13, 18]


def test_legacy_only_summary_is_cached(fake_volume, monkeypatch):
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_ROLLOUTS_SUMMARY, [_rollout(step=12).to_summary()]
    )
    assert [
        s.rollout_id for s in TrainingRolloutResult.list_summaries_for_run("run")
    ] == [12]
    from modal_training_gym.common import training_rollout

    def unexpected_legacy_read(*args, **kwargs):
        pytest.fail("legacy index was reread")

    monkeypatch.setattr(
        training_rollout, "vol_get_summary_items", unexpected_legacy_read
    )
    assert [
        s.rollout_id for s in TrainingRolloutResult.list_summaries_for_run("run")
    ] == [12]


def test_recovery_ignores_bad_records_and_survives_cache_failure(
    fake_volume, monkeypatch
):
    result = _rollout(step=13)
    metadata.vol_put(
        MetadataStore.TRAINING_ROLLOUTS,
        result.storage_key,
        result.model_dump(mode="json"),
    )
    metadata.vol_put(
        MetadataStore.TRAINING_ROLLOUTS, "run__00000018", {"bad": "record"}
    )
    from modal_training_gym.common import training_rollout

    def fail_cache(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(training_rollout, "vol_put_many", fail_cache)
    assert [
        s.rollout_id for s in TrainingRolloutResult.list_summaries_for_run("run")
    ] == [13]
