"""Drive the real ``save()`` chain to completion without Modal or a GPU.

``TrainingRun.save()``, ``TrainResult.save()``, and
``TrainingRolloutResult.save(is_async=True)`` are exercised against an in-memory
``FakeVolume`` (see ``conftest.py``) so the full serialize-and-write path is
covered in CI: the payload must stay JSON-serializable, and ``save()`` (in both
sync and ``is_async=True`` modes) must complete even when ``Volume.reload()``
is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from modal_training_gym.common import run as run_mod
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.training_rollout import TrainingRolloutResult
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


@pytest.mark.parametrize("fw", list(Framework))
def test_training_run_save_survives_unmounted_volume(fake_volume, fw):
    """TrainingRun.save() completes when reload() raises, for every framework."""
    run_mod.TrainingRun(training_run_id="t1", framework=fw, config={}).save()

    blob = fake_volume.files[f"{MetadataStore.TRAINING_RUNS.value}/t1.json"]
    assert json.loads(blob)["framework"] == fw.value


@pytest.mark.parametrize("fw", list(Framework))
def test_train_result_save_survives_unmounted_volume(fake_volume, fw):
    """TrainResult.save() completes when reload() raises, for every framework."""
    TrainResult(app_name="a", framework=fw, training_run_id="t2").save()

    blob = fake_volume.files[f"{MetadataStore.TRAIN_RESULTS.value}/t2.json"]
    assert json.loads(blob)["framework"] == fw.value


def test_rollout_async_save_survives_unmounted_volume(fake_volume):
    """TrainingRolloutResult.save(is_async=True) completes when reload() raises."""
    asyncio.run(
        TrainingRolloutResult(
            training_run_id="t3",
            rollout_id=0,
            samples=[{"score": 1.0, "prompt": "p", "response": "r"}],
        ).save(is_async=True)
    )

    blob = fake_volume.files[
        f"{MetadataStore.TRAINING_ROLLOUTS.value}/t3__00000000.json"
    ]
    assert json.loads(blob)["rollout_id"] == 0
    summary = fake_volume.files[
        f"{MetadataStore.TRAINING_ROLLOUTS_SUMMARY.value}/summary.json"
    ]
    summary_item = json.loads(summary)["items"][0]
    assert summary_item["summary_key"] == "t3__00000000"
    assert summary_item["export_size_bytes"] == len(
        (json.dumps(json.loads(blob), ensure_ascii=False, indent=2) + "\n").encode()
    )


@pytest.mark.parametrize("fw", list(Framework))
def test_train_result_payload_is_json_serializable(fw):
    payload = TrainResult(app_name="a", framework=fw, training_run_id="t")._to_dict()
    assert json.loads(json.dumps(payload))["framework"] == fw.value


def _put_json(fake_volume, path: str, payload: object) -> None:
    fake_volume.files[path] = (json.dumps(payload) + "\n").encode()


def test_summary_heal_reads_only_missing_canonical_items(fake_volume, monkeypatch):
    canonical_store = MetadataStore.TRAINING_RUNS.value
    summary_store = MetadataStore.TRAINING_RUNS_SUMMARY.value
    for run_id in ("run-1", "run-2", "run-3"):
        _put_json(
            fake_volume,
            f"{canonical_store}/{run_id}.json",
            {"training_run_id": run_id, "created_at": 1},
        )
    _put_json(
        fake_volume,
        f"{summary_store}/summary.json",
        {
            "items": [
                {"training_run_id": "run-1", "created_at": 1},
                {"training_run_id": "run-2", "created_at": 1},
            ]
        },
    )

    read_paths: list[str] = []
    original_read_file = fake_volume.read_file

    def read_file(path: str):
        read_paths.append(path)
        return original_read_file(path)

    monkeypatch.setattr(fake_volume, "read_file", read_file)
    repaired = metadata.vol_get_summary_items_healed(
        MetadataStore.TRAINING_RUNS_SUMMARY
    )

    assert {item["training_run_id"] for item in repaired} == {
        "run-1",
        "run-2",
        "run-3",
    }
    assert [path for path in read_paths if path.startswith(f"{canonical_store}/")] == [
        f"{canonical_store}/run-3.json"
    ]


def test_summary_heal_without_drift_reads_no_canonical_items(fake_volume, monkeypatch):
    canonical_store = MetadataStore.TRAINING_RUNS.value
    summary_store = MetadataStore.TRAINING_RUNS_SUMMARY.value
    for run_id in ("run-1", "run-2"):
        _put_json(
            fake_volume,
            f"{canonical_store}/{run_id}.json",
            {"training_run_id": run_id, "created_at": 1},
        )
    _put_json(
        fake_volume,
        f"{summary_store}/summary.json",
        {
            "items": [
                {"training_run_id": "run-1", "created_at": 1},
                {"training_run_id": "run-2", "created_at": 1},
            ]
        },
    )

    read_paths: list[str] = []
    original_read_file = fake_volume.read_file

    def read_file(path: str):
        read_paths.append(path)
        return original_read_file(path)

    monkeypatch.setattr(fake_volume, "read_file", read_file)
    metadata.vol_get_summary_items_healed(MetadataStore.TRAINING_RUNS_SUMMARY)

    assert not [path for path in read_paths if path.startswith(f"{canonical_store}/")]


def test_summary_heal_drops_stale_summary_items(fake_volume):
    canonical_store = MetadataStore.TRAINING_RUNS.value
    summary_store = MetadataStore.TRAINING_RUNS_SUMMARY.value
    _put_json(
        fake_volume,
        f"{canonical_store}/run-1.json",
        {"training_run_id": "run-1", "created_at": 1},
    )
    _put_json(
        fake_volume,
        f"{summary_store}/summary.json",
        {
            "items": [
                {"training_run_id": "run-1", "created_at": 1},
                {"training_run_id": "gone", "created_at": 2},
            ]
        },
    )

    repaired = metadata.vol_get_summary_items_healed(
        MetadataStore.TRAINING_RUNS_SUMMARY
    )

    assert repaired == [{"training_run_id": "run-1", "created_at": 1}]
    stored = json.loads(fake_volume.files[f"{summary_store}/summary.json"])
    assert stored["items"] == repaired


@pytest.mark.parametrize("canonical_keys", [set(), None])
def test_summary_heal_ignores_empty_or_failed_canonical_listing(
    fake_volume, monkeypatch, canonical_keys
):
    summary_store = MetadataStore.TRAINING_RUNS_SUMMARY.value
    original_summary = {"items": [{"training_run_id": "run-1", "created_at": 1}]}
    _put_json(fake_volume, f"{summary_store}/summary.json", original_summary)
    monkeypatch.setattr(metadata, "vol_list_item_keys", lambda _store: canonical_keys)
    monkeypatch.setattr(
        metadata,
        "vol_put_summary_items",
        lambda *_args, **_kwargs: pytest.fail("summary should not be rewritten"),
    )

    assert (
        metadata.vol_get_summary_items_healed(MetadataStore.TRAINING_RUNS_SUMMARY)
        == original_summary["items"]
    )
    assert (
        json.loads(fake_volume.files[f"{summary_store}/summary.json"])
        == original_summary
    )


def test_summary_heal_falls_back_to_compaction_on_incremental_failure(
    fake_volume, monkeypatch
):
    canonical_store = MetadataStore.TRAINING_RUNS.value
    summary_store = MetadataStore.TRAINING_RUNS_SUMMARY.value
    for run_id in ("run-1", "run-2"):
        _put_json(
            fake_volume,
            f"{canonical_store}/{run_id}.json",
            {"training_run_id": run_id, "created_at": 1},
        )
    _put_json(
        fake_volume,
        f"{summary_store}/summary.json",
        {"items": [{"training_run_id": "run-1", "created_at": 1}]},
    )
    fallback = [{"training_run_id": "from-compaction"}]
    monkeypatch.setattr(
        metadata,
        "_read_metadata_records",
        lambda _entries: ([], RuntimeError("read failed")),
    )
    monkeypatch.setattr(
        metadata, "compact_summary_store", lambda _summary_store: fallback
    )

    assert (
        metadata.vol_get_summary_items_healed(MetadataStore.TRAINING_RUNS_SUMMARY)
        == fallback
    )


@pytest.mark.skipif(
    os.environ.get("RUN_MODAL_TESTS") != "1",
    reason="hits Modal (no GPU); opt in with RUN_MODAL_TESTS=1",
)
def test_remote_save_from_unmounted_container():
    """The faithful remote counterpart: run save() inside a real Modal container
    that does *not* mount the metadata volume — the exact context where the
    original training run crashed with `volume … not attached`. The fake-volume
    tests simulate that; this proves it against real Modal Volume semantics
    (reload unavailable, but the client-side write still lands).
    """
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("modal>=1.4.0", "pydantic")
        .add_local_python_source("modal_training_gym")
    )
    app = modal.App("training-gym-metadata-save-probe")

    # NB: deliberately no volumes= — this is the unmounted case.
    @app.function(image=image, serialized=True)
    def _save_probe() -> str:
        from modal_training_gym.common.framework import Framework
        from modal_training_gym.common.run import TrainingRun
        from modal_training_gym.common.train_result import TrainResult

        # Framework is incidental here — this probes volume/reload mechanics,
        # which are framework-independent. Per-framework serialization is
        # covered by the fast parametrized tests above, so we don't pay for N
        # real Modal apps to re-check it.
        rid = "ci-remote-save-probe"  # fixed id → overwrites, no junk accrual
        TrainingRun(training_run_id=rid, framework=Framework.SLIME, config={}).save()
        TrainResult(app_name=rid, framework=Framework.SLIME, training_run_id=rid).save()
        return "ok"

    with modal.enable_output():
        with app.run():
            assert _save_probe.remote() == "ok"
