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


@pytest.mark.parametrize("is_async", [False, True])
def test_summary_upsert_survives_unreadable_summary_file(
    fake_volume, monkeypatch, is_async
):
    """Summary upserts rebuild from canonical items when the cache is unreadable."""
    summary_path = (
        f"{metadata._store_path(MetadataStore.TRAINING_RUNS_SUMMARY)}/summary.json"
    )
    original_read = fake_volume._read_file
    original_read_async = fake_volume._read_file_async

    def unreadable_read(path: str):
        if path == summary_path:
            from modal.exception import ExecutionError

            raise ExecutionError(
                "Request failed with status 404 Not Found: block not found"
            )
        return original_read(path)

    async def unreadable_read_async(path: str):
        if path == summary_path:
            from modal.exception import ExecutionError

            raise ExecutionError(
                "Request failed with status 404 Not Found: block not found"
            )
        async for chunk in original_read_async(path):
            yield chunk

    def put(*args, **kwargs):
        return metadata.vol_put(*args, is_async=is_async, **kwargs)

    def put_with_summary(*args, **kwargs):
        return metadata.vol_put_with_summary(*args, is_async=is_async, **kwargs)

    if is_async:
        asyncio.run(
            put(
                MetadataStore.TRAINING_RUNS,
                "run-a",
                {"training_run_id": "run-a"},
            )
        )
    else:
        put(
            MetadataStore.TRAINING_RUNS,
            "run-a",
            {"training_run_id": "run-a"},
        )

    with monkeypatch.context() as summary_monkeypatch:
        summary_monkeypatch.setattr(fake_volume, "_read_file", unreadable_read)
        summary_monkeypatch.setattr(
            fake_volume, "_read_file_async", unreadable_read_async
        )
        summary_monkeypatch.setattr(fake_volume.read_file, "_sync_fn", unreadable_read)
        summary_monkeypatch.setattr(fake_volume.read_file, "aio", unreadable_read_async)

        write = put_with_summary(
            MetadataStore.TRAINING_RUNS,
            "run-b",
            {"training_run_id": "run-b"},
            summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
            item_id_key="training_run_id",
        )
        if is_async:
            asyncio.run(write)

    if is_async:
        items = asyncio.run(
            metadata.vol_get_summary_items(
                MetadataStore.TRAINING_RUNS_SUMMARY, is_async=True
            )
        )
    else:
        items = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert {item["training_run_id"] for item in items or []} == {"run-a", "run-b"}


@pytest.mark.parametrize("fw", list(Framework))
def test_train_result_payload_is_json_serializable(fw):
    payload = TrainResult(app_name="a", framework=fw, training_run_id="t")._to_dict()
    assert json.loads(json.dumps(payload))["framework"] == fw.value


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
