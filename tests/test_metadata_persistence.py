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
from modal_training_gym.utils.metadata import MetadataStore, vol_count_items


@pytest.mark.parametrize("fw", list(Framework))
def test_training_run_save_survives_unmounted_volume(fake_volume, fw):
    """TrainingRun.save() completes when reload() raises, for every framework."""
    run_mod.TrainingRun(training_run_id="t1", framework=fw, config={}).save()

    blob = fake_volume.files[f"{MetadataStore.TRAINING_RUNS.value}/t1.json"]
    assert json.loads(blob)["framework"] == fw.value


def test_count_items_ignores_files_nested_under_the_store(fake_volume):
    """Only ``<store>/<id>.json`` counts, so the summary can catch up with it.

    A file nested deeper never reaches a summary, so counting it would hold the
    count permanently above the summary and make every reader rebuild.
    """
    run_mod.TrainingRun(
        training_run_id="t1", framework=Framework.SLIME, config={}
    ).save()
    store = MetadataStore.TRAINING_RUNS.value
    fake_volume.files[f"{store}/Qwen/Qwen3-4B/notes.json"] = b"{}"

    assert vol_count_items(MetadataStore.TRAINING_RUNS) == 1


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
