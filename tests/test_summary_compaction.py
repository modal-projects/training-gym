"""Incremental summary compaction against the in-memory FakeVolume."""

from __future__ import annotations

import json
import threading

from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import (
    COMPACTION_KEY,
    MetadataStore,
    compact_summary_store,
    vol_get,
    vol_get_summary_items,
    vol_put,
    vol_put_summary_items,
)

RUNS = MetadataStore.TRAINING_RUNS
SUMMARY = MetadataStore.TRAINING_RUNS_SUMMARY


def _run(rid: str, created_at: int, **extra) -> dict:
    return {"training_run_id": rid, "created_at": created_at, **extra}


def _reads(fake_volume, monkeypatch) -> list[str]:
    seen: list[str] = []
    original = fake_volume._read_file

    def tracking(path: str):
        seen.append(path)
        return original(path)

    monkeypatch.setattr(fake_volume.read_file, "_sync_fn", tracking)
    return seen


def _canonical_reads(seen: list[str]) -> list[str]:
    return [p for p in seen if p.startswith(f"{RUNS.value}/")]


def test_first_compaction_reads_everything_and_records_watermark(
    fake_volume, monkeypatch
):
    for i in range(3):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))
    seen = _reads(fake_volume, monkeypatch)

    items = compact_summary_store(SUMMARY)

    assert [item["training_run_id"] for item in items] == ["run-2", "run-1", "run-0"]
    assert sorted(_canonical_reads(seen)) == sorted(
        f"{RUNS.value}/run-{i}.json" for i in range(3)
    )
    assert vol_get(SUMMARY, COMPACTION_KEY) == {
        "mtime": fake_volume.mtimes[f"{RUNS.value}/run-2.json"]
    }


def test_incremental_compaction_reads_only_changed_files(fake_volume, monkeypatch):
    for i in range(3):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))
    compact_summary_store(SUMMARY)

    vol_put(RUNS, "run-1", _run("run-1", 1, status="completed"))
    vol_put(RUNS, "run-3", _run("run-3", 3))
    seen = _reads(fake_volume, monkeypatch)

    items = compact_summary_store(SUMMARY)

    # run-2 sits exactly on the watermark, so it is re-read (inclusive bound
    # guards against same-second writes); run-0 is not.
    assert sorted(_canonical_reads(seen)) == [
        f"{RUNS.value}/run-1.json",
        f"{RUNS.value}/run-2.json",
        f"{RUNS.value}/run-3.json",
    ]
    by_id = {item["training_run_id"]: item for item in items}
    assert set(by_id) == {"run-0", "run-1", "run-2", "run-3"}
    assert by_id["run-1"]["status"] == "completed"


def test_incremental_compaction_only_rereads_watermark_file_when_unchanged(
    fake_volume, monkeypatch
):
    for i in range(3):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))
    compact_summary_store(SUMMARY)
    seen = _reads(fake_volume, monkeypatch)

    items = compact_summary_store(SUMMARY)

    assert _canonical_reads(seen) == [f"{RUNS.value}/run-2.json"]
    assert len(items) == 3


def test_incremental_compaction_heals_clobbered_summary(fake_volume, monkeypatch):
    for i in range(3):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))
    compact_summary_store(SUMMARY)

    # A racing read-modify-write upsert collapsed the summary to one item.
    vol_put_summary_items(SUMMARY, [_run("run-2", 2)])
    seen = _reads(fake_volume, monkeypatch)

    items = compact_summary_store(SUMMARY)

    assert sorted(_canonical_reads(seen)) == [
        f"{RUNS.value}/run-0.json",
        f"{RUNS.value}/run-1.json",
        f"{RUNS.value}/run-2.json",
    ]
    assert {item["training_run_id"] for item in items} == {"run-0", "run-1", "run-2"}
    assert {
        item["training_run_id"] for item in vol_get_summary_items(SUMMARY) or []
    } == {"run-0", "run-1", "run-2"}


def test_full_compaction_ignores_watermark(fake_volume, monkeypatch):
    for i in range(2):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))
    compact_summary_store(SUMMARY)
    seen = _reads(fake_volume, monkeypatch)

    compact_summary_store(SUMMARY, full=True)

    assert len(_canonical_reads(seen)) == 2


def test_corrupt_watermark_falls_back_to_full_read(fake_volume, monkeypatch):
    for i in range(2):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))
    compact_summary_store(SUMMARY)
    fake_volume.files[f"{SUMMARY.value}/{COMPACTION_KEY}.json"] = json.dumps(
        {"mtime": "not-an-int"}
    ).encode()
    seen = _reads(fake_volume, monkeypatch)

    compact_summary_store(SUMMARY)

    assert len(_canonical_reads(seen)) == 2


def test_sync_list_reads_files_concurrently(fake_volume, monkeypatch):
    for i in range(40):
        vol_put(RUNS, f"run-{i}", _run(f"run-{i}", i))

    active = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Event()
    original = fake_volume._read_file

    def slow_read(path: str):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active >= 2:
                barrier.set()
        barrier.wait(timeout=2)
        try:
            return original(path)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(fake_volume.read_file, "_sync_fn", slow_read)

    records = metadata.vol_list(RUNS)

    assert len(records) == 40
    assert peak > 1
