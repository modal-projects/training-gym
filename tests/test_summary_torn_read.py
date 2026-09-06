"""A summary file caught mid-rewrite must not abort the writer that reads it."""

from __future__ import annotations

import asyncio
import json

import pytest

from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put,
    vol_put_with_summary,
)

SUMMARY_PATH = f"{MetadataStore.TRAINING_RUNS_SUMMARY.value}/summary.json"


def _truncated(items: list[dict]) -> bytes:
    whole = json.dumps({"items": items}).encode()
    return whole[: len(whole) // 2]


def test_persistently_torn_summary_reads_as_absent(fake_volume) -> None:
    fake_volume.files[SUMMARY_PATH] = _truncated([{"training_run_id": "old"}])

    assert vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) is None
    assert (
        asyncio.run(
            vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY, is_async=True)
        )
        is None
    )


class _FlakyRead:
    """A Volume ``read_file`` that returns truncated bytes once, then the file."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files
        self.calls = 0

    def _payload(self, path: str) -> bytes:
        self.calls += 1
        data = self._files[path]
        return data[: len(data) // 2] if self.calls == 1 else data

    def __call__(self, path: str):
        return [self._payload(path)]

    async def aio(self, path: str):
        yield self._payload(path)


def test_torn_summary_read_heals_on_retry(fake_volume) -> None:
    fake_volume.files[SUMMARY_PATH] = json.dumps(
        {"items": [{"training_run_id": "old"}]}
    ).encode()
    flaky = _FlakyRead(fake_volume.files)
    fake_volume.read_file = flaky

    assert vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) == [
        {"training_run_id": "old"}
    ]
    assert flaky.calls == 2


def test_upsert_rebuilds_a_torn_summary_from_canonical_files(fake_volume) -> None:
    vol_put(
        MetadataStore.TRAINING_RUNS,
        "old",
        {"training_run_id": "old", "status": "completed"},
    )
    fake_volume.files[SUMMARY_PATH] = _truncated([{"training_run_id": "old"}])

    vol_put_with_summary(
        MetadataStore.TRAINING_RUNS,
        "new",
        {"training_run_id": "new", "status": "running"},
        summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
        item_id_key="training_run_id",
    )

    items = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert items is not None
    assert sorted(item["training_run_id"] for item in items) == ["new", "old"]


@pytest.mark.parametrize("is_async", [False, True])
def test_valid_summary_is_read_unchanged(fake_volume, is_async) -> None:
    fake_volume.files[SUMMARY_PATH] = json.dumps(
        {"items": [{"training_run_id": "a"}, {"training_run_id": "b"}]}
    ).encode()
    result = vol_get_summary_items(
        MetadataStore.TRAINING_RUNS_SUMMARY, is_async=is_async
    )
    if is_async:
        result = asyncio.run(result)
    assert result == [{"training_run_id": "a"}, {"training_run_id": "b"}]
