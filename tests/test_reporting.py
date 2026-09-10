from __future__ import annotations

import json
from http.client import IncompleteRead
from io import BytesIO
from queue import Queue
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from modal_training_gym.common import reporting


@pytest.fixture(autouse=True)
def isolated_reporter(monkeypatch):
    from modal_training_gym.common import config

    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_DEADLINE", None)
    monkeypatch.setattr(reporting, "_REPORT_QUEUE", Queue())
    monkeypatch.setattr(reporting, "_ensure_worker", lambda: None)
    monkeypatch.setattr(config, "modal_proxy_auth_headers", lambda: {})
    monkeypatch.setattr(reporting, "_report_token", lambda: "test-token")
    monkeypatch.setenv(
        "TRAINING_GYM_FRAMEWORK_STATUS_URL",
        "https://dashboard.test/api/framework-status",
    )


def _item(kind="rollout", text="text"):
    enqueue = (
        reporting._enqueue_rollout
        if kind == "rollout"
        else reporting._enqueue_advantage
    )
    enqueue(
        {"training_run_id": "run", "rollout_id": 13, "samples": [{"response": text}]}
    )
    item = reporting._REPORT_QUEUE.get_nowait()
    reporting._REPORT_QUEUE.task_done()
    return item


@pytest.mark.parametrize("kind", ["rollout", "advantage"])
@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        URLError("reset"),
        IncompleteRead(b""),
        408,
        429,
        500,
        502,
        503,
        504,
    ],
)
def test_transient_upload_failure_retries_same_payload(monkeypatch, kind, failure):
    calls = []
    sleeps = []

    def send(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            if isinstance(failure, int):
                raise HTTPError(request.full_url, failure, "unavailable", {}, None)
            raise failure
        return BytesIO(b"ok")

    monkeypatch.setattr(reporting, "urlopen", send)
    monkeypatch.setattr(reporting.time, "sleep", sleeps.append)
    assert reporting._post_record(_item(kind))
    assert len(calls) == 2
    assert calls[0][0].data == calls[1][0].data
    assert calls[0][0].get_header("Authorization") == "Bearer test-token"
    assert not any(k.startswith("_") for k in json.loads(calls[0][0].data))
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "status, attempts", [(401, 1), (403, 1), (413, 1), (422, 1), (503, 3)]
)
@pytest.mark.parametrize("kind", ["rollout", "advantage"])
def test_failures_are_bounded_and_visible(monkeypatch, capsys, status, attempts, kind):
    calls = []

    def send(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, status, "failed", {}, None)

    monkeypatch.setattr(reporting, "urlopen", send)
    monkeypatch.setattr(reporting.time, "sleep", lambda _: None)
    item = _item(kind, text="private sample contents")
    reporting._REPORT_QUEUE.put(item)
    reporting._REPORT_QUEUE.put(None)
    reporting._worker()
    assert len(calls) == attempts
    assert reporting._REPORT_QUEUE.unfinished_tasks == 0
    output = capsys.readouterr().out
    assert f"{kind} upload for run run step 13" in output
    assert f"http_status={status}" in output
    assert "private sample contents" not in output


@pytest.mark.parametrize(
    "size, expected", [(10, 10), (5_000_000, 20), (31_000_000, 120)]
)
def test_timeout_scales_with_payload_and_is_capped(monkeypatch, size, expected):
    timeouts = []

    def send(request, timeout):
        timeouts.append(timeout)
        return BytesIO(b"ok")

    monkeypatch.setattr(reporting, "urlopen", send)
    assert reporting._post_record(_item(text="x" * size))
    assert timeouts[0] == pytest.approx(expected, abs=0.01)


def test_drain_preserves_large_upload_timeout_within_deadline(monkeypatch):
    item = _item(text="x" * 5_000_000)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", True)
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_DEADLINE", 130.0)
    monkeypatch.setattr(reporting.time, "monotonic", lambda: 100.0)
    timeouts = []

    def send(request, timeout):
        timeouts.append(timeout)
        return BytesIO(b"ok")

    monkeypatch.setattr(reporting, "urlopen", send)
    assert reporting._post_record(item)
    assert 20 < timeouts[0] < 21
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_DEADLINE", 105.0)
    assert reporting._post_record(item)
    assert timeouts[-1] == 5


def test_drain_stops_retrying_at_deadline(monkeypatch):
    item = _item()
    now = [100.0]
    calls = []
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", True)
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_DEADLINE", 100.5)
    monkeypatch.setattr(reporting.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        reporting.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )

    def send(request, timeout):
        calls.append(timeout)
        raise TimeoutError()

    monkeypatch.setattr(reporting, "urlopen", send)
    assert not reporting._post_record(item)
    assert calls == [0.5]
    assert now[0] == 100.5


def test_shutdown_budget_allows_full_large_record_retries(monkeypatch):
    item = _item(text="x" * 31_000_000)
    thread = Mock()
    now = [100.0]
    monkeypatch.setattr(reporting.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        reporting.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )
    monkeypatch.setattr(reporting, "_REPORTER_THREAD", thread)
    monkeypatch.setattr(reporting, "_REPORTER_STARTED", True)
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_SENTINEL_QUEUED", False)
    monkeypatch.setattr(reporting, "_run_pre_drain_hooks", lambda: None)
    reporting._drain_report_queue()
    thread.join.assert_called_once_with(timeout=reporting.REPORT_DRAIN_TIMEOUT_SECONDS)
    now[0] += (
        reporting.REPORT_DRAIN_FINAL_POST_TIMEOUT_SECONDS
        * reporting.REPORT_DRAIN_FINAL_POST_COUNT
    )
    timeouts = []

    def send(request, timeout):
        timeouts.append(timeout)
        now[0] += timeout
        raise TimeoutError()

    monkeypatch.setattr(reporting, "urlopen", send)
    assert not reporting._post_record(item)
    assert timeouts == [120.0, 120.0, 120.0]
    assert now[0] <= reporting._REPORT_DRAIN_DEADLINE


@pytest.mark.parametrize("kind", ["rollout", "advantage"])
def test_queue_full_is_visible(monkeypatch, capsys, kind):
    queue = Queue(maxsize=1)
    queue.put({})
    monkeypatch.setattr(reporting, "_REPORT_QUEUE", queue)
    enqueue = (
        reporting._enqueue_rollout
        if kind == "rollout"
        else reporting._enqueue_advantage
    )
    enqueue({"training_run_id": "run", "rollout_id": 13})
    assert f"{kind} upload for run run step 13" in capsys.readouterr().out
    assert queue.qsize() == 1
