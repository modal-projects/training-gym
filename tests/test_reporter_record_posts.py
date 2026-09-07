"""Rollout records are large; their upload budget scales and drops are visible."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from modal_training_gym.common import reporting


@pytest.fixture
def capture_urlopen(monkeypatch):
    calls: list[dict] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        calls.append({"timeout": timeout, "bytes": len(request.data)})
        return _Response()

    monkeypatch.setattr(reporting, "urlopen", fake_urlopen)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    return calls


def test_rollout_post_timeout_grows_with_payload_size(capture_urlopen) -> None:
    samples = [{"prompt": "p", "response": "x" * 300_000} for _ in range(32)]
    item = {
        "_url": "http://dashboard/api/training-rollouts",
        "_timeout": reporting._ROLLOUT_TIMEOUT_SECONDS,
        "_scale_timeout_with_size": True,
        "training_run_id": "run",
        "rollout_id": 5,
        "samples": samples,
    }

    assert reporting._post(item) is True

    call = capture_urlopen[0]
    expected = call["bytes"] / 1e6 * reporting._TIMEOUT_SECONDS_PER_MB
    assert call["bytes"] > 9_000_000
    assert call["timeout"] == pytest.approx(expected)
    assert call["timeout"] > reporting._ROLLOUT_TIMEOUT_SECONDS


def test_small_rollout_post_keeps_its_base_timeout(capture_urlopen) -> None:
    item = {
        "_url": "http://dashboard/api/training-rollouts",
        "_timeout": reporting._ROLLOUT_TIMEOUT_SECONDS,
        "_scale_timeout_with_size": True,
        "training_run_id": "run",
        "rollout_id": 0,
        "samples": [{"prompt": "p", "response": "r"}],
    }

    assert reporting._post(item) is True
    assert capture_urlopen[0]["timeout"] == reporting._ROLLOUT_TIMEOUT_SECONDS


def test_phase_posts_are_not_scaled(capture_urlopen) -> None:
    item = {
        "_url": "http://dashboard/api/status",
        "_timeout": 1.0,
        "phase": "x" * 5_000_000,
    }

    reporting._post(item)

    assert capture_urlopen[0]["timeout"] == 1.0


def test_dropped_rollout_record_is_reported(monkeypatch, capsys) -> None:
    def failing_urlopen(request, timeout):
        raise HTTPError(request.full_url, 500, "boom", {}, None)

    monkeypatch.setattr(reporting, "urlopen", failing_urlopen)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    item = {
        "_url": "http://dashboard/api/training-rollouts",
        "_timeout": 10.0,
        "training_run_id": "run",
        "rollout_id": 5,
        "samples": [{"prompt": "p", "response": "r"}],
    }

    assert reporting._post(item) is False
    reporting._warn_if_record_dropped(item)

    out = capsys.readouterr().out
    assert "rollout record for step 5 was not stored" in out
    assert "http_status=500" in out


def test_phase_payload_failures_stay_quiet(capsys) -> None:
    reporting._warn_if_record_dropped(
        {"rollout_id": 3, "phase": "rollout", "_failure_reason": {}}
    )
    assert capsys.readouterr().out == ""


def test_enqueued_rollout_is_marked_for_size_scaling(monkeypatch) -> None:
    queued: list[dict] = []
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    monkeypatch.setattr(
        reporting, "_rollout_url", lambda: "http://dashboard/api/training-rollouts"
    )
    monkeypatch.setattr(reporting, "_ensure_worker", lambda *a, **k: None)
    monkeypatch.setattr(reporting._REPORT_QUEUE, "put_nowait", queued.append)

    reporting._enqueue_rollout(
        {"training_run_id": "run", "rollout_id": 1, "samples": []}
    )

    assert queued[0]["_scale_timeout_with_size"] is True
    assert queued[0]["_timeout"] == reporting._ROLLOUT_TIMEOUT_SECONDS
    assert json.dumps({k: v for k, v in queued[0].items() if not k.startswith("_")})
