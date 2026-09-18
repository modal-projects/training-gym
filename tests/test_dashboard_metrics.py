"""Metric-series ingestion, persistence and read API."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.metric_series import (
    CHUNK_STEPS,
    MetricPoint,
    downsample,
    load_chunk,
    merge_points,
    series_response,
)
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore

RUN_ID = "metric-run"
STORE = f"{MetadataStore.METRIC_SERIES.value}/{RUN_ID}"
AUTH = {"Authorization": "Bearer secret"}
WRITER = _dashboard.METRIC_WRITER_ID


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_run(status: TrainingRunStatus = TrainingRunStatus.RUNNING) -> None:
    TrainingRun(
        training_run_id=RUN_ID,
        modal_app_id="ap-metrics",
        framework=Framework.SLIME,
        config={"model": {"model_name": "Qwen/Qwen3-4B"}},
        created_at=100,
        started_at=100,
        updated_at=150,
        status=status,
    ).save()
    metadata.vol_put(MetadataStore.FRAMEWORK_STATUS_TOKENS, RUN_ID, {"token": "secret"})


def _batch(points, *, final=False, run_id=RUN_ID) -> dict:
    return {
        "training_run_id": run_id,
        "points": [{"step": step, "metrics": metrics} for step, metrics in points],
        "final": final,
    }


def _chunk_files(fake_volume) -> dict[str, dict]:
    prefix = f"{STORE}/"
    return {
        path.removeprefix(prefix): json.loads(data)
        for path, data in fake_volume.files.items()
        if path.startswith(prefix)
    }


def test_ingest_requires_the_run_token(fake_volume, monkeypatch, tmp_path):
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        anon = client.post("/api/metric-points", json=_batch([(0, {"a": 1})]))
        wrong = client.post(
            "/api/metric-points",
            json=_batch([(0, {"a": 1})]),
            headers={"Authorization": "Bearer nope"},
        )
        ghost = client.post(
            "/api/metric-points",
            json=_batch([(0, {"a": 1})], run_id="ghost"),
            headers=AUTH,
        )
        assert (anon.status_code, anon.json()) == (wrong.status_code, wrong.json())
        assert (anon.status_code, anon.json()) == (ghost.status_code, ghost.json())
        assert anon.status_code == 403
        assert client.get(f"/api/runs/{RUN_ID}/metrics").json()["keys"] == []


def test_ingest_buffers_in_memory_and_persists_chunks_on_final(
    fake_volume, monkeypatch, tmp_path
):
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        first = client.post(
            "/api/metric-points",
            json=_batch([(0, {"train/loss": 1.0}), (1, {"train/loss": 0.8})]),
            headers=AUTH,
        )
        assert first.json() == {"status": "ok", "accepted": 2}
        # Live reads come from memory; nothing has hit the volume yet.
        assert _chunk_files(fake_volume) == {}
        live = client.get(f"/api/runs/{RUN_ID}/metrics").json()
        assert live["keys"] == ["train/loss"]
        assert live["series"]["train/loss"] == [[0, 1.0], [1, 0.8]]
        assert live["latest"] == {"train/loss": 0.8}
        assert live["step_range"] == [0, 1]
        assert live["stale"] is False

        second = client.post(
            "/api/metric-points",
            json=_batch(
                [
                    (1, {"train/loss": 0.7, "reward": 0.5}),  # overwrite + new key
                    (CHUNK_STEPS + 3, {"reward": 0.9}),
                ],
                final=True,
            ),
            headers=AUTH,
        )
        assert second.status_code == 200
        files = _chunk_files(fake_volume)
        assert sorted(files) == [
            f"chunk-000000-{WRITER}.json",
            f"chunk-000001-{WRITER}.json",
        ]
        assert files[f"chunk-000000-{WRITER}.json"]["steps"]["1"] == {
            "train/loss": 0.7,
            "reward": 0.5,
        }
        assert files[f"chunk-000001-{WRITER}.json"]["steps"] == {
            str(CHUNK_STEPS + 3): {"reward": 0.9}
        }

        result = client.get(
            f"/api/runs/{RUN_ID}/metrics", params={"keys": "reward"}
        ).json()
        assert result["keys"] == ["reward", "train/loss"]
        assert list(result["series"]) == ["reward"]
        assert result["series"]["reward"] == [[1, 0.5], [CHUNK_STEPS + 3, 0.9]]
        assert result["point_count"] == 3


def test_reads_pick_up_chunks_written_by_another_replica(
    fake_volume, monkeypatch, tmp_path
):
    """Two replicas each write their own copy of a chunk; reads merge them."""
    _save_run()
    metadata.vol_put(
        STORE, "chunk-000000-aaaa", {"steps": {"3": {"lr": 0.1}, "bad": "skip"}}
    )
    metadata.vol_put(STORE, "chunk-000000-bbbb", {"steps": {"4": {"lr": 0.2}}})
    metadata.vol_put(STORE, "chunk-000002-bbbb", {})
    with _client(monkeypatch, tmp_path) as client:
        result = client.get(f"/api/runs/{RUN_ID}/metrics").json()
        assert result["series"] == {"lr": [[3, 0.1], [4, 0.2]]}
        client.post(
            "/api/metric-points",
            json=_batch([(5, {"lr": 0.3})], final=True),
            headers=AUTH,
        )
        files = _chunk_files(fake_volume)
        assert f"chunk-000000-{WRITER}.json" in files  # others left untouched
        assert files["chunk-000000-aaaa.json"]["steps"]["3"] == {"lr": 0.1}
        assert client.get("/api/runs/ghost/metrics").status_code == 404


def test_reading_a_finished_run_persists_whatever_is_still_buffered(
    fake_volume, monkeypatch, tmp_path
):
    """A container that dies mid-run never sends ``final``; the first read
    after the launcher marks the run terminal writes the buffered points."""
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        client.post("/api/metric-points", json=_batch([(0, {"a": 1.0})]), headers=AUTH)
        assert client.get(f"/api/runs/{RUN_ID}/metrics").status_code == 200
        assert _chunk_files(fake_volume) == {}
        _save_run(TrainingRunStatus.FAILED)
        result = client.get(f"/api/runs/{RUN_ID}/metrics").json()
        assert result["series"] == {"a": [[0, 1.0]]}
        assert list(_chunk_files(fake_volume)) == [f"chunk-000000-{WRITER}.json"]


def test_rejects_oversized_or_malformed_points(fake_volume, monkeypatch, tmp_path):
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        bad_step = client.post(
            "/api/metric-points",
            json={"training_run_id": RUN_ID, "points": [{"step": -1, "metrics": {}}]},
            headers=AUTH,
        )
        assert bad_step.status_code == 422
        bad_value = client.post(
            "/api/metric-points",
            json={
                "training_run_id": RUN_ID,
                "points": [{"step": 0, "metrics": {"a": "text"}}],
            },
            headers=AUTH,
        )
        assert bad_value.status_code == 422
        too_many = client.get(
            f"/api/runs/{RUN_ID}/metrics", params={"max_points": 10**9}
        )
        assert too_many.status_code == 422


# ── pure helpers ─────


def test_merge_and_load_prefer_newest_in_memory_values():
    table = {}
    touched = merge_points(
        table,
        [
            MetricPoint(step=1, metrics={"a": 1.0}),
            MetricPoint(step=1, metrics={"a": 2.0, "b": 3.0}),
            MetricPoint(step=2500, metrics={}),  # no metrics: ignored
        ],
    )
    assert touched == {"chunk-000000"}
    assert table == {1: {"a": 2.0, "b": 3.0}}
    load_chunk(table, {"steps": {"1": {"a": 99.0, "d": 5.0}, "7": {"a": 1}}})
    assert table == {1: {"a": 2.0, "b": 3.0, "d": 5.0}, 7: {"a": 1}}


def test_downsample_keeps_endpoints_and_extremes():
    rows = [[i, 0.0] for i in range(1000)]
    rows[437][1] = 50.0  # a spike
    rows[612][1] = -7.0  # a dip
    out = downsample(rows, 100)
    assert len(out) <= 100
    assert out[0] == rows[0] and out[-1] == rows[-1]
    assert rows[437] in out and rows[612] in out
    assert all(out[i][0] < out[i + 1][0] for i in range(len(out) - 1))
    assert downsample(rows[:50], 100) == rows[:50]
    assert downsample(rows, 2) == [rows[0], rows[-1]]
    assert downsample(rows, 1) == [rows[0]]


def test_series_response_filters_unknown_keys_and_downsamples():
    table = {}
    merge_points(
        table, [MetricPoint(step=s, metrics={"x": float(s)}) for s in range(10)]
    )
    out = series_response("r", table, keys=["x", "missing"], max_points=2)
    assert out["keys"] == ["x"]
    assert list(out["series"]) == ["x"]
    assert out["series"]["x"] == [[0, 0.0], [9, 9.0]]
    assert out["latest"] == {"x": 9.0}
    assert series_response("r", table, max_points=1)["latest"] == {"x": 9.0}
    assert out["step_range"] == [0, 9]
