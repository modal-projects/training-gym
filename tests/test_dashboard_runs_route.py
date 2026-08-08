from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.training_rollout import TrainingRolloutResult
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    (static / "apple-touch-icon.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    )
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_records() -> None:
    TrainingRun(
        training_run_id="run-route-1",
        modal_app_id="ap-route",
        framework=Framework.SLIME,
        config={
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "dataset": {"hf_repo": "openai/gsm8k"},
            "recipe": {"gpu_type": "H100"},
        },
        created_at=100,
        started_at=100,
        updated_at=150,
        metadata={"group_id": "route-group"},
    ).save()
    TrainResult(
        app_name="route-app",
        framework=Framework.SLIME,
        training_run_id="run-route-1",
        checkpoint_dir="/checkpoints/run-route-1",
    ).save()


def test_runs_route_returns_typed_joined_summaries(fake_volume, monkeypatch, tmp_path):
    _save_records()

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert len(response.json()) == 1
    summary = response.json()[0]
    assert summary["training_run_id"] == "run-route-1"
    assert summary["run_id"] == "run-route-1"
    assert summary["status"] == "running"
    assert summary["model"] == "Qwen/Qwen3-4B"
    assert summary["dataset"] == "openai/gsm8k"
    assert summary["recipe"] == "slime"
    assert summary["group_id"] == "route-group"
    assert summary["display_status"] == "completed"
    assert summary["display_stage"] == ""
    assert "list_fields" not in summary
    assert summary["has_train_result"] is True
    assert summary["train_result"]["checkpoint_dir"] == ("/checkpoints/run-route-1")


def test_get_run_route_returns_one_typed_summary(fake_volume, monkeypatch, tmp_path):
    _save_records()

    def fail_summary_scan(_store):
        raise AssertionError("single-run route must not scan summary stores")

    monkeypatch.setattr(metadata, "vol_get_summary_items_healed", fail_summary_scan)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/run-route-1")

    assert response.status_code == 200
    summary = response.json()
    assert summary["training_run_id"] == "run-route-1"
    assert summary["model"] == "Qwen/Qwen3-4B"
    assert summary["display_status"] == "completed"
    assert summary["train_result"]["checkpoint_dir"] == "/checkpoints/run-route-1"


def test_get_run_route_returns_404_for_unknown_run(fake_volume, monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Training run 'missing' not found"


def test_timings_route_returns_empty_for_unknown_run(
    fake_volume, monkeypatch, tmp_path
):
    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/missing/timings")

    assert response.status_code == 200
    assert response.json() == {}


def test_timings_route_derives_legacy_substeps(fake_volume, monkeypatch, tmp_path):
    run = TrainingRun(
        training_run_id="legacy-timing-run",
        modal_app_id="ap-legacy",
        framework=Framework.SLIME,
        config={},
        substep_times={
            "0": {
                "train": {"start": 100.0, "duration_s": 2.0},
            }
        },
    )
    run.save()

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/legacy-timing-run/timings")

    assert response.status_code == 200
    assert response.json()["metadata"]["legacy_derived"] is True


def test_rollout_route_preserves_raw_text_and_adds_cleaned_text(
    fake_volume, monkeypatch, tmp_path
):
    raw_prompt = "<|im_start|>user\nraw prompt<|im_end|>"
    raw_response = "<think>secret</think>raw answer"
    TrainingRolloutResult(
        training_run_id="run-route-1",
        rollout_id=3,
        samples=[
            {
                "score": 0.5,
                "prompt": raw_prompt,
                "response": raw_response,
                "parsed_response": {
                    "content": "raw answer",
                    "thinking": "secret",
                },
            }
        ],
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        display = client.get("/api/runs/run-route-1/rollouts/3")

    assert display.status_code == 200
    display_sample = display.json()["samples"][0]
    assert display_sample["prompt"] == "raw prompt"
    assert display_sample["response"] == "raw answer"
    assert display_sample["raw_prompt"] == raw_prompt
    assert display_sample["raw_response"] == raw_response
    assert display_sample["parsed_response"]["thinking"] == "secret"


def test_runs_route_keeps_runs_when_train_result_store_fails(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    original = metadata.vol_get_summary_items_healed

    def fail_results(store):
        if store is MetadataStore.TRAIN_RESULTS_SUMMARY:
            raise RuntimeError("result store unavailable")
        return original(store)

    monkeypatch.setattr(metadata, "vol_get_summary_items_healed", fail_results)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["has_train_result"] is False


def test_runs_route_filters_and_sorts_by_last_update(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    TrainingRun(
        training_run_id="run-route-2",
        framework=Framework.MILES,
        status="failed",
        config={
            "model": {"model_name": "other/model"},
            "dataset": {"hf_repo": "other/data"},
        },
        created_at=50,
        started_at=50,
        updated_at=2_000_000_000,
        metadata={"group_id": "other-group"},
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        filtered = client.get("/api/runs?display_status=failed&since=175&limit=1")
        all_runs = client.get("/api/runs")

    assert filtered.status_code == 200
    assert [run["run_id"] for run in filtered.json()] == ["run-route-2"]
    assert [run["run_id"] for run in all_runs.json()] == [
        "run-route-2",
        "run-route-1",
    ]


def test_apple_touch_icon_is_served_as_png(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/apple-touch-icon.png")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG")


def test_runs_route_isolates_invalid_run_records(fake_volume, monkeypatch, tmp_path):
    _save_records()
    runs = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert runs is not None
    invalid_run = {
        **runs[0],
        "training_run_id": "invalid-run",
        "step_times": {"1": {"phase": {"not": "an integer"}}},
    }
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_RUNS_SUMMARY, [invalid_run, runs[0]]
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["training_run_id"] for run in response.json()] == ["run-route-1"]


def test_run_log_stream_events_always_include_json_data(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-token-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-token-secret")

    batch = SimpleNamespace(
        entry_id="entry-1",
        task_id="task-1",
        items=[
            SimpleNamespace(data="first\n", timestamp=100.0),
            SimpleNamespace(data="second\n", timestamp=101.0),
        ],
        app_done=True,
    )

    class FakeAppGetLogs:
        def __init__(self):
            self.calls = 0

        def unary_stream(self, _request):
            self.calls += 1

            async def stream():
                if self.calls == 1:
                    raise RuntimeError("temporary failure")
                yield batch

            return stream()

    fake_rpc = FakeAppGetLogs()
    fake_modal_client = SimpleNamespace(
        stub=SimpleNamespace(AppGetLogs=fake_rpc),
    )

    async def from_credentials(*_args, **_kwargs):
        return fake_modal_client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("modal.client._Client.from_credentials", from_credentials)
    monkeypatch.setattr(_dashboard.asyncio, "sleep", no_sleep)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/run-route-1/logs/stream?max_lines_per_sec=1")

    assert response.status_code == 200
    blocks = [block for block in response.text.split("\n\n") if block]
    assert [block.splitlines()[0] for block in blocks] == [
        "event: reconnect",
        'data: {"task_id": "task-1", "line": "first\\n", "ts": 100.0}',
        "event: dropped",
        "event: done",
    ]
    reconnect_data = blocks[0].splitlines()[1]
    assert json.loads(reconnect_data.removeprefix("data: ")) == {
        "reason": "temporary failure"
    }
    for block in blocks:
        data_line = next(
            line for line in block.splitlines() if line.startswith("data: ")
        )
        assert isinstance(json.loads(data_line.removeprefix("data: ")), dict)


@pytest.mark.parametrize("bound", ["since", "until"])
def test_run_logs_rejects_invalid_time_bound(bound, fake_volume, monkeypatch, tmp_path):
    _save_records()

    with _client(monkeypatch, tmp_path) as client:
        response = client.get(
            "/api/runs/run-route-1/logs",
            params={bound: "yesterday-ish"},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith(
        f"{bound} must be epoch seconds, ISO 8601, or a relative time"
    )
