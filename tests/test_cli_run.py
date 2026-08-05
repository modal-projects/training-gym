from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from modal_training_gym import cli as cli_module
from modal_training_gym.cli import run as run_module
from modal_training_gym.cli.errors import CLIError, ExitCode
from modal_training_gym.common.run import (
    TrainingRun,
    TrainingRunStatus,
    finalize_training_run,
)


class FakeDashboardClient:
    payload: object = []
    payloads: dict[str, object] = {}
    streams: dict[str, list[tuple[str, str]]] = {}
    not_found_paths: set[str] = set()
    requests: list[tuple[str, dict[str, object]]] = []
    posts: list[tuple[str, object]] = []
    post_payloads: dict[str, object] = {}
    timeouts: list[tuple[str, float | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_json(self, path, *, params=None, not_found_error=None, timeout=None):
        self.requests.append((path, params))
        self.timeouts.append((path, timeout))
        if path in self.not_found_paths and not_found_error is not None:
            raise not_found_error
        return self.payloads.get(path, self.payload)

    def post_json(self, path, *, json=None, not_found_error=None, timeout=None):
        self.posts.append((path, json))
        if path in self.not_found_paths and not_found_error is not None:
            raise not_found_error
        return self.post_payloads.get(
            path,
            {"action": "killed", "skip_reason": None, "status": "stopped"},
        )

    def iter_event_stream(self, path, *, params=None, not_found_error=None):
        self.requests.append((path, params))
        if path in self.not_found_paths and not_found_error is not None:
            raise not_found_error
        yield from self.streams.get(path, [])


@pytest.fixture(autouse=True)
def fake_dashboard(monkeypatch):
    FakeDashboardClient.payload = []
    FakeDashboardClient.payloads = {}
    FakeDashboardClient.streams = {}
    FakeDashboardClient.not_found_paths = set()
    FakeDashboardClient.requests = []
    FakeDashboardClient.posts = []
    FakeDashboardClient.post_payloads = {}
    FakeDashboardClient.timeouts = []
    monkeypatch.setattr(run_module, "DashboardClient", FakeDashboardClient)
    monkeypatch.setattr(run_module, "app_live_status", lambda _app_id: None)


def _summary(**overrides):
    value = {
        "training_run_id": "run-1",
        "run_id": "run-1",
        "status": "failed",
        "display_status": "failed",
        "display_stage": "Training",
        "framework_status": "training",
        "model": "org/model",
        "dataset": "org/data",
        "recipe": "slime",
        "group_id": "nightly",
        "created_at": 100,
        "updated_at": 200,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("1,4,9", {1, 4, 9}),
        ("4-10", {4, 5, 6, 7, 8, 9}),
        ("4-10:2", {4, 6, 8}),
        ("1,4-10:2,9", {1, 4, 6, 8, 9}),
        (" 1, 4-10:2 ", {1, 4, 6, 8}),
    ],
)
def test_parse_steps(value, expected):
    assert run_module._parse_steps(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1,",
        "-1",
        "4-4",
        "10-4",
        "4-10:0",
        "4-10:-1",
        "not-a-step",
    ],
)
def test_parse_steps_rejects_invalid_values(value):
    with pytest.raises(click.BadParameter):
        run_module._parse_steps(value)


def test_run_list_help_derives_filter_flags_from_schema():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "list", "--help"])

    assert result.exit_code == 0
    for flag in (
        "--status",
        "--model",
        "--dataset",
        "--recipe",
        "--group",
    ):
        assert flag in result.stdout
    assert "--since" in result.stdout
    assert "--limit" in result.stdout
    assert "-j, --json" in result.stdout
    assert "--group GROUP" in result.stdout


def test_run_list_cli_field_names_are_unique():
    fields = run_module.run_list_field_metadata()
    output_names = [run_module.CLI_FIELD_NAMES.get(name, name) for name in fields]
    option_names = [
        run_module.CLI_FIELD_NAMES.get(name, name).replace("_", "-")
        for name, metadata in fields.items()
        if metadata.get("filterable")
    ]

    assert len(output_names) == len(set(output_names))
    assert len(option_names) == len(set(option_names))


def test_run_get_help_documents_flags_and_examples():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "get", "--help"])

    assert result.exit_code == 0
    assert "Show status and top-level metadata for a single run." in result.stdout
    assert "RUN_ID" in result.stdout
    assert "--verbose" in result.stdout
    assert "-j, --json" in result.stdout
    assert "Examples:" in result.stdout


def test_run_get_prints_top_level_status():
    FakeDashboardClient.payload = _summary(
        display_status="pending",
        display_stage="Generating rollouts",
        framework_progress={
            "current": 3,
            "total": 10,
            "unit": "step",
        },
        latest_rollout={
            "rollout_id": 2,
            "mean": 0.625,
            "total": 16,
            "created_at": 190,
        },
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "get", "run-1"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [("/api/runs/run-1", None)]
    assert "PENDING" in result.stdout
    assert "Generating rollouts" in result.stdout
    assert "3 / 10 step" in result.stdout
    assert "0.625" in result.stdout
    assert "Field" not in result.stdout
    assert "Value" not in result.stdout


def test_run_get_verbose_json_includes_reward_history_and_rollouts():
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(
            display_status="pending",
            framework_progress={"current": 4, "total": 10, "unit": "step"},
            latest_rollout={
                "rollout_id": 2,
                "mean": 0.75,
                "total": 8,
                "created_at": 200,
            },
        ),
        "/api/runs/run-1/rollouts": [
            {
                "training_run_id": "run-1",
                "rollout_id": 1,
                "created_at": 150,
                "total": 8,
                "mean": 0.5,
                "rollout_time": 12.5,
            },
            {
                "training_run_id": "run-1",
                "rollout_id": 2,
                "created_at": 200,
                "total": 8,
                "mean": 0.75,
            },
        ],
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "get", "run-1", "--verbose", "-j"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [
        ("/api/runs/run-1", None),
        ("/api/runs/run-1/rollouts", None),
    ]
    payload = json.loads(result.stdout)
    assert payload["current_step"] == 4
    assert payload["total_steps"] == 10
    assert payload["current_reward"] == 0.75
    assert payload["reward_over_time"] == [
        {
            "rollout_id": 1,
            "reward": 0.5,
            "created_at": "1970-01-01T00:02:30Z",
        },
        {
            "rollout_id": 2,
            "reward": 0.75,
            "created_at": "1970-01-01T00:03:20Z",
        },
    ]
    assert (
        payload["rollouts"] == FakeDashboardClient.payloads["/api/runs/run-1/rollouts"]
    )


def test_run_get_missing_run_returns_not_found_without_fetching_rollouts():
    FakeDashboardClient.not_found_paths = {"/api/runs/missing"}

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "get", "missing", "--verbose"],
    )

    assert result.exit_code == 3
    assert "Training run 'missing' was not found." in result.stderr
    assert "training-gym run list" in result.stderr
    assert FakeDashboardClient.requests == [("/api/runs/missing", None)]


def test_run_params_prints_raw_recipe_as_json():
    params = {
        "gpu_type": "H200",
        "num_rollout": 4,
        "sequence_parallel": False,
        "eval_prompt_data": ["eval", "/data/eval.jsonl"],
        "sglang_config": {"attention_backend": "flashinfer"},
        "load": "",
        "critic_num_nodes": None,
    }
    FakeDashboardClient.payload = _summary(
        framework="slime",
        config={"recipe": params},
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "params", "run-1", "-j"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [("/api/runs/run-1", None)]
    assert json.loads(result.stdout) == params


def test_run_params_supports_non_slime_frameworks():
    params = {"gpu_type": "H100", "num_nodes": 2}
    FakeDashboardClient.payload = _summary(
        framework="miles",
        config={"recipe": params},
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "params", "run-1", "-j"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == params
    assert FakeDashboardClient.requests == [("/api/runs/run-1", None)]


def test_run_logs_help_documents_modes_and_filters():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "logs", "--help"])

    assert result.exit_code == 0
    assert "Show logs for a training run." in result.stdout
    for flag in ("--follow", "--since", "--until", "--tail", "--search", "--json"):
        assert flag in result.stdout
    assert "RUN_ID" in result.stdout
    assert "training-gym run logs brave-falcon-3fa8 --follow" in result.stdout


def test_run_logs_fetches_recent_filtered_entries():
    FakeDashboardClient.payload = {
        "logs": [
            {"task_id": "task-1", "line": "first\n", "fd": 1, "ts": 100.0},
            {"task_id": "task-1", "line": "second\n", "fd": 1, "ts": 101.0},
        ],
        "has_more": False,
        "next_until": None,
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "logs",
            "run-1",
            "--since",
            "30m",
            "--until",
            "2026-07-28T12:00:00Z",
            "--tail",
            "25",
            "--search",
            "worker",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "first\nsecond\n"
    assert FakeDashboardClient.requests == [
        (
            "/api/runs/run-1/logs",
            {
                "since": "30m",
                "until": "2026-07-28T12:00:00Z",
                "max_lines": 25,
                "search": "worker",
            },
        )
    ]


def test_run_logs_json_preserves_entries_and_pagination():
    logs = [{"task_id": "task-1", "line": "hello", "fd": 1, "ts": 100.0}]
    FakeDashboardClient.payload = {
        "logs": logs,
        "has_more": True,
        "next_until": 99.5,
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "-j"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "logs": logs,
        "has_more": True,
        "next_until": 99.5,
    }
    assert "older logs are available with --until 99.5" in result.stderr
    assert FakeDashboardClient.requests[0][1]["max_lines"] == 100


def test_run_logs_warns_when_human_output_is_truncated():
    FakeDashboardClient.payload = {
        "logs": [{"task_id": "task-1", "line": "hello"}],
        "has_more": True,
        "next_until": None,
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1"],
    )

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert "showing the newest 1 logs; older logs are available" in result.stderr


def test_run_logs_follow_streams_lines_until_done():
    FakeDashboardClient.streams["/api/runs/run-1/logs/stream"] = [
        ("message", '{"task_id":"task-1","line":"hello\\n","ts":100}'),
        ("dropped", '{"dropped":2}'),
        ("reconnect", '{"reason":"temporary error"}'),
        ("message", '{"task_id":"task-1","line":"world\\n","ts":101}'),
        ("done", "{}"),
    ]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow", "--search", "worker"],
    )

    assert result.exit_code == 0
    assert result.stdout == "hello\nworld\n"
    assert "[dropped 2 log lines]" in result.stderr
    assert "[reconnecting log stream]" in result.stderr
    assert FakeDashboardClient.requests == [
        ("/api/runs/run-1/logs/stream", {"search": "worker"})
    ]


def test_run_logs_follow_fails_when_stream_ends_without_done():
    FakeDashboardClient.streams["/api/runs/run-1/logs/stream"] = [
        ("message", '{"task_id":"task-1","line":"hello\\n","ts":100}'),
    ]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow"],
    )

    assert result.exit_code == 5
    assert result.stdout == "hello\n"
    assert "Dashboard log stream ended unexpectedly." in result.stderr
    assert "Re-run the command to reconnect." in result.stderr


def test_run_logs_follow_json_uses_newline_delimited_events():
    FakeDashboardClient.streams["/api/runs/run-1/logs/stream"] = [
        ("message", '{"task_id":"task-1","line":"first","ts":100}'),
        ("message", '{"task_id":"task-1","line":"second","ts":101}'),
        ("message", '{"task_id":"task-1","line":"third","ts":102}'),
        ("done", "{}"),
    ]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow", "-j"],
    )

    assert result.exit_code == 0
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"task_id": "task-1", "line": "first", "ts": 100},
        {"task_id": "task-1", "line": "second", "ts": 101},
        {"task_id": "task-1", "line": "third", "ts": 102},
    ]


def test_run_logs_rejects_historical_bounds_with_follow():
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "logs", "run-1", "--follow", "--since", "30m"],
    )

    assert result.exit_code == 2
    assert "apply only when fetching logs without --follow" in result.stderr
    assert FakeDashboardClient.requests == []


def test_run_trace_help_documents_flags_and_examples():
    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "trace", "--help"])

    assert result.exit_code == 0
    assert "Download agent traces for a run" in result.stdout
    assert "RUN_ID" in result.stdout
    for flag in ("--out", "--step", "--dry-run", "--yes", "--force", "--json"):
        assert flag in result.stdout
    assert "training-gym run trace brave-falcon-3fa8" in result.stdout


def test_mark_run_terminal_updates_running_record(monkeypatch):
    run = TrainingRun(
        training_run_id="run-1",
        framework="slime",
        config={},
        started_at=100,
        metadata={"existing": "value"},
    )
    saved = []
    monkeypatch.setattr(
        TrainingRun,
        "from_id",
        classmethod(lambda _cls, _run_id: run),
    )
    monkeypatch.setattr(
        TrainingRun,
        "save",
        lambda self: saved.append(self),
    )

    status = finalize_training_run(
        "run-1",
        status=TrainingRunStatus.STOPPED,
        reason="killed_by_cli",
        ended_at=225,
    )

    assert status == TrainingRunStatus.STOPPED
    assert saved == [run]
    assert run.status == TrainingRunStatus.STOPPED
    assert run.ended_at == 225
    assert run.completed_at == 225
    assert run.duration_seconds == 125
    assert run.metadata == {
        "existing": "value",
        "terminal_reason": "killed_by_cli",
    }


def test_mark_run_terminal_preserves_terminal_record(monkeypatch):
    run = TrainingRun(
        training_run_id="run-1",
        framework="slime",
        config={},
        status=TrainingRunStatus.COMPLETED,
    )
    monkeypatch.setattr(
        TrainingRun,
        "from_id",
        classmethod(lambda _cls, _run_id: run),
    )
    monkeypatch.setattr(
        TrainingRun,
        "save",
        lambda _self: pytest.fail("terminal record should not be saved"),
    )

    status = finalize_training_run(
        "run-1",
        status=TrainingRunStatus.STOPPED,
        reason="killed_by_cli",
        ended_at=225,
    )

    assert status == TrainingRunStatus.COMPLETED
    assert run.status == TrainingRunStatus.COMPLETED


def test_run_kill_dry_run_reports_jobs_without_stopping(monkeypatch):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(
            status="running",
            display_status="pending",
            modal_app_id="ap-1",
            started_at=100,
            duration_seconds=125,
            framework_progress={"current": 3, "total": 10, "unit": "step"},
        ),
        "/api/runs/run-2": _summary(
            training_run_id="run-2",
            run_id="run-2",
            status="failed",
            display_status="failed",
            modal_app_id="ap-2",
            duration_seconds=60,
        ),
    }
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "run-2", "--dry-run", "--json"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.posts == []
    assert FakeDashboardClient.requests == [
        ("/api/runs/run-1", None),
        ("/api/runs/run-2", None),
    ]
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["kill_count"] == 1
    assert payload["skipped_count"] == 1
    assert payload["error_count"] == 0
    assert payload["runs"] == [
        {
            "run_id": "run-1",
            "modal_app_id": "ap-1",
            "modal_app_live": None,
            "status": "running",
            "current_step": 3,
            "total_steps": 10,
            "step_unit": "step",
            "duration_seconds": 125,
            "action": "would_kill",
            "skip_reason": None,
        },
        {
            "run_id": "run-2",
            "modal_app_id": "ap-2",
            "modal_app_live": None,
            "status": "failed",
            "current_step": None,
            "total_steps": None,
            "step_unit": "step",
            "duration_seconds": 60,
            "action": "skipped",
            "skip_reason": "already_terminal",
        },
    ]


def test_run_kill_treats_finished_non_live_run_as_successful_noop(monkeypatch):
    FakeDashboardClient.payload = _summary(
        status="completed",
        display_status="completed",
        modal_app_id="ap-1",
    )
    monkeypatch.setattr(run_module, "app_live_status", lambda _app_id: False)

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "--dry-run", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kill_count"] == 0
    assert payload["error_count"] == 0
    assert payload["runs"][0]["action"] == "skipped"
    assert payload["runs"][0]["skip_reason"] == "already_terminal"


def test_run_kill_stops_each_active_modal_app_and_skips_terminal_runs(monkeypatch):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(
            status="running",
            display_status="pending",
            modal_app_id="ap-1",
        ),
        "/api/runs/run-2": _summary(
            training_run_id="run-2",
            run_id="run-2",
            status="running",
            display_status="pending",
            modal_app_id="ap-2",
        ),
        "/api/runs/run-3": _summary(
            training_run_id="run-3",
            run_id="run-3",
            status="stopped",
            display_status="stopped",
            modal_app_id="ap-3",
        ),
    }
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "run-2", "run-3", "--yes", "--json"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.posts == [
        ("/api/runs/run-1/kill", None),
        ("/api/runs/run-2/kill", None),
    ]
    payload = json.loads(result.stdout)
    assert [run["action"] for run in payload["runs"]] == [
        "killed",
        "killed",
        "skipped",
    ]
    assert [run["status"] for run in payload["runs"]] == [
        "stopped",
        "stopped",
        "stopped",
    ]


def test_run_kill_dry_run_reports_active_run_without_modal_app():
    FakeDashboardClient.payload = _summary(
        status="running",
        display_status="pending",
        modal_app_id="",
    )

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "--dry-run", "--json"],
    )

    assert result.exit_code == ExitCode.BACKEND
    payload = json.loads(result.stdout)
    assert payload["kill_count"] == 0
    assert payload["skipped_count"] == 1
    assert payload["error_count"] == 1
    assert payload["runs"][0]["action"] == "skipped"
    assert payload["runs"][0]["skip_reason"] == "missing_modal_app_id"


def test_run_kill_continues_after_run_without_modal_app(monkeypatch):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(
            status="running",
            display_status="pending",
            modal_app_id="",
        ),
        "/api/runs/run-2": _summary(
            training_run_id="run-2",
            run_id="run-2",
            status="running",
            display_status="pending",
            modal_app_id="ap-2",
        ),
    }
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "run-2", "--yes", "--json"],
    )

    assert result.exit_code == ExitCode.BACKEND
    assert FakeDashboardClient.posts == [("/api/runs/run-2/kill", None)]
    payload = json.loads(result.stdout)
    assert payload["error_count"] == 1
    assert [(run["run_id"], run["action"]) for run in payload["runs"]] == [
        ("run-1", "skipped"),
        ("run-2", "killed"),
    ]


def test_run_kill_reports_canonical_terminal_status_after_stopping(monkeypatch):
    FakeDashboardClient.payload = _summary(
        status="running",
        display_status="pending",
        modal_app_id="ap-1",
    )
    monkeypatch.setattr(run_module, "app_live_status", lambda _app_id: True)
    FakeDashboardClient.post_payloads["/api/runs/run-1/kill"] = {
        "action": "killed",
        "skip_reason": None,
        "status": "completed",
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "--yes", "--json"],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.posts == [("/api/runs/run-1/kill", None)]
    payload = json.loads(result.stdout)
    assert payload["runs"][0]["action"] == "killed"
    assert payload["runs"][0]["status"] == "completed"


def test_run_kill_reports_dashboard_kill_failure(monkeypatch):
    FakeDashboardClient.payload = _summary(
        status="running",
        display_status="pending",
        modal_app_id="ap-1",
    )

    def fail_post(_self, _path, **_kwargs):
        raise CLIError(
            "Could not stop the Modal app",
            error="dashboard_server_error",
            exit_code=ExitCode.BACKEND,
        )

    monkeypatch.setattr(FakeDashboardClient, "post_json", fail_post)

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "--yes"],
    )

    assert result.exit_code == ExitCode.BACKEND
    assert "Could not stop the Modal app" in result.stderr


def test_run_kill_rejects_invalid_dashboard_response():
    FakeDashboardClient.payload = _summary(
        status="running",
        display_status="pending",
        modal_app_id="ap-1",
    )
    FakeDashboardClient.post_payloads["/api/runs/run-1/kill"] = {}

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "kill", "run-1", "--yes"],
    )

    assert result.exit_code == ExitCode.BACKEND
    assert "Dashboard returned invalid kill data" in result.stderr


def test_run_trace_dry_run_filters_steps_without_downloading(tmp_path):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(),
        "/api/runs/run-1/rollouts": [
            {
                "training_run_id": "run-1",
                "rollout_id": step,
                "created_at": 100 + step,
                "total": 8,
                "mean": step / 10,
                "export_size_bytes": 1000 + step,
            }
            for step in range(5)
        ],
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--step",
            "0-4:2",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["step_count"] == 2
    assert payload["sample_count"] == 16
    assert payload["export_size_bytes"] == 2002
    assert [step["step"] for step in payload["steps"]] == [0, 2]
    assert set(payload["steps"][0]) == {
        "step",
        "file_name",
        "samples",
        "mean_reward",
        "export_size_bytes",
    }
    assert payload["output_path"] == str(tmp_path / "run-1")
    assert FakeDashboardClient.requests == [
        ("/api/runs/run-1", None),
        ("/api/runs/run-1/rollouts", None),
    ]
    assert not (tmp_path / "run-1").exists()


def test_run_trace_file_names_use_highest_step_width(tmp_path):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(),
        "/api/runs/run-1/rollouts": [
            {
                "training_run_id": "run-1",
                "rollout_id": step,
                "created_at": 100,
                "total": 8,
                "mean": 0.5,
                "export_size_bytes": 1000,
            }
            for step in (2, 10_000)
        ],
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [step["file_name"] for step in payload["steps"]] == [
        "step_00002.json",
        "step_10000.json",
    ]


def test_run_trace_dry_run_reports_unknown_size_for_legacy_rollouts(tmp_path):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(),
        "/api/runs/run-1/rollouts": [
            {
                "training_run_id": "run-1",
                "rollout_id": 0,
                "created_at": 100,
                "total": 8,
                "mean": 0.5,
            }
        ],
    }

    human = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
    )
    json_result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
    )

    assert human.exit_code == 0
    assert "approximately unknown size" in human.stdout
    payload = json.loads(json_result.stdout)
    assert payload["export_size_bytes"] is None
    assert payload["steps"][0]["export_size_bytes"] is None


def test_run_trace_downloads_steps_and_writes_manifest(tmp_path):
    summaries = [
        {
            "training_run_id": "run-1",
            "rollout_id": step,
            "created_at": 100 + step,
            "total": 1,
            "mean": reward,
            "export_size_bytes": 500,
        }
        for step, reward in ((0, 0.25), (2, 0.75))
    ]
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(),
        "/api/runs/run-1/rollouts": summaries,
        "/api/runs/run-1/rollouts/0": {
            "training_run_id": "run-1",
            "rollout_id": 0,
            "samples": [
                {
                    "score": 0.25,
                    "prompt": "original question",
                    "response": "original answer",
                    "raw_prompt": "<|im_start|>user\noriginal question<|im_end|>",
                    "raw_response": "<think>hidden</think>original answer",
                    "parsed_response": {
                        "content": "original answer",
                        "thinking": "hidden",
                    },
                    "trace": [{"name": "generate", "start": 1.0, "end": 2.0}],
                }
            ],
        },
        "/api/runs/run-1/rollouts/2": {
            "training_run_id": "run-1",
            "rollout_id": 2,
            "samples": [
                {
                    "score": 0.75,
                    "prompt": "another",
                    "response": "response",
                    "trace": [{"name": "reward", "start": 3.0, "end": 4.0}],
                }
            ],
        },
    }

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0
    output_path = tmp_path / "run-1"
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest == {
        "run_id": "run-1",
        "steps": [
            {
                "step": 0,
                "file_name": "step_0000.json",
                "samples": 1,
                "mean_reward": 0.25,
                "export_size_bytes": (output_path / "step_0000.json").stat().st_size,
            },
            {
                "step": 2,
                "file_name": "step_0002.json",
                "samples": 1,
                "mean_reward": 0.75,
                "export_size_bytes": (output_path / "step_0002.json").stat().st_size,
            },
        ],
    }
    step_zero = json.loads((output_path / "step_0000.json").read_text())
    assert step_zero["samples"][0]["trace"][0]["name"] == "generate"
    assert step_zero["samples"][0]["prompt"] == "original question"
    assert step_zero["samples"][0]["response"] == "original answer"
    assert step_zero["samples"][0]["raw_prompt"] == (
        "<|im_start|>user\noriginal question<|im_end|>"
    )
    assert step_zero["samples"][0]["raw_response"] == (
        "<think>hidden</think>original answer"
    )
    assert step_zero["samples"][0]["parsed_response"] == {
        "content": "original answer",
        "thinking": "hidden",
    }
    assert json.loads((output_path / "step_0002.json").read_text())["rollout_id"] == 2
    payload = json.loads(result.stdout)
    assert payload["output_path"] == str(output_path)
    assert payload["dry_run"] is False
    assert payload["export_size_bytes"] > 0
    assert set(payload["steps"][0]) == {
        "step",
        "file_name",
        "samples",
        "mean_reward",
        "export_size_bytes",
    }
    assert [
        timeout
        for path, timeout in FakeDashboardClient.timeouts
        if timeout == run_module.TRACE_DOWNLOAD_TIMEOUT_SECONDS
    ] == [
        run_module.TRACE_DOWNLOAD_TIMEOUT_SECONDS,
        run_module.TRACE_DOWNLOAD_TIMEOUT_SECONDS,
    ]


def test_run_trace_rejects_missing_and_invalid_steps(tmp_path):
    FakeDashboardClient.payloads = {
        "/api/runs/run-1": _summary(),
        "/api/runs/run-1/rollouts": [],
    }

    missing = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--step",
            "4",
            "--dry-run",
        ],
    )
    invalid = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "trace",
            "run-1",
            "--out",
            str(tmp_path),
            "--step",
            "4-2",
            "--dry-run",
        ],
    )

    assert missing.exit_code == 2
    assert "Step(s) not found" in missing.stderr
    assert invalid.exit_code == 2
    assert "start-inclusive, end-exclusive ranges such as 4-100:2" in invalid.stderr


@pytest.mark.parametrize(
    ("flag", "value", "backend_field"),
    [
        ("--status", "failed", "display_status"),
        ("--model", "org/model", "model"),
        ("--dataset", "org/data", "dataset"),
        ("--recipe", "slime", "recipe"),
        ("--group", "nightly", "group_id"),
    ],
)
def test_run_list_forwards_each_filter_flag(flag, value, backend_field):
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "list", flag, value, "-j"],
    )

    assert result.exit_code == 0
    path, params = FakeDashboardClient.requests[0]
    assert path == "/api/runs"
    assert params[backend_field] == value


def test_run_list_forwards_filters_and_prints_configured_fields_as_json():
    FakeDashboardClient.payload = [_summary()]

    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        [
            "run",
            "list",
            "--status",
            "failed",
            "--group",
            "nightly",
            "--since",
            "2026-07-23T12:00:00Z",
            "--limit",
            "3",
            "-j",
        ],
    )

    assert result.exit_code == 0
    assert FakeDashboardClient.requests == [
        (
            "/api/runs",
            {
                "display_status": "failed",
                "model": None,
                "dataset": None,
                "recipe": None,
                "group_id": "nightly",
                "since": 1784808000,
                "limit": 3,
            },
        )
    ]
    assert json.loads(result.stdout) == [
        {
            "run_id": "run-1",
            "status": "failed",
            "stage": "Training",
            "model": "org/model",
            "dataset": "org/data",
            "recipe": "slime",
            "group": "nightly",
            "created_at": "1970-01-01T00:01:40Z",
            "last_updated_at": "1970-01-01T00:03:20Z",
        }
    ]


def test_run_list_rejects_invalid_since_before_request():
    result = CliRunner().invoke(
        cli_module.entrypoint_cli,
        ["run", "list", "--since", "yesterday-ish"],
    )

    assert result.exit_code == 2
    assert "epoch seconds, ISO 8601, or a relative time" in result.stderr
    assert FakeDashboardClient.requests == []


def test_run_list_renders_schema_columns():
    FakeDashboardClient.payload = [_summary()]

    result = CliRunner().invoke(cli_module.entrypoint_cli, ["run", "list"])

    assert result.exit_code == 0
    for heading in (
        "Run",
        "Status",
        "Stage",
        "Model",
        "Dataset",
        "Recipe",
        "Group",
        "Created",
        "Last updated",
    ):
        assert heading in result.stdout
