"""trajectory.json (Harbor ATIF-v1.7) is written at ingest and obeys the
ATIF validator rules: sequential step ids from 1, agent-only fields on agent
steps only, every observation referencing an issued tool_call_id.

Runs offline against the demo fixture (--no-upload), so no modal dep.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from observatory import cli, schema
from observatory.normalize import atif

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "demo" / \
    "ws_claude_dspy_20260717T090000"

AGENT_ONLY_FIELDS = ("reasoning_content", "tool_calls", "model_name")


def _validate(trajectory: dict) -> None:
    steps = trajectory["steps"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    call_ids = set()
    for step in steps:
        assert step["source"] in ("user", "agent", "system")
        assert isinstance(step["timestamp"], str) and "T" in step["timestamp"]
        if step["source"] != "agent":
            for field in AGENT_ONLY_FIELDS:
                assert field not in step, f"{field} on {step['source']} step"
        for call in step.get("tool_calls", []):
            assert call["tool_call_id"] and call["function_name"]
            call_ids.add(call["tool_call_id"])
    for step in steps:
        for result in step.get("observation", {}).get("results", []):
            assert result["source_call_id"] in call_ids
    assert trajectory["final_metrics"]["total_steps"] == len(steps)


class ConverterRules(unittest.TestCase):
    def test_fixture_record_converts_to_valid_atif(self):
        from observatory.normalize import collect
        record, _ = collect.build_record(FIXTURE)
        trajectory = atif.events_to_atif(record, record["events"])
        self.assertEqual(trajectory["schema_version"], "ATIF-v1.7")
        self.assertTrue(trajectory["steps"])
        _validate(trajectory)

    def test_tool_results_fold_onto_calling_step(self):
        record = {"meta": {"run_id": "r1", "launched_at": "2026-01-01T00:00:00Z"},
                  "index_row": {"run_id": "r1"}, "summary": {}}
        events = [
            {"type": "assistant", "session_idx": 0, "ts": "2026-01-01T00:00:01Z",
             "blocks": [{"type": "tool_use", "id": "item_1", "name": "command",
                         "input": {"command": "ls"}}]},
            {"type": "user", "session_idx": 0, "ts": "2026-01-01T00:00:02Z",
             "blocks": [{"type": "tool_result", "tool_use_id": "item_1",
                         "content": "a.txt"}]},
            # a resume reuses the same item id in a new session
            {"type": "assistant", "session_idx": 1, "ts": "2026-01-01T00:01:00Z",
             "blocks": [{"type": "tool_use", "id": "item_1", "name": "command",
                         "input": {"command": "pwd"}}]},
            {"type": "user", "session_idx": 1, "ts": "2026-01-01T00:01:01Z",
             "blocks": [{"type": "tool_result", "tool_use_id": "item_1",
                         "content": "/root"}]},
        ]
        trajectory = atif.events_to_atif(record, events)
        _validate(trajectory)
        agent_steps = [s for s in trajectory["steps"] if s["source"] == "agent"]
        self.assertEqual(len(agent_steps), 2)
        self.assertEqual(
            agent_steps[0]["observation"]["results"][0]["content"], "a.txt")
        self.assertEqual(
            agent_steps[1]["observation"]["results"][0]["content"], "/root")


class StagedArtifact(unittest.TestCase):
    def test_ingest_writes_trajectory_json(self):
        with tempfile.TemporaryDirectory() as td:
            rc = cli.main(["ingest", str(FIXTURE), "--no-upload",
                           "--data-dir", td])
            self.assertEqual(rc, 0)
            run_dir = next((Path(td) / schema.RUNS_PREFIX).iterdir())
            path = run_dir / schema.TRAJECTORY_FILE
            self.assertTrue(path.is_file())
            trajectory = json.loads(path.read_text())
            self.assertEqual(trajectory["schema_version"], "ATIF-v1.7")
            _validate(trajectory)


if __name__ == "__main__":
    unittest.main()
