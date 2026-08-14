"""Ingestion tests over the committed demo fixture run dir."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from observatory.normalize import collect
from observatory.normalize.workspace import snapshot

FIXTURE = (Path(__file__).resolve().parent.parent / "fixtures" / "demo"
           / "ws_claude_dspy_20260717T090000")
RUN_DIR = FIXTURE / "workspace" / "agents" / "_runs" / "claude_dspy_20260717T090000"


class TestIngest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record, cls.ws = collect.build_record(FIXTURE)

    def test_dir_resolution_equivalence(self):
        for alt in (FIXTURE / "workspace", RUN_DIR):
            rec, _ = collect.build_record(alt, include_workspace=False)
            self.assertEqual(rec["meta"]["run_id"], "claude_dspy_20260717T090000")

    def test_events(self):
        events = self.record["events"]
        # 109 events from the borrowed PostTrainBench trajectory + 12 spliced
        # in by make_demo_run.py's _inject_learning_events (Task 7: 2 narration
        # events + 5 Bash tool_use/tool_result pairs demonstrating the
        # learning timeline) = 121.
        self.assertEqual(len(events), 121)
        for e in events:
            if e["type"] in ("assistant", "user"):
                self.assertTrue(e.get("blocks"), f"event {e['i']} has no blocks")

    def test_event_ts_from_sidecar(self):
        first_row = json.loads(
            (RUN_DIR / ".obs" / "line_ts.jsonl").read_text().splitlines()[0])
        events = self.record["events"]
        self.assertEqual(events[0]["ts"], first_row["ts"])
        self.assertTrue(all(e.get("ts") for e in events))

    def test_turn_numbers(self):
        turns = [e["turn"] for e in self.record["events"] if e["type"] == "assistant"]
        self.assertTrue(turns)
        self.assertTrue(all(t >= 1 for t in turns))
        self.assertEqual(turns, sorted(turns))

    def test_scores_result_entry(self):
        r = self.record["scores"]["results"][0]
        self.assertEqual(r["tag"], "demo_qa_v1")
        self.assertEqual(r["split"], "dev")
        self.assertEqual(r["budget"], 5)
        self.assertEqual(r["mean"], 0.57)  # from the file, never recomputed
        self.assertEqual(r["n_failed"], 1)
        failed_q = r["per_question"]["dspy_05f6a7b8c9da"]
        self.assertIsNone(failed_q["claim_score"])  # null stays null, never 0
        self.assertEqual(failed_q["tool_calls"], 0)
        self.assertAlmostEqual(r["tool_calls_avg"], 3.17, places=2)

    def test_index_row(self):
        row = self.record["index_row"]
        self.assertEqual(row["task"], "dspy")
        self.assertEqual(row["scaffold"], "claude")
        self.assertEqual(row["state"], "finished")
        self.assertIs(row["canonical"], False)
        self.assertEqual(row["integrity"], "OK")
        self.assertEqual(row["audit"], "CLEAN")
        self.assertEqual(row["best_dev_score"], 0.57)

    def test_workspace_snapshot(self):
        self.assertIsNotNone(self.ws)
        paths = {f["path"]: f for f in self.ws["files"]}
        self.assertFalse(any(p.startswith("agents/_runs") for p in paths))
        self.assertFalse(any(".obs" in Path(p).parts for p in paths))
        eval_py = paths.get("submission/eval.py")
        self.assertIsNotNone(eval_py)
        self.assertTrue(eval_py["inline"])
        self.assertIn("WEIGHTS", eval_py["content"])

    def test_workspace_snapshot_excludes_dotenv_secrets(self):
        secret = "ANTHROPIC_API_KEY=must-not-enter-workspace-json"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in (".env", ".env.local", ".env.production", ".envrc"):
                (root / name).write_text(secret)
            (root / ".env.example").write_text("ANTHROPIC_API_KEY=placeholder")
            (root / "visible.txt").write_text("safe")

            ws = snapshot(root)

        paths = {f["path"] for f in ws["files"]}
        self.assertNotIn(".env", paths)
        self.assertNotIn(".env.local", paths)
        self.assertNotIn(".env.production", paths)
        self.assertNotIn(".envrc", paths)
        self.assertIn(".env.example", paths)
        self.assertIn("visible.txt", paths)
        self.assertNotIn(secret, json.dumps(ws))

    def test_judgements_audit(self):
        self.assertEqual(self.record["judgements"]["audit"]["integrity"], "CLEAN")

    def test_summary(self):
        self.assertIsNotNone(self.record["summary"]["total_cost_usd"])

    def test_demo_fixture_carries_track(self):
        # observatory/fixtures/demo/.../run_meta.json (track "easy") sits at the
        # ws_root.parent level ($RUN_PARENT), same place the seeding routine writes it.
        self.assertEqual(self.record["meta"]["track"], "easy")
        self.assertEqual(self.record["index_row"]["track"], "easy")

    def test_demo_fixture_carries_learning_actions(self):
        # Guards fixture/generator drift (Task 7): make_demo_run.py splices a
        # datagen call, a bench.py train, a rubric_eval, and one invented
        # script (scripts/quick_dev_probe.py, absent from seed_manifest.txt,
        # present in the workspace snapshot) used twice into the demo trace.
        actions = self.record["learning"]
        self.assertEqual(
            [(a["kind"], a["tool"], a["provenance"]) for a in actions],
            [("data", "grounded_qa", "seed"),
             ("train", "sft", "seed"),
             ("tool", "scripts/quick_dev_probe.py", "invented"),
             ("eval", "rubric_eval", "seed"),
             ("tool", "scripts/quick_dev_probe.py", "invented")])
        self.assertEqual([a["nth_use"] for a in actions], [1, 1, 1, 1, 2])
        # event_i must index into this same record's events (jump-to-event).
        events = self.record["events"]
        for a in actions:
            self.assertEqual(events[a["event_i"]]["ts"], a["ts"])
            self.assertIn("Bash", str(events[a["event_i"]]["blocks"]))
        self.assertEqual(self.record["index_row"]["learning_counts"],
                         {"data": 1, "train": 1, "eval": 1, "evolve": 0,
                          "harness": 0, "infra": 0, "invented_tools": 1})


def _bare_run_dir(root: Path, run_id: str = "claude_dspy_20260101_000000") -> Path:
    """Minimal run dir collect.build_record can ingest: no trace, no scores, no
    solve_status.txt (-> state 'running') — just enough for resolve_dirs to work."""
    run_dir = root / "workspace" / "agents" / "_runs" / run_id
    run_dir.mkdir(parents=True)
    return run_dir


class TestRunMetaTrack(unittest.TestCase):
    """run_meta.json = {"track","scaffold","task","hours","prepared_at"}, written by
    the seeding routine (workspace_setup/prepare_workspace.sh) into $RUN_PARENT (== ws_root.parent). Old runs that predate
    tracks have no run_meta.json anywhere and must still ingest cleanly."""

    def test_track_from_run_meta_at_ws_root_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _bare_run_dir(root)
            (root / "run_meta.json").write_text(json.dumps({
                "track": "hard", "scaffold": "claude", "task": "dspy",
                "hours": "24", "prepared_at": "2026-01-01T00:00:00Z",
            }))
            rec, _ = collect.build_record(root, include_workspace=False)
            self.assertEqual(rec["meta"]["track"], "hard")
            self.assertEqual(rec["index_row"]["track"], "hard")

    def test_track_none_without_run_meta_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _bare_run_dir(root)
            rec, _ = collect.build_record(root, include_workspace=False)
            self.assertIsNone(rec["meta"]["track"])
            self.assertIsNone(rec["index_row"]["track"])

    def test_track_none_on_malformed_run_meta_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _bare_run_dir(root)
            (root / "run_meta.json").write_text("{not valid json")
            rec, _ = collect.build_record(root, include_workspace=False)
            self.assertIsNone(rec["meta"]["track"])
            self.assertIsNone(rec["index_row"]["track"])

    def test_track_none_on_non_string_track_value(self):
        # Valid JSON object, but "track" is present and not a string — the
        # isinstance guard in _load_track must yield None, never crash.
        for bad in (None, 123):
            with self.subTest(track=bad):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _bare_run_dir(root)
                    (root / "run_meta.json").write_text(json.dumps({
                        "track": bad, "scaffold": "claude", "task": "dspy",
                    }))
                    rec, _ = collect.build_record(root, include_workspace=False)
                    self.assertIsNone(rec["meta"]["track"])
                    self.assertIsNone(rec["index_row"]["track"])

    def test_run_dir_run_meta_takes_precedence_over_ws_root_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = _bare_run_dir(root)
            (root / "run_meta.json").write_text(json.dumps({"track": "medium"}))
            (run_dir / "run_meta.json").write_text(json.dumps({"track": "easy"}))
            rec, _ = collect.build_record(root, include_workspace=False)
            self.assertEqual(rec["meta"]["track"], "easy")
            self.assertEqual(rec["index_row"]["track"], "easy")


class TestUsageTotals(unittest.TestCase):
    """Header totals must aggregate across sessions and trace dialects:
    claude carries usage on per-session result events; codex carries it on
    per-turn token_count/turn.completed events with its own key names."""

    def test_codex_usage_summed_and_canonicalized(self):
        from observatory.normalize import codex
        lines = []
        for sess in ("t1", "t2"):
            lines += [
                json.dumps({"type": "thread.started", "thread_id": sess}),
                json.dumps({"type": "item.started", "item": {
                    "id": f"c_{sess}", "item_type": "command_execution",
                    "command": "ls"}}),
                json.dumps({"type": "item.completed", "item": {
                    "id": f"c_{sess}", "item_type": "command_execution",
                    "command": "ls", "aggregated_output": "ok", "exit_code": 0}}),
                json.dumps({"type": "item.completed", "item": {
                    "id": "m1", "item_type": "agent_message", "text": "hi"}}),
                json.dumps({"type": "turn.completed", "usage": {
                    "input_tokens": 100, "cached_input_tokens": 40,
                    "cache_write_input_tokens": 10, "output_tokens": 7,
                    "reasoning_output_tokens": 3}}),
            ]
        parsed = codex.parse_trace(lines)
        u = parsed["summary"]["usage_total"]
        self.assertEqual(u["input_tokens"], 200)
        self.assertEqual(u["output_tokens"], 14)
        self.assertEqual(u["cache_read_input_tokens"], 80)   # cached_input_tokens
        self.assertEqual(u["cache_creation_input_tokens"], 20)  # cache_write_
        self.assertNotIn("cached_input_tokens", u)
        self.assertEqual(parsed["summary"]["session_count"], 2)
        # no result events in codex traces -> falls back to the turn counter
        max_turn = max(e.get("turn") or 0 for e in parsed["events"])
        self.assertGreaterEqual(max_turn, 2)
        self.assertEqual(parsed["summary"]["num_turns"], max_turn)

    def test_claude_multi_session_results_summed(self):
        from observatory.normalize import claude_stream
        def session(sid, cost, turns):
            return [
                json.dumps({"type": "system", "subtype": "init", "session_id": sid,
                            "model": "m", "tools": []}),
                json.dumps({"type": "assistant", "session_id": sid, "message": {
                    "role": "assistant", "content": [{"type": "text", "text": "x"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5}}}),
                json.dumps({"type": "result", "subtype": "success", "session_id": sid,
                            "num_turns": turns, "duration_ms": 1000,
                            "total_cost_usd": cost,
                            "usage": {"input_tokens": 10, "output_tokens": 5,
                                      "cache_read_input_tokens": 2}}),
            ]
        parsed = claude_stream.parse_trace(session("a", 0.5, 3) + session("b", 0.25, 4))
        s = parsed["summary"]
        self.assertEqual(s["usage_total"]["input_tokens"], 20)
        self.assertEqual(s["usage_total"]["cache_read_input_tokens"], 4)
        self.assertEqual(s["num_turns"], 7)
        self.assertEqual(s["duration_ms"], 2000)
        self.assertEqual(s["total_cost_usd"], 0.75)


if __name__ == "__main__":
    unittest.main()
