"""Tests for observatory/normalize/learning.py — the seed-tool registry and
pure command classification behind the observatory's "learning timeline".

`learning.py` itself may not touch the filesystem or import from toolbox/ (it
must ingest runs on machines without the repo's training/eval deps). The
anti-drift guarantee — that the registry never silently misses a new
toolbox/data_toolbox/gen/*.py generator — lives here instead, as a test that
walks the real gen/ directory in this repo checkout.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from observatory.normalize import collect, learning

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "toolbox" / "data_toolbox" / "gen"


def _is_generator_script(path: Path) -> bool:
    """A gen/*.py file counts as a generator CLI — and must classify as
    (kind="data", tool=<stem>) — iff it is a standalone, argparse-driven
    script: it imports argparse AND defines an `if __name__ == "__main__":`
    entry point. This is the documented skip rule: it excludes __init__.py
    (a package docstring only, no CLI) and would exclude any future
    shared-helper module under gen/ that isn't meant to be invoked directly,
    without excluding any real generator (every gen/*.py script in this repo
    as of writing — annotation, grounded_qa, hygiene, implications, mix,
    paraphrase, react_trace_gen, reasoning, self_distill — satisfies both
    conditions)."""
    text = path.read_text()
    return "argparse" in text and "__main__" in text


class TestRegistryDrift(unittest.TestCase):
    """The registry can't silently miss a new generator: every gen/*.py CLI
    script found on disk must classify via classify_command as (data, stem)."""

    def test_every_generator_stem_classifies_as_data(self):
        if not GEN_DIR.is_dir():
            self.skipTest("toolbox/data_toolbox/gen not present in this checkout")
        stems = [p.stem for p in sorted(GEN_DIR.glob("*.py"))
                 if p.name != "__init__.py" and _is_generator_script(p)]
        self.assertTrue(stems, "expected at least one generator script under gen/")
        for stem in stems:
            with self.subTest(stem=stem):
                cmd = f"python3 toolbox/data_toolbox/gen/{stem}.py --corpus c --out o"
                actions = learning.classify_command(cmd)
                self.assertEqual(len(actions), 1)
                self.assertEqual(actions[0]["kind"], "data")
                self.assertEqual(actions[0]["tool"], stem)
                self.assertEqual(actions[0]["provenance"], "seed")

    def test_every_folder_form_tool_classifies_with_category_kind(self):
        """Folder-form tools (TOOL_SPEC.md): every <category>/…/<tool>/run.py on
        disk classifies as (kind-of-category, folder name) — seed and future
        invented tools alike, no per-tool registry entries to maintain."""
        toolbox = GEN_DIR.parents[1]
        kind_of = {"data_tool": "data", "training_tool": "train", "eval_tool": "eval",
                   "harness_tool": "harness", "self_evolve_tool": "evolve",
                   "inference_tool": "infra", "agentic_toolbox": "harness",
                   # legacy dirs — keep classifying forever
                   "data_toolbox": "data", "eval_toolbox": "eval",
                   "harness_toolbox": "eval", "training_toolbox": "train",
                   "evolve": "evolve"}
        skip = {"axolotl", "slime", "sdft", "__pycache__", "recipes", "runs",
                "tests", "papers", "prompts"}
        found = []
        for cat, kind in kind_of.items():
            root = toolbox / cat
            if not root.is_dir():
                continue
            for run in root.rglob("run.py"):
                rel = run.relative_to(toolbox)
                if any(part in skip for part in rel.parts):
                    continue
                found.append((f"toolbox/{rel.as_posix()}", kind, run.parent.name))
        if not found:
            self.skipTest("no folder-form tools in this checkout")
        for path, kind, name in found:
            with self.subTest(tool=path):
                actions = learning.classify_command(f"python3 {path} --x y")
                self.assertEqual(len(actions), 1)
                self.assertEqual(actions[0]["kind"], kind)
                self.assertEqual(actions[0]["tool"], name)
                self.assertEqual(actions[0]["provenance"], "seed")

    def test_invented_folder_form_tool_classifies_without_registry_change(self):
        actions = learning.classify_command(
            "python3 toolbox/data_toolbox/gen/statement_qa/run.py --corpus c --out o")
        self.assertEqual([(a["kind"], a["tool"]) for a in actions],
                         [("data", "statement_qa")])

    def test_skip_rule_excludes_init_and_non_cli_files(self):
        # __init__.py (package docstring, no argparse/no __main__) must never
        # be asserted as a generator by the drift test above.
        init_py = GEN_DIR / "__init__.py"
        if not init_py.exists():
            self.skipTest("toolbox/data_toolbox/gen/__init__.py not present")
        self.assertFalse(_is_generator_script(init_py))


class TestClassifyCommand(unittest.TestCase):
    def test_plain_datagen_call(self):
        cmd = ("python3 toolbox/data_toolbox/gen/paraphrase.py "
               "--corpus tasks/fav2/corpus --out data/fav2_paraphrase.rows.jsonl")
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "data")
        self.assertEqual(actions[0]["tool"], "paraphrase")
        self.assertEqual(actions[0]["provenance"], "seed")
        self.assertEqual(actions[0]["command"], cmd)

    def test_bench_train_is_exactly_one_action(self):
        cmd = "python3 bench.py train --task dspy --rows data/x.jsonl --tag t1"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "train")
        self.assertEqual(actions[0]["tool"], "sft")
        self.assertEqual(actions[0]["args"], {"rows": "data/x.jsonl", "tag": "t1"})

    def test_bench_train_tolerates_extra_whitespace(self):
        cmd = "python3 bench.py    train --task dspy --rows data/x.jsonl --tag t1"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "sft")

    def test_pipeline_train_py_also_maps_to_sft(self):
        cmd = "modal run pipeline/train.py::train_entry --task openclaw --rows data/x.jsonl --tag t1"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "train")
        self.assertEqual(actions[0]["tool"], "sft")

    def test_modal_run_pipeline_rl_py(self):
        cmd = "modal run pipeline/rl.py::rl_entry --task dspy --tag t1 --num-rollout 24"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "train")
        self.assertEqual(actions[0]["tool"], "rl")
        self.assertEqual(actions[0]["args"], {"tag": "t1", "num_rollout": "24"})

    def test_bench_rl_maps_to_train_rl(self):
        cmd = "python3 bench.py rl --task dspy --tag t1 --num-rollout 24"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "train")
        self.assertEqual(actions[0]["tool"], "rl")

    def test_rubric_eval_dev_flag_extracted(self):
        cmd = ("python3 toolbox/eval_toolbox/rubric_eval.py --dev tasks/fav2/dev.json "
               "--answers ans.json --task fav2 --out results.json")
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "eval")
        self.assertEqual(actions[0]["tool"], "rubric_eval")
        self.assertEqual(actions[0]["args"].get("dev"), "tasks/fav2/dev.json")

    def test_bench_score_maps_to_eval_bench_score(self):
        cmd = "python3 bench.py score --task fav2 --model /out/models/t1/merged --split dev --tag t1"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "eval")
        self.assertEqual(actions[0]["tool"], "bench_score")
        self.assertEqual(actions[0]["args"].get("tag"), "t1")

    def test_run_recipe_method_flag_extracted(self):
        cmd = "python3 toolbox/evolve/run_recipe.py --recipe recipes/foo.json --method paraphrase"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "evolve")
        self.assertEqual(actions[0]["tool"], "run_recipe")
        self.assertEqual(actions[0]["args"].get("method"), "paraphrase")

    def test_chained_command_hits_two_in_order(self):
        cmd = ("cd ws && python3 toolbox/data_toolbox/gen/paraphrase.py --out o "
               "&& python3 bench.py train --task dspy --rows data/x.jsonl --tag t1")
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 2)
        self.assertEqual((actions[0]["kind"], actions[0]["tool"]), ("data", "paraphrase"))
        self.assertEqual((actions[1]["kind"], actions[1]["tool"]), ("train", "sft"))
        # flags belong to the sub-command they trail, not the whole chain
        self.assertEqual(actions[0]["args"], {})
        self.assertEqual(actions[1]["args"], {"rows": "data/x.jsonl", "tag": "t1"})

    def test_no_matches_returns_empty_list(self):
        self.assertEqual(learning.classify_command("ls -la && echo hi"), [])

    def test_empty_command_returns_empty_list(self):
        self.assertEqual(learning.classify_command(""), [])

    def test_flag_equals_value_form(self):
        cmd = "python3 bench.py train --task=dspy --rows=data/x.jsonl --tag=t1"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["args"], {"rows": "data/x.jsonl", "tag": "t1"})

    def test_quoted_flag_value_with_spaces(self):
        cmd = 'python3 bench.py train --task dspy --rows data/x.jsonl --tag "release one"'
        actions = learning.classify_command(cmd)
        self.assertEqual(actions[0]["args"]["tag"], "release one")

    def test_missing_flags_are_omitted_not_null(self):
        cmd = "python3 toolbox/data_toolbox/gen/paraphrase.py --corpus c --out o"
        actions = learning.classify_command(cmd)
        self.assertEqual(actions[0]["args"], {})

    def test_bare_bench_py_other_subcommands_do_not_match(self):
        # bench.py eval/judge/freeze/verify/leaderboard are not one of the
        # four learning kinds, and the bare substring "bench.py" alone must
        # never match (only the two-token train/rl/score phrases do).
        for sub in ("eval", "judge", "freeze", "verify", "leaderboard"):
            with self.subTest(sub=sub):
                cmd = f"python3 bench.py {sub} --task dspy"
                self.assertEqual(learning.classify_command(cmd), [])

    def test_env_var_and_cd_prefixes_dont_block_a_match(self):
        cmd = "cd /root/ws && FOO=bar python3 toolbox/data_toolbox/gen/mix.py --pools a,b --out o"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "mix")

    def test_uv_run_prefix_still_matches(self):
        cmd = "uv run toolbox/data_toolbox/gen/reasoning.py --corpus c --out o"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "reasoning")

    def test_bash_c_quoting_still_matches(self):
        cmd = "bash -c 'cd ws && python3 toolbox/data_toolbox/gen/hygiene.py --in a --out b'"
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool"], "hygiene")

    def test_repeated_datagen_call_is_two_actions_not_deduped(self):
        cmd = ("python3 toolbox/data_toolbox/gen/paraphrase.py --out o1 && "
               "python3 toolbox/data_toolbox/gen/paraphrase.py --out o2")
        actions = learning.classify_command(cmd)
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(a["tool"] == "paraphrase" for a in actions))


class TestExtractScriptPaths(unittest.TestCase):
    def test_excludes_registry_matches_orders_and_dedups(self):
        cmd = ("cd ws && python3 toolbox/data_toolbox/gen/paraphrase.py --out o "
               "&& python3 tools/mystery_a.py --x 1 "
               "&& bash tools/mystery_b.sh "
               "&& python3 tools/mystery_a.py --y 2")
        paths = learning.extract_script_paths(cmd)
        self.assertEqual(paths, ["tools/mystery_a.py", "tools/mystery_b.sh"])

    def test_bench_py_train_script_token_excluded_too(self):
        # "bench.py" the bare token overlaps the two-token "bench.py train"
        # registry match span even though the match itself isn't a single
        # contiguous script token.
        cmd = "python3 bench.py train --task dspy --rows data/x.jsonl --tag t1"
        self.assertEqual(learning.extract_script_paths(cmd), [])

    def test_non_script_paths_are_not_surfaced(self):
        cmd = ("python3 toolbox/data_toolbox/gen/paraphrase.py "
               "--corpus tasks/fav2/corpus --out data/fav2_paraphrase.rows.jsonl")
        # tasks/fav2/corpus and the .jsonl output both contain "/", but only
        # .py/.sh tokens are scripts worth surfacing for invented-tool checks.
        self.assertEqual(learning.extract_script_paths(cmd), [])

    def test_no_scripts_returns_empty_list(self):
        self.assertEqual(learning.extract_script_paths("ls -la && echo hi"), [])

    def test_strips_common_quoting(self):
        cmd = 'python3 "tools/quoted_invented.py" --x 1'
        self.assertEqual(learning.extract_script_paths(cmd), ["tools/quoted_invented.py"])


# ---------------------------------------------------------------------------
# Task 6: classifier wiring + invented-tool detection in collect.build_record.
# These build a real record from a synthetic temp run dir (following
# test_ingest.py's fixture pattern) to exercise the whole pipeline: command
# extraction from tool_use blocks, seed-manifest loading, provenance, invented
# detection against the workspace snapshot, nth_use, and learning_counts.
# ---------------------------------------------------------------------------

RUN_ID = "claude_fav2_20260720T000000"


def _sys_init() -> dict:
    return {"type": "system", "subtype": "init", "tools": [{"name": "Bash"}],
            "model": "claude-x", "cwd": "/ws"}


def _claude_bash(cmd: str, call_id: str) -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": call_id, "name": "Bash",
         "input": {"command": cmd}}]}}


def _codex_exec(argv: list[str], call_id: str) -> dict:
    # codex legacy `msg` vocab: exec_command_begin carries an argv list, which
    # codex.py joins (shlex) into the tool_use block's input["command"].
    return {"msg": {"type": "exec_command_begin", "call_id": call_id,
                    "command": argv}}


def _make_run(root: Path, trace_lines: list[dict], *, manifest=None,
              ws_files=None) -> Path:
    """Lay out root/workspace/agents/_runs/<RUN_ID> with trace.jsonl + an
    arrival-time sidecar; optional seed_manifest.txt at ws_root.parent (== root,
    where the seeding routine writes it) and workspace files (for the snapshot)."""
    run_dir = root / "workspace" / "agents" / "_runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in trace_lines))
    obs = run_dir / ".obs"
    obs.mkdir()
    (obs / "line_ts.jsonl").write_text("".join(
        json.dumps({"line": n, "ts": f"2026-07-20T00:00:{n:02d}Z"}) + "\n"
        for n in range(1, len(trace_lines) + 1)))
    if manifest is not None:
        (root / "seed_manifest.txt").write_text(
            "".join(f"100644 blob {'0' * 40}\t{p}\n" for p in manifest))
    for rel, content in (ws_files or {}).items():
        p = root / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return run_dir


class TestBuildRecordLearningWiring(unittest.TestCase):
    def test_claude_registry_actions_order_and_counts(self):
        trace = [
            _sys_init(),
            _claude_bash("python3 toolbox/data_toolbox/gen/paraphrase.py "
                         "--corpus tasks/fav2/corpus --out data/x.jsonl", "b1"),
            _claude_bash("python3 bench.py train --rows r.jsonl --tag t1", "b2"),
            _claude_bash("python3 toolbox/eval_toolbox/rubric_eval.py "
                         "--dev dev.json --answers a.json --task fav2", "b3"),
            _claude_bash("python3 toolbox/data_toolbox/gen/mix.py --out o "
                         "&& python3 bench.py score --tag t2", "b4"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_run(root, trace, manifest=["bench.py"])
            rec, _ = collect.build_record(root)

        actions = rec["learning"]
        self.assertEqual(
            [(a["kind"], a["tool"]) for a in actions],
            [("data", "paraphrase"), ("train", "sft"), ("eval", "rubric_eval"),
             ("data", "mix"), ("eval", "bench_score")])
        # the chained && command's two hits share one event; others are distinct
        self.assertEqual([a["event_i"] for a in actions], [1, 2, 3, 4, 4])
        self.assertTrue(all(a["provenance"] == "seed" for a in actions))
        # ts flows from the owning event (arrival-time sidecar), never None here
        events = rec["events"]
        for a in actions:
            self.assertIsNotNone(a["ts"])
            self.assertEqual(a["ts"], events[a["event_i"]]["ts"])
        # flags are scoped to their sub-command, not leaked across the chain
        self.assertEqual(actions[1]["args"], {"rows": "r.jsonl", "tag": "t1"})
        self.assertEqual(actions[4]["args"], {"tag": "t2"})
        self.assertEqual(actions[0]["args"], {})  # datagen has no surfaced flags
        # command is carried (and length-bounded)
        self.assertTrue(actions[0]["command"].startswith("python3 toolbox"))
        self.assertTrue(all(len(a["command"]) <= 500 for a in actions))
        # every action's first (and only) use here is nth_use 1
        self.assertTrue(all(a["nth_use"] == 1 for a in actions))
        self.assertEqual(rec["index_row"]["learning_counts"],
                         {"data": 2, "train": 1, "eval": 2, "evolve": 0,
                          "harness": 0, "infra": 0, "invented_tools": 0})

    def test_invented_detection_and_seed_in_same_run(self):
        trace = [
            _sys_init(),
            _claude_bash("python3 scripts/mytool.py --x 1", "b1"),
            _claude_bash("python3 bench.py train --rows r.jsonl --tag t1", "b2"),
            _claude_bash("python3 scripts/mytool.py --x 2", "b3"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_run(root, trace,
                      # manifest deliberately WITHOUT scripts/mytool.py
                      manifest=["bench.py", "AGENTS.md",
                                "toolbox/data_toolbox/gen/paraphrase.py"],
                      ws_files={"scripts/mytool.py": "print('hi')\n"})
            rec, _ = collect.build_record(root)  # include_workspace default True

        actions = rec["learning"]
        invented = [a for a in actions if a["provenance"] == "invented"]
        self.assertEqual(len(invented), 2)
        for a in invented:
            self.assertEqual(a["kind"], "tool")
            self.assertEqual(a["tool"], "scripts/mytool.py")
            self.assertEqual(a["args"], {})
        self.assertEqual([a["nth_use"] for a in invented], [1, 2])
        self.assertEqual([a["event_i"] for a in invented], [1, 3])
        # the registry match in the same run is provenance "seed" (manifest present)
        seed = [a for a in actions if a["provenance"] == "seed"]
        self.assertEqual(len(seed), 1)
        self.assertEqual((seed[0]["kind"], seed[0]["tool"]), ("train", "sft"))
        counts = rec["index_row"]["learning_counts"]
        self.assertEqual(counts["invented_tools"], 1)  # distinct paths, not uses
        self.assertEqual(counts["train"], 1)

    def test_invented_tool_invoked_by_absolute_path_is_relativized(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws_root = (root / "workspace").resolve()
            abs_path = ws_root / "scripts" / "mytool.py"
            trace = [_sys_init(),
                     _claude_bash(f"python3 {abs_path} --x 1", "b1")]
            _make_run(root, trace, manifest=["bench.py"],
                      ws_files={"scripts/mytool.py": "print('hi')\n"})
            rec, _ = collect.build_record(root)

        invented = [a for a in rec["learning"] if a["provenance"] == "invented"]
        self.assertEqual(len(invented), 1)
        self.assertEqual(invented[0]["tool"], "scripts/mytool.py")  # prefix stripped

    def test_no_manifest_registry_unknown_and_no_invented(self):
        trace = [
            _sys_init(),
            _claude_bash("python3 bench.py train --rows r.jsonl --tag t1", "b1"),
            _claude_bash("python3 scripts/mytool.py --x 1", "b2"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # NO manifest; the invented-looking script IS in the workspace
            _make_run(root, trace, ws_files={"scripts/mytool.py": "print('hi')\n"})
            rec, _ = collect.build_record(root)

        actions = rec["learning"]
        self.assertTrue(actions)  # the registry match is still classified
        self.assertTrue(all(a["provenance"] == "unknown" for a in actions))
        self.assertFalse(any(a["kind"] == "tool" for a in actions))  # invented skipped
        self.assertEqual(rec["index_row"]["learning_counts"]["invented_tools"], 0)

    def test_snapshot_none_skips_invented_but_keeps_seed(self):
        trace = [
            _sys_init(),
            _claude_bash("python3 bench.py train --rows r.jsonl --tag t1", "b1"),
            _claude_bash("python3 scripts/mytool.py --x 1", "b2"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_run(root, trace, manifest=["bench.py"],
                      ws_files={"scripts/mytool.py": "print('hi')\n"})
            # include_workspace=False -> ws_snapshot None -> noise guard unavailable
            rec, _ = collect.build_record(root, include_workspace=False)

        actions = rec["learning"]
        self.assertEqual(len(actions), 1)
        self.assertEqual((actions[0]["kind"], actions[0]["tool"]), ("train", "sft"))
        self.assertEqual(actions[0]["provenance"], "seed")  # manifest present
        self.assertFalse(any(a["kind"] == "tool" for a in actions))  # no unguarded guess

    def test_codex_format_command_classifies(self):
        trace = [
            {"msg": {"type": "session_configured", "session_id": "s1",
                     "model": "gpt-5"}},
            _codex_exec(["python3", "bench.py", "score", "--task", "fav2",
                         "--tag", "t1"], "c1"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_run(root, trace)  # no manifest -> provenance unknown
            rec, _ = collect.build_record(root, include_workspace=False)

        self.assertEqual(rec["meta"]["trace_format"], "codex-events")
        actions = rec["learning"]
        self.assertEqual(len(actions), 1)
        self.assertEqual((actions[0]["kind"], actions[0]["tool"]),
                         ("eval", "bench_score"))
        self.assertEqual(actions[0]["event_i"], 1)  # after the init event
        self.assertEqual(actions[0]["args"].get("tag"), "t1")

    def test_empty_learning_run_has_none_counts(self):
        trace = [
            _sys_init(),
            _claude_bash("ls -la /ws", "b1"),
            _claude_bash("echo hi && cat notes.txt", "b2"),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_run(root, trace, manifest=["bench.py"])
            rec, _ = collect.build_record(root)

        self.assertEqual(rec["learning"], [])
        self.assertIsNone(rec["index_row"]["learning_counts"])

    def test_malformed_tool_input_never_crashes(self):
        # input missing / non-dict / command non-string must all be skipped
        # gracefully, and unrelated tools ignored — no learning action, no raise.
        trace = [
            _sys_init(),
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "b1", "name": "Bash"}]}},         # no input
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "b2", "name": "Bash",
                 "input": "not-a-dict"}]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "b3", "name": "Bash",
                 "input": {"command": ["not", "a", "string"]}}]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "b4", "name": "Read",
                 "input": {"file_path": "bench.py"}}]}},                      # non-Bash
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_run(root, trace, manifest=["bench.py"])
            rec, _ = collect.build_record(root)
        self.assertEqual(rec["learning"], [])
        self.assertIsNone(rec["index_row"]["learning_counts"])


if __name__ == "__main__":
    unittest.main()
