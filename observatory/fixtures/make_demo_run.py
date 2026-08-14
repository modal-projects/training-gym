#!/usr/bin/env python3
"""Build the observatory demo fixtures from ptb_slice.json.

Produces two things (both git-committed, both reproducible from this script):

1. fixtures/demo/ws_claude_dspy_20260717T090000/ — a faithful LAB run dir
   (the exact layout agents/run_clone.sh + agents/run.sh produce), whose
   trace.jsonl is genuine Claude Code stream-json reconstructed from a real
   trace. This is the ingestion CLI's end-to-end test input. Trace CONTENT is
   sample data from a public PostTrainBench trajectory, not a real LAB run —
   except for a handful of Bash tool_use/tool_result events spliced in by
   _inject_learning_events (data/train/eval seed-tool calls + one invented
   script used twice), added so the demo's Learning tab has real content;
   see run_meta.json/seed_manifest.txt, also written into this dir.

2. fixtures/sample_record.json / sample_workspace.json / sample_status.json —
   already-normalized outputs conforming to observatory/schema.py, so the
   frontend can be developed (via static_dev_server.py) against realistic
   data without needing the collector at request time. `events` and
   `learning`/`learning_counts` are read back from the real collector run
   over (1) above, not re-derived by hand — see build_sample_normalized.

Usage: python3 observatory/fixtures/make_demo_run.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for observatory.schema

from observatory import schema  # noqa: E402
from observatory.normalize import collect  # noqa: E402

RUN_STAMP = "20260717T090000"
RUN_NAME = f"claude_dspy_{RUN_STAMP}"
WS_DIR = HERE / "demo" / f"ws_{RUN_NAME}"
RUN_ID = RUN_NAME
TAG = "demo_qa_v1"
SESSION_ID = None  # filled from slice

# WS_DIR is the sandbox's $RUN_PARENT (contains workspace/ as a sibling of
# run_meta.json — see workspace_setup/prepare_workspace.sh step 5). Track "easy" matches the demo's
# scaffold/task; prepared_at is derived from RUN_STAMP, not invented.
TRACK = "easy"
PREPARED_AT = (f"{RUN_STAMP[0:4]}-{RUN_STAMP[4:6]}-{RUN_STAMP[6:8]}T"
               f"{RUN_STAMP[9:11]}:{RUN_STAMP[11:13]}:{RUN_STAMP[13:15]}Z")

# Task 7: a few learning-timeline actions spliced into the borrowed PTB trace
# (see _inject_learning_events) so the demo run's Learning tab has real seed
# (data/train/eval) AND invented-tool content to render. GEN_STEM is a real
# toolbox/data_toolbox/gen/*.py generator; INVENTED_SCRIPT_REL is a script the
# demo agent "wrote" mid-run: present in the workspace snapshot, deliberately
# left out of SEED_FILES (-> seed_manifest.txt), so it classifies as invented.
GEN_STEM = "grounded_qa"
INVENTED_SCRIPT_REL = "scripts/quick_dev_probe.py"
SEED_FILES = [
    "AGENTS.md", "bench.py", "README.md",
    f"toolbox/data_toolbox/gen/{GEN_STEM}.py",
    "toolbox/eval_toolbox/rubric_eval.py",
    "submission/eval.py", "timer.sh",
]

PROVENANCE = {
    "judge_model": "claude-opus-4-20250514",
    "pinned_judge_model": "claude-opus-4-20250514",
    "judge_backend": "cli-claude",
    "canonical": False,
    "n_votes": 3,
    "judge_prompt_sha": "66cbb968" + "0" * 56,
    "harness_sha": "7458d986" + "0" * 56,
    "sys_sha": "35827112" + "0" * 56,
    "gold_sha": "a9576aad" + "0" * 56,
    "config_sha": "5c256fe8" + "0" * 56,
    "corpus_pin": "v3.2.1",
    "budget": 5,
    "limit": 0,
    "eval_temperature": 0.0,
    "seed": 0,
    "integrity": "OK",
    "timestamp": "2026-07-17T11:05:00+00:00",
}

QIDS = [f"dspy_{i:012x}" for i in (0x24049972DFC, 0x3A1B2C3D4E5, 0x5F6A7B8C9DA, 0x7E8F9A0B1C2, 0x91A2B3C4D5E, 0xB3C4D5E6F7A)]

# Bash tool_use calls spliced into the borrowed PTB trace: (tool_id, ts_call,
# ts_result, command, description, result_text). Timestamped into the
# multi-hour gap between the borrowed trajectory's last tool call
# (task_started at 2026-06-06T20:16:56Z) and its final result
# (2026-06-07T05:51:15Z) — narratively "meanwhile, the LAB data/train/eval
# loop for the dspy task ran in parallel." Two calls hit INVENTED_SCRIPT_REL
# (nth_use 1 and 2); the rest are seed-tool registry hits (data/train/eval).
_LEARNING_CALLS = [
    ("toolu_demo_learn_data", "2026-06-06T21:05:00Z", "2026-06-06T21:05:04Z",
     'python3 toolbox/data_toolbox/gen/grounded_qa.py --corpus tasks/dspy/corpus '
     '--glob "**/*.py" --n 60 --seed 0 --backend mock '
     '--out data/dspy_grounded_qa.rows.jsonl',
     "Generate grounded QA training rows from the dspy corpus",
     "wrote 60 rows -> data/dspy_grounded_qa.rows.jsonl"),
    ("toolu_demo_learn_train", "2026-06-06T22:10:00Z", "2026-06-06T22:10:05Z",
     "python3 bench.py train --task dspy --rows data/dspy_grounded_qa.rows.jsonl "
     "--tag demo_qa_v1 --epochs 3 --lr 2e-4",
     "SFT a LoRA student on the generated QA rows",
     "SFT complete: adapter saved to /out/models/demo_qa_v1/merged"),
    ("toolu_demo_learn_probe1", "2026-06-06T23:40:00Z", "2026-06-06T23:40:03Z",
     f"python3 {INVENTED_SCRIPT_REL} --answers runs/demo_qa_v1/budget_5/candidates.json",
     "Quick sanity check of the candidates before the full judge pass",
     "quick probe: 5/5 candidates present, avg length 210 words"),
    ("toolu_demo_learn_eval", "2026-06-07T00:15:00Z", "2026-06-07T00:15:20Z",
     "python3 toolbox/eval_toolbox/rubric_eval.py --dev tasks/dspy/dev.json "
     "--answers runs/demo_qa_v1/budget_5/candidates.json --task dspy --n-votes 3 "
     "--judge-model claude-opus-4-20250514 --judge-backend cli "
     "--out runs/demo_qa_v1/budget_5/results_dev.json",
     "Run the canonical judge over the candidates",
     "dev mean 0.57 [0.31, 0.72] over 5/6 questions (1 failed)"),
    ("toolu_demo_learn_probe2", "2026-06-07T01:00:00Z", "2026-06-07T01:00:04Z",
     f"python3 {INVENTED_SCRIPT_REL} --answers runs/demo_qa_v1/budget_5/candidates.json --verbose",
     "Re-check candidates after the judge run",
     "quick probe: 5/5 candidates present, avg length 214 words"),
]


def _learning_action_events(session_id: str | None) -> list[dict]:
    """ptb_slice.json-shaped event dicts (see reconstruct_stream_line) for
    _LEARNING_CALLS, preceded by one thinking/text pair introducing them."""
    events = [
        {"ts": "2026-06-06T21:00:00Z", "type": "assistant", "session_id": session_id,
         "session_idx": 0, "parent_tool_use_id": None,
         "blocks": [{"type": "thinking",
                     "thinking": "The baseline AIME eval is running in the background. "
                                 "While that's going, let's also run the LAB data/train/"
                                 "eval loop end to end for the dspy task."}],
         "usage": {}, "model": "claude-opus-4-8", "uuid": "uuid-demo-learn-think"},
        {"ts": "2026-06-06T21:00:02Z", "type": "assistant", "session_id": session_id,
         "session_idx": 0, "parent_tool_use_id": None,
         "blocks": [{"type": "text",
                     "text": "Meanwhile: generate training data from the dspy corpus, "
                             "fine-tune, and score against the dev set."}],
         "usage": {}, "model": "claude-opus-4-8", "uuid": "uuid-demo-learn-text"},
    ]
    for tool_id, ts_call, ts_result, cmd, desc, result in _LEARNING_CALLS:
        # Short, collision-free uuid: reconstruct_stream_line derives the
        # reconstructed message "id" from this by stripping dashes and
        # truncating to 24 chars — tool_id itself ("toolu_demo_learn_probe1"
        # vs "...probe2") is too long and collides after that truncation.
        short = tool_id.removeprefix("toolu_demo_learn_")
        events.append({
            "ts": ts_call, "type": "assistant", "session_id": session_id,
            "session_idx": 0, "parent_tool_use_id": None,
            "blocks": [{"type": "tool_use", "id": tool_id, "name": "Bash",
                        "input": {"command": cmd, "description": desc}}],
            "usage": {}, "model": "claude-opus-4-8", "uuid": f"uuid-demo-{short}",
        })
        events.append({
            "ts": ts_result, "type": "user", "session_id": session_id,
            "session_idx": 0, "parent_tool_use_id": None,
            "blocks": [{"type": "tool_result", "tool_use_id": tool_id,
                        "content": result, "is_error": False}],
        })
    return events


def _inject_learning_events(events: list[dict]) -> list[dict]:
    """Splice _learning_action_events in right before the trajectory's final
    "result" event (never after — a result event ends the trace)."""
    session_id = next((e.get("session_id") for e in events if e.get("session_id")), None)
    idx = next(i for i, e in enumerate(events) if e.get("type") == "result")
    return events[:idx] + _learning_action_events(session_id) + events[idx:]


def reconstruct_stream_line(ev: dict) -> dict | None:
    """Map one ptb-normalized event back to a Claude Code stream-json line."""
    t = ev["type"]
    sid = ev.get("session_id")
    if t == "system" and ev.get("subtype") == "init":
        return {
            "type": "system", "subtype": "init", "cwd": str(WS_DIR / "workspace"),
            "session_id": sid, "tools": ["Task", "Bash", "Glob", "Grep", "Read",
                                         "Edit", "Write", "WebFetch", "TodoWrite"],
            "mcp_servers": [], "model": "claude-opus-4-8",
            "permissionMode": "bypassPermissions", "apiKeySource": "none",
        }
    if t == "assistant":
        blocks = [dict(b) for b in ev.get("blocks", [])]
        return {
            "type": "assistant",
            "message": {
                "id": "msg_" + (ev.get("uuid", "x") or "x").replace("-", "")[:24],
                "type": "message", "role": "assistant",
                "model": ev.get("model", "claude-opus-4-8"),
                "content": blocks, "stop_reason": None, "stop_sequence": None,
                "usage": ev.get("usage", {}),
            },
            "parent_tool_use_id": ev.get("parent_tool_use_id"),
            "session_id": sid, "uuid": ev.get("uuid"),
        }
    if t == "user":
        blocks = [dict(b) for b in ev.get("blocks", [])]
        if not blocks:
            return None
        return {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": ev.get("parent_tool_use_id"),
            "session_id": sid,
        }
    if t == "result":
        return {
            "type": "result", "subtype": ev.get("subtype", "success"),
            "is_error": False, "duration_ms": ev.get("duration_ms"),
            "duration_api_ms": ev.get("duration_ms"),
            "num_turns": ev.get("num_turns"),
            "result": ev.get("result", "Run complete."),
            "session_id": sid, "total_cost_usd": ev.get("total_cost_usd"),
            "usage": ev.get("usage", {}),
            "stop_reason": ev.get("stop_reason"),
        }
    return None  # rate_limit_event etc. — not part of claude stream output


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_json(path: Path, obj) -> None:
    write(path, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def build_run_dir(slice_: dict) -> None:
    if WS_DIR.exists():
        shutil.rmtree(WS_DIR)
    ws = WS_DIR / "workspace"
    run_dir = ws / "agents" / "_runs" / RUN_NAME

    # --- run_meta.json (prepare_workspace.sh step 5, written into $RUN_PARENT) ---
    write_json(WS_DIR / "run_meta.json", {
        "track": TRACK, "scaffold": "claude", "task": "dspy",
        "hours": "24", "prepared_at": PREPARED_AT,
    })

    # --- seed_manifest.txt (`git -C "$ROOT" ls-tree -r HEAD`, written into
    # $RUN_PARENT) --- deliberately omits INVENTED_SCRIPT_REL, which IS in the
    # workspace snapshot below: that gap is what makes it classify as invented.
    write(WS_DIR / "seed_manifest.txt",
          "".join(f"100644 blob {'0' * 40}\t{p}\n" for p in SEED_FILES))

    # --- trace.jsonl + arrival-time sidecar -------------------------------
    lines, ts_rows = [], []
    for ev in slice_["events"]:
        line = reconstruct_stream_line(ev)
        if line is None:
            continue
        lines.append(json.dumps(line, ensure_ascii=False))
        if ev.get("ts"):
            ts_rows.append({"line": len(lines), "ts": ev["ts"]})
    write(run_dir / "trace.jsonl", "\n".join(lines) + "\n")
    write(run_dir / ".obs" / "line_ts.jsonl",
          "\n".join(json.dumps(r) for r in ts_rows) + "\n")

    # --- watcher-shaped local system samples (cpu-only) --------------------
    mon = []
    for s in slice_["system_monitor"]:
        mon.append({"ts": s["ts"], "gpu": s.get("gpu"),
                    "cpu_load_1m": s.get("cpu_load_1m"),
                    "cpu_load_5m": s.get("cpu_load_5m"),
                    "mem_used_gib": s.get("mem_used_gib"),
                    "mem_total_gib": s.get("mem_total_gib"),
                    "source": "local-watcher"})
    write(run_dir / ".obs" / "system_monitor.jsonl",
          "\n".join(json.dumps(r) for r in mon) + "\n")

    # --- run.sh artifacts ---------------------------------------------------
    write(run_dir / "prompt.txt",
          "We want to train the student model Qwen/Qwen3.5-9B to answer expert "
          "questions about a domain corpus it has never seen. [demo fixture — "
          "truncated task spec]\n")
    write(run_dir / "solve.err", "")
    write(run_dir / "solve_status.txt", "exit=0\nseconds=2903\n")

    ckpt_before = [
        {"tag": "oc_pilot_qa_v1", "task": "openclaw",
         "model_path": "/out/models/oc_pilot_qa_v1/merged",
         "dev_score": 0.0188, "method": "pilot row from earlier run (fixture)"},
    ]
    new_row = {"tag": TAG, "task": "dspy",
               "model_path": f"/out/models/{TAG}/merged", "dev_score": 0.2104,
               "method": "Demo: LoRA-SFT (r32 a64 lr2e-4 3ep, seed 0) on 60 "
                         "corpus-grounded QA rows (fixture data)."}
    write(run_dir / "checkpoints.before",
          "\n".join(json.dumps(r) for r in ckpt_before) + "\n")
    write(run_dir / "checkpoints.after",
          "\n".join(json.dumps(r) for r in ckpt_before + [new_row]) + "\n")
    write(run_dir / "submitted.jsonl", json.dumps(new_row) + "\n")

    write_json(run_dir / "audit.json", {
        "task": "dspy", "integrity": "CLEAN", "findings": [],
        "access_counts": {"reads_test": 0, "split_test_runs": 0,
                          "source_pack_touches": 0},
        "behavior": {"agent_tool_calls": 41, "train_launches_seen": 1,
                     "rl_launches_seen": 0, "score_launches_seen": 2},
        "caveat": "tool calls made by sub-agents are not visible in the trace",
    })

    # --- workspace results tree (what the agent's bench.py score left) -----
    wruns = ws / "runs"
    write(wruns / "CHECKPOINTS.jsonl",
          "\n".join(json.dumps(r) for r in ckpt_before + [new_row]) + "\n")

    per_question, verdicts_dir = {}, wruns / TAG / "budget_5" / "verdicts_dev"
    scores = [0.55, 0.7, None, 0.35, 0.8, 0.45]  # one failed (None) — never 0
    for qid, sc in zip(QIDS, scores):
        failed = sc is None
        claims = {f"c{k}": (None if failed else round(min(1.0, sc + 0.1 * (k % 2)), 2))
                  for k in range(1, 4)}
        per_question[qid] = {
            "failed": failed, "claim_score": sc,
            "verdicts": claims if not failed else {},
            "secondary": {"kind": "python_compiles",
                          "score": 0.0 if failed else 1.0, "detail": {}},
        }
        if not failed:
            write_json(verdicts_dir / f"{qid}.json", {
                "n_votes": 3, "backend": "cli-claude", "canonical": False,
                "votes": {c: [v, v, v] for c, v in claims.items()},
                "final": claims,
            })

    ok = [s for s in scores if s is not None]
    write_json(wruns / TAG / "budget_5" / "results_dev.json", {
        "mean": round(sum(ok) / len(ok), 4), "bootstrap_ci95": [0.31, 0.72],
        "n": len(ok), "n_failed": len(scores) - len(ok), "all_failed": False,
        "failed": [q for q, s in zip(QIDS, scores) if s is None],
        "secondary_metric": "python_compiles", "secondary_mean": 0.83,
        "canonical": False, "integrity": "OK", "provenance": PROVENANCE,
        "per_question": per_question,
    })
    write_json(wruns / TAG / "budget_5" / "candidates.json",
               {qid: f"Demo answer for {qid} (fixture)." for qid in QIDS})
    write_json(wruns / TAG / "budget_5" / "eval_meta.json",
               {qid: {"tool_calls": tc, "completion_tokens": 900 + 37 * i}
                for i, (qid, tc) in enumerate(zip(QIDS, [3, 5, 0, 4, 2, 5]))})

    write(wruns / "LEADERBOARD.jsonl", json.dumps({
        "task": "dspy", "tag": TAG, "split": "dev", "score": 0.57,
        "ci": [0.31, 0.72], "n": 5, "failed": False, "n_failed": 1,
        "secondary_mean": 0.83, "judge_model": "claude-opus-4-20250514",
        "backend": "cli-claude", "canonical": False, "integrity": "OK",
        "provenance": PROVENANCE}) + "\n")

    # --- a few workspace files so the snapshot has something to show -------
    write(ws / "tasks" / "dspy" / "ROUNDS.md",
          "# dspy rounds\n\n## Round 1 (demo)\n- LoRA-SFT r32 a64 lr2e-4 3ep "
          f"on 60 QA rows -> dev 0.57 [0.31,0.72], tag `{TAG}`\n- avg tool_calls 3.2 "
          "(search preserved)\n")
    write(ws / "submission" / "eval.py",
          '"""Demo submission stub (fixture)."""\nWEIGHTS = "/out/models/'
          f'{TAG}/merged"\n\n\ndef generate_answers(questions):\n'
          '    raise NotImplementedError("demo fixture")\n')
    write(ws / "README.md", "Demo workspace (fixture) for observatory ingest tests.\n")
    write(ws / "timer.sh", 'echo "Remaining time (hours:minutes):"\necho "0:00"\n')

    # --- invented tool (Task 7 demo): present in the workspace snapshot, absent
    # from seed_manifest.txt above -> classifies as an invented, non-seed tool.
    write(ws / INVENTED_SCRIPT_REL,
          '"""Ad hoc dev-set spot check the agent wrote instead of always going\n'
          'through toolbox/eval_toolbox/rubric_eval.py (fixture: an invented,\n'
          'non-seed tool, used twice in this demo run).\n"""\n'
          'import argparse\nimport json\n\n\n'
          'def main() -> None:\n'
          '    p = argparse.ArgumentParser()\n'
          '    p.add_argument("--answers", required=True)\n'
          '    p.add_argument("--verbose", action="store_true")\n'
          '    args = p.parse_args()\n'
          '    rows = json.loads(open(args.answers).read())\n'
          '    print(f"quick probe: {len(rows)}/{len(rows)} candidates present")\n\n\n'
          'if __name__ == "__main__":\n    main()\n')

    print(f"run dir: {run_dir}")
    print(f"  trace lines: {len(lines)}, ts rows: {len(ts_rows)}, monitor: {len(mon)}")


def build_sample_normalized(slice_: dict) -> None:
    """sample_record/workspace/status.json — target schema, for frontend dev.

    build_run_dir() already wrote a real run dir (trace.jsonl, seed_manifest.txt,
    run_meta.json, the workspace tree) to disk, so `events` (and `learning` /
    `learning_counts`, below) are pulled straight from the real collector
    instead of being hand-rolled a second time here: a single source of truth,
    so this hand-assembled sample can't drift out of sync with what
    observatory/normalize/collect.py actually produces from the same dir
    (in particular, `learning[].event_i` must index into the same `events`
    list a real ingest would produce for "jump to trace event" to land right).
    """
    real_rec, _ = collect.build_record(WS_DIR)
    events = real_rec["events"]
    learning_actions = real_rec["learning"]
    learning_counts = real_rec["index_row"]["learning_counts"]

    result_ev = next((e for e in reversed(events) if e["type"] == "result"), {})
    ses = slice_["sessions"][0]
    launched = events[0].get("ts") if events else None
    finished = result_ev.get("ts")

    results_path = (WS_DIR / "workspace" / "runs" / TAG / "budget_5"
                    / "results_dev.json")
    results = json.loads(results_path.read_text())
    eval_meta = json.loads((results_path.parent / "eval_meta.json").read_text())
    for qid, m in eval_meta.items():
        if qid in results["per_question"]:
            results["per_question"][qid].update(m)
    tcs = [m["tool_calls"] for m in eval_meta.values()]
    result_entry = dict(results, tag=TAG, split="dev", budget=5,
                        tool_calls_avg=round(sum(tcs) / len(tcs), 2))

    run_dir = WS_DIR / "workspace" / "agents" / "_runs" / RUN_NAME
    monitor = [json.loads(l) for l in
               (run_dir / ".obs" / "system_monitor.jsonl").read_text().splitlines()]
    audit = json.loads((run_dir / "audit.json").read_text())
    checkpoints = [json.loads(l) for l in
                   (WS_DIR / "workspace" / "runs" / "CHECKPOINTS.jsonl")
                   .read_text().splitlines()]
    leaderboard = [json.loads(l) for l in
                   (WS_DIR / "workspace" / "runs" / "LEADERBOARD.jsonl")
                   .read_text().splitlines()]

    summ = slice_["summary"]
    record = {
        "schema_version": schema.SCHEMA_VERSION,
        "index_row": {
            "run_id": RUN_ID, "kind": "agent_run", "state": schema.STATE_FINISHED,
            "task": "dspy", "scaffold": "claude", "agent_model": "claude-opus-4-8",
            "base_model": "Qwen/Qwen3.5-9B",
            "trace_format": schema.TRACE_CLAUDE, "time_budget_h": 24.0,
            "launched_at": launched, "finished_at": finished,
            "duration_s": 2903, "num_turns": result_ev.get("num_turns"),
            "num_events": len(events), "session_count": 1,
            "total_cost_usd": result_ev.get("total_cost_usd"),
            "best_dev_score": results["mean"],
            "best_dev_ci": results["bootstrap_ci95"], "best_tag": TAG,
            "track": TRACK, "learning_counts": learning_counts,
            "canonical": False, "integrity": "OK", "audit": "CLEAN",
            "has_system_monitor": True, "has_workspace": True,
            "updated_at": finished or "2026-07-17T12:00:00Z",
        },
        "meta": {
            "run_id": RUN_ID, "run_dir": str(run_dir), "scaffold": "claude",
            "task": "dspy", "base_model": "Qwen/Qwen3.5-9B",
            "trace_format": schema.TRACE_CLAUDE, "time_budget_h": 24.0,
            "launched_at": launched, "finished_at": finished, "exit_code": 0,
            "track": TRACK,
            "build_ts": "2026-07-17T12:00:00Z",
            "schema_version": schema.SCHEMA_VERSION,
        },
        "summary": {
            "agent_models": ["claude-opus-4-8"],
            "tools_offered": summ.get("tools_offered", [])[:12],
            "permission_mode": "bypassPermissions",
            "cwd": str(WS_DIR / "workspace"),
            "num_turns": result_ev.get("num_turns"),
            "duration_ms": result_ev.get("duration_ms"),
            "total_cost_usd": result_ev.get("total_cost_usd"),
            "usage_total": {k: v for k, v in (result_ev.get("usage") or {}).items()
                            if isinstance(v, int)},
            "stop_reasons": [result_ev.get("stop_reason") or "end_turn"],
            "final_result_text": (result_ev.get("result") or "")[:2000],
            "session_count": 1, "session_ids": [ses["session_id"]],
        },
        "sessions": [{
            "session_idx": 0, "session_id": ses["session_id"],
            "ts_start": launched, "model": "claude-opus-4-8",
            "cwd": str(WS_DIR / "workspace"),
            "permission_mode": "bypassPermissions",
            "tools": ses.get("tools", [])[:12],
        }],
        "events": events,
        "scores": {"checkpoints": checkpoints, "leaderboard": leaderboard,
                   "results": [result_entry]},
        "judgements": {"audit": audit},
        "system_monitor": monitor,
        "learning": learning_actions,
    }
    write_json(HERE / "sample_record.json", record)

    ws_root = WS_DIR / "workspace"
    files = []
    total = 0
    for p in sorted(ws_root.rglob("*")):
        if p.is_dir() or any(part in schema.WS_EXCLUDE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ws_root).as_posix()
        size = p.stat().st_size
        total += size
        inline = size <= schema.WS_INLINE_MAX_BYTES
        files.append({"path": rel, "size": size, "inline": inline,
                      "content": p.read_text(errors="replace") if inline else None,
                      "truncated": False})
    write_json(HERE / "sample_workspace.json", {
        "built_at": "2026-07-17T12:00:00Z", "root": str(ws_root),
        "total_files": len(files), "total_bytes": total,
        "inlined_files": sum(f["inline"] for f in files), "files": files,
    })

    write_json(HERE / "sample_status.json", {
        "run_id": RUN_ID, "state": schema.STATE_FINISHED,
        "updated_at": finished or "2026-07-17T12:00:00Z",
        "num_events": len(events), "last_event_ts": finished, "exit_code": 0,
    })
    print(f"sample_record.json: {len(events)} events, "
          f"{len(monitor)} monitor samples, {len(files)} workspace files")


def main() -> None:
    slice_ = json.loads((HERE / "ptb_slice.json").read_text())
    slice_["events"] = _inject_learning_events(slice_["events"])
    build_run_dir(slice_)
    build_sample_normalized(slice_)


if __name__ == "__main__":
    main()
