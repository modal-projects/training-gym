#!/usr/bin/env python3
"""Deterministic post-run audit of a contestant agent's trace.

PostTrainBench uses an LLM "contamination judge" because its forbidden thing —
"did you train on GSM8K?" — is fuzzy. Learning Agent's forbidden thing is exact: the hidden
test set lives at one known path. So the audit is a deterministic scan — but it must
scan the right thing.

CRUCIAL: a naive substring scan over the whole trace is WRONG. The trace contains the
agent reading AGENTS.md, bench/config.yaml, and `bench.py --help`, all of which quote
`tasks/<task>/test.json` and `--split test` (in the very rules that forbid them). The
agent's careful datagen prompts even say "DO NOT read test.json". Flagging any of that
is a false positive — it fired on the first real run and marked a clean agent dirty.

So we inspect only what the agent ITSELF chose to access — its tool-call INPUTS:
  - Read       -> the file_path it opened
  - Bash       -> the command it ran
  - Grep/Glob  -> the path/target it searched
We do NOT inspect tool RESULTS, file CONTENT, or the agent's authored prompts/scripts
(Task/Write bodies) — those legitimately mention the forbidden names.

A hit means the agent read the hidden test set or scored on the hidden split:
  - a Read whose path is tasks/<task>/test.json
  - a Bash command that reads that file (cat/head/…/python open/grep) or copies it
  - a Bash command that actually invokes `--split test` / `split=test`
  - a Grep/Glob targeting that file

LIMITATION (documented, not hidden): sub-agents spawned via the Task tool run their own
loops whose tool calls are NOT in this trace, so a read inside a sub-agent is invisible
here. Closing that needs the physical seal (replacing test.json in the agent's
environment). Until then this audit catches direct and naive-deliberate access; it is a
strong tripwire, not a sandbox.

Exit 0 = clean, 1 = contaminated, so run.sh / CI can gate on it.
Usage: audit_trace.py --trace <raw.jsonl> --task <T> --root <repo> [--out <verdict.json>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---- extracting what the AGENT actively accessed (tool-call inputs only) ----

def agent_accesses(trace_path: str) -> list[tuple[str, str]]:
    """Return [(kind, text)] for the agent's own accesses:
       ('read', path) | ('bash', command) | ('search', target).
    Parses Claude stream-json (assistant tool_use blocks) and, best-effort, Codex
    JSONL exec events. Ignores tool results, file content, and Task/Write bodies."""
    out: list[tuple[str, str]] = []
    for raw in Path(trace_path).read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(e, dict):
            continue

        # Claude stream-json: assistant message with tool_use blocks.
        if e.get("type") == "assistant":
            for b in e.get("message", {}).get("content", []):
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                name = b.get("name", "")
                inp = b.get("input", {}) or {}
                if name == "Read":
                    fp = inp.get("file_path") or inp.get("path")
                    if isinstance(fp, str):
                        out.append(("read", fp))
                elif name == "Bash":
                    cmd = inp.get("command")
                    if isinstance(cmd, str):
                        out.append(("bash", cmd))
                elif name in ("Grep", "Glob"):
                    tgt = inp.get("path") or inp.get("glob") or inp.get("pattern")
                    if isinstance(tgt, str):
                        out.append(("search", tgt))
                # Task/Write/Edit inputs are the agent's AUTHORED content, not an
                # access — deliberately skipped (they defensively mention test.json).

        # Codex JSONL: exec/command events (best-effort — field names vary).
        else:
            et = e.get("type", "")
            if "exec" in et or "command" in et or et.endswith("shell_call"):
                for k in ("command", "cmd", "shell", "input"):
                    v = e.get(k)
                    if isinstance(v, str):
                        out.append(("bash", v)); break
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        out.append(("bash", " ".join(v))); break
    return out


# ---- the forbidden-access predicates ----

def check(accesses: list[tuple[str, str]], task: str) -> tuple[list[str], dict]:
    """Return (findings, counts). A finding is a human-readable violation string."""
    test_paths = (f"tasks/{task}/test.json", "task/test.json")
    # reading verbs that pull a file's CONTENT into view
    read_cmd = re.compile(
        r"\b(cat|head|tail|less|more|nl|sed|awk|xxd|od|strings|jq|cp|mv|rsync|"
        r"open|read_csv|read_json|load|loads)\b")
    split_test = re.compile(r"--split[ =]+test\b|\bsplit\s*=\s*['\"]?test\b")

    # fav2's source pack (question provenance, gold answers, and the full test
    # membership) lives OUTSIDE the repo on this machine; touching it in any way
    # is contamination regardless of the assigned task.
    source_pack = re.compile(r"fav2_studypack")

    findings: list[str] = []
    counts = {"reads_test": 0, "bash_reads_test": 0, "split_test_runs": 0, "search_test": 0,
              "source_pack_touches": 0}

    for kind, text in accesses:
        if source_pack.search(text):
            counts["source_pack_touches"] += 1
            findings.append(f"{kind} touched the benchmark source pack: {text.strip()[:120]}")
        if kind == "read" and (any(tp in text for tp in test_paths) or text.endswith("/test.json")):
            counts["reads_test"] += 1
            findings.append(f"Read tool opened the hidden test set: {text}")
        elif kind == "search" and any(tp in text for tp in test_paths):
            counts["search_test"] += 1
            findings.append(f"Grep/Glob targeted the hidden test set: {text}")
        elif kind == "bash":
            if split_test.search(text):
                counts["split_test_runs"] += 1
                findings.append(f"Bash invoked hidden-split scoring: {text.strip()[:120]}")
            if any(tp in text for tp in test_paths) and read_cmd.search(text):
                counts["bash_reads_test"] += 1
                findings.append(f"Bash read/copied the hidden test set: {text.strip()[:120]}")
    return findings, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    accesses = agent_accesses(args.trace)
    findings, counts = check(accesses, args.task)
    contaminated = bool(findings)

    # Descriptive behavior (whole-trace substring tallies — NOT used for the verdict).
    text = Path(args.trace).read_text(errors="replace")
    def c(p): return len(re.findall(p, text))
    behavior = {
        "agent_tool_calls": len(accesses),
        "bash_calls": sum(1 for k, _ in accesses if k == "bash"),
        "reads": sum(1 for k, _ in accesses if k == "read"),
        "searches": sum(1 for k, _ in accesses if k == "search"),
        "train_launches_seen": c(r"bench\.py\s+train"),
        "rl_launches_seen": c(r"bench\.py\s+rl"),
        "score_launches_seen": c(r"bench\.py\s+score"),
    }

    verdict = {
        "task": args.task,
        "integrity": "CONTAMINATED" if contaminated else "CLEAN",
        "findings": findings,
        "access_counts": counts,
        "behavior": behavior,
        "caveat": "Sub-agent (Task tool) tool calls are not in this trace; a read inside "
                  "a spawned sub-agent is invisible here. Physical sealing of test.json "
                  "is the airtight control.",
    }

    out = json.dumps(verdict, indent=2)
    if args.out:
        Path(args.out).write_text(out + "\n")
    print(out)

    if contaminated:
        print(f"\n[audit] CONTAMINATION DETECTED ({len(findings)} finding(s)) — "
              "run not eligible for scoring.", file=sys.stderr)
        return 1
    print("\n[audit] clean — the agent made no direct access to the hidden test set.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
