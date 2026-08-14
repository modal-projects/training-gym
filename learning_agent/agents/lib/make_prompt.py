#!/usr/bin/env python3
"""Assemble the prompt handed to a contestant agent for a Learning Agent run.

The prompt is the workspace's AGENTS.md — already stitched concrete at
seeding (workspace_setup/setup_agent_md.py) — prefixed with a short
evaluation preamble that states the assignment, the working directory, and
the stop condition. This script never edits the spec; it only frames it.

Usage: make_prompt.py --task <task> --root <workspace> [--hours N] [--track easy|medium|hard]
Prints the assembled prompt to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PREAMBLE = """\
You are an AI agent being evaluated on the Learning Agent benchmark. This is a real, scored run.

- Your assigned task is `{task}` (already filled in throughout the specification below).
- Your working directory is the repository root: `{root}`. Operate there —
  this repository is what gets scored.
- You have `{hours}` hours of wall-clock. Run `bash timer.sh` at any time to see how
  much remains. When it reaches zero, stop — later work will not be scored.
- Work autonomously to the end of your budget: there is no user to answer questions.
  Do not stop early after one iteration; keep improving until time runs low.
- When you are finished, make sure `submission/eval.py` runs under the exact contract
  command and is wired to your best system, record your experiments in
  `runs/LEARNING_LOG.jsonl`, then end your session.

HEADLESS EXECUTION — READ CAREFULLY. This session is non-interactive: when your turn
ends, the whole run terminates immediately and any background work is killed. Therefore:
  - Run long jobs (training, GPU evals) in the FOREGROUND
    and wait for them to return. They block for many minutes — that is expected; stay in
    the turn. Do NOT launch them with `&`/`nohup`, and do NOT delegate the wait to a
    background sub-task or "notification" — ending your turn to wait will kill the job.
  - Only end your turn when the timer is nearly spent, or you have registered the best
    checkpoint you can produce. Never end a turn merely to "wait" for something.

Your full task specification follows.

========================================================================
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--hours", default="24")
    ap.add_argument("--track", default="easy",
                    help="accepted for interface stability; the spec is already stitched")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    spec_path = root / "AGENTS.md"
    if not spec_path.exists():
        raise SystemExit(f"no stitched spec at {spec_path} — seed the workspace first "
                         "(workspace_setup/prepare_workspace.sh)")
    preamble = PREAMBLE.format(task=args.task, root=root, hours=args.hours)
    print(preamble + spec_path.read_text())


if __name__ == "__main__":
    main()
