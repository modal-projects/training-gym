#!/usr/bin/env python3
"""ONE dev-eval command for every task archetype — measures YOUR submission.

    python toolbox/eval_tool/dev_eval.py                 # dev split, default budget
    python toolbox/eval_tool/dev_eval.py --model /out/models/<tag>/merged
    python toolbox/eval_tool/dev_eval.py --trials 4      # agentic: up to the pin

The number this prints is the official instrument pointed at the dev split:
whatever is wired into submission/ right now is what gets measured, exactly
as final scoring will measure it. If you improved a harness but not the
score, the improvement is not wired in — that is signal, not noise.

How it dispatches (task/task_meta.yaml `archetype`):

  qa       runs `python submission/eval.py --input task/dev.json` (YOUR
           harness, YOUR serving via build()), then judges the answers with
           eval_tool/rubric_eval.py against dev gold through the pinned judge
           service. `--answers <file>` skips the production step and judges
           an existing answers JSON — EXPERIMENT MODE: that number is not a
           submission claim, because nothing guarantees the wired-in
           submission produces those answers.

  agentic  packages submission/ + toolbox/ (harness code only, no packages)
           and calls the operator's dev-rollout service (Modal app
           lab-rollout, fn dev_rollout). The service resolves the task
           config, env pin, user simulator, and DEV ids server-side, serves
           your checkpoint, imports YOUR build() from the snapshot, and runs
           the episodes — the same machinery as official scoring, at
           --trials (default 1, capped at the pinned trial count). Episode
           transcripts land on the lab-out volume; the printed path says
           where.

The task's `surface` (task/task_meta.yaml) says which layers your edits can
move; anything outside it is pinned server-side and editing it changes
nothing about your score.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the workspace root


def read_meta() -> dict:
    p = ROOT / "task" / "task_meta.yaml"
    if not p.is_file():
        raise SystemExit("no task/task_meta.yaml — is this a seeded workspace?")
    meta = {}
    for line in p.read_text().splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"')
    return meta


def qa_dev_eval(args, meta: dict) -> None:
    answers = args.answers
    if answers:
        print("[dev_eval] EXPERIMENT MODE: judging an existing answers file — "
              "this number is NOT a submission claim (nothing guarantees the "
              "wired-in submission produces these answers).", file=sys.stderr)
    else:
        answers = str(ROOT / "runs" / "dev_eval_answers.json")
        cmd = [sys.executable, str(ROOT / "submission" / "eval.py"),
               "--input", str(ROOT / "task" / "dev.json"), "--output", answers]
        print("[dev_eval] producing answers through the submission:",
              " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
    out = args.out or str(ROOT / "runs" / "dev_eval_results.json")
    cmd = [sys.executable, str(Path(__file__).parent / "rubric_eval.py"),
           "--dev", str(ROOT / "task" / "dev.json"), "--answers", answers,
           "--task", meta["task"], "--out", out]
    subprocess.run(cmd, check=True)
    print(f"[dev_eval] results: {out}")


def snapshot_bytes() -> bytes:
    """submission/ + toolbox/ harness code, NOT the cloned training packages
    (they are pinned upstream repos — the service does not need them)."""
    buf = io.BytesIO()
    skip_dirs = {"__pycache__", ".git"}
    heavy = {str(ROOT / "toolbox" / "training_tool"),
             str(ROOT / "toolbox" / "data_tool")}

    def keep(ti: tarfile.TarInfo) -> "tarfile.TarInfo | None":
        parts = set(Path(ti.name).parts)
        if parts & skip_dirs:
            return None
        return ti

    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(ROOT / "submission", arcname="submission", filter=keep)
        for child in sorted((ROOT / "toolbox").iterdir()):
            if str(child) in heavy or not child.is_dir():
                continue
            tf.add(child, arcname=f"toolbox/{child.name}", filter=keep)
    return buf.getvalue()


def agentic_dev_eval(args, meta: dict) -> None:
    import modal
    model = args.model
    if not model:
        raise SystemExit("--model <checkpoint path or HF id> is required for "
                         "agentic dev-eval (the service serves it; /out paths "
                         "from your training runs work)")
    snap = snapshot_bytes()
    session = os.environ.get("LEARNING_AGENT_SESSION", "dev-eval")
    print(f"[dev_eval] calling dev_rollout: task={meta['task']} model={model} "
          f"trials={args.trials} snapshot={len(snap) // 1024}KB session={session}",
          file=sys.stderr)
    fn = modal.Function.from_name("lab-rollout", "dev_rollout")
    summary = fn.remote(task=meta["task"], model=model, snapshot=snap,
                        trials=args.trials, session=session,
                        base_url=args.base_url,
                        track=os.environ.get("LEARNING_AGENT_TRACK", ""))
    out = args.out or str(ROOT / "runs" / "dev_eval_results.json")
    Path(out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in
                      ("task", "split", "mean", "n", "trials", "episodes_path")},
                     indent=1))
    print(f"[dev_eval] results: {out}\n[dev_eval] episode transcripts: "
          f"modal volume get lab-out {summary['episodes_path'].removeprefix('/out/')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="",
                    help="checkpoint to measure (agentic: required; the service "
                         "serves it)")
    ap.add_argument("--trials", type=int, default=1,
                    help="agentic: trials per scenario (capped at the pinned count)")
    ap.add_argument("--base-url", default="",
                    help="agentic: drive an ALREADY-served endpoint instead of "
                         "having the service serve --model")
    ap.add_argument("--answers", default="",
                    help="qa: judge an existing answers JSON (EXPERIMENT MODE)")
    ap.add_argument("--out", default="", help="where to write the results JSON")
    args = ap.parse_args()
    meta = read_meta()
    if meta.get("archetype") == "agentic":
        agentic_dev_eval(args, meta)
    else:
        qa_dev_eval(args, meta)


if __name__ == "__main__":
    main()
