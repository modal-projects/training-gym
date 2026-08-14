#!/usr/bin/env python3
"""Learning Agent — THE one entrypoint.

Learning Agent measures an LLM's ability to run the applied-ML loop — build data -> train ->
eval -> iterate — to make a student model expert on a domain corpus, with NO
verifiable reward (answers are graded by a weighted-claim LLM judge, not unit tests).

Everything is pinned in bench/config.yaml. Subcommands:

  python bench.py score  --task <T> --model <M> --split dev|test [--tag <tag>]
        end-to-end scoring, dispatched on the task's archetype (task.yaml):
        qa      -> ReAct-search eval, then the canonical judge, then record
        agentic -> harness/rollout.py: env episodes scored by the env's OWN
                   verifier (no LLM judge), same artifacts + leaderboard row
  python bench.py eval   --task <T> --model <M> --split dev|test [--tag --budgets]
        just produce candidates (runs/<tag>/budget_*/candidates.json). QA only.
  python bench.py judge  --task <T> --tag <tag> --split dev|test [--budget]
        just judge existing candidates -> results_<split>.json + row. QA only.
  python bench.py rollout --task <T> --split dev|test [--model|--base-url|--backend mock]
        agentic episodes only (the eval+score of the agentic path).
  python bench.py leaderboard [--task <T>] [--split <S>]
        pretty-print runs/LEADERBOARD.jsonl (incl. integrity + backend columns).
  python bench.py freeze
        write bench/pins.json — sha256 pins of the fixed benchmark surface.
  python bench.py verify
        report pin status. score/judge REFUSE on mismatch (--allow-dirty overrides,
        stamping integrity:"DIRTY" into provenance and the leaderboard).

Pins live in bench/config.yaml (global) + task_configs/<T>.yaml (per task — a task
exists iff that file exists); harness/config.py is the one loader. bench/pins.json
locks the surface those pins describe.

Runs are config-driven: score/eval/judge accept --config <override.yaml> (same
schema as task.yaml, any subset of keys, plus run:/judge: sections). Precedence:
task.yaml defaults -> --config override -> CLI flags. The resolved config is
snapshotted to runs/<tag>/run_config.yaml — one file reproduces the run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "harness"))
import envfile  # noqa: E402  (.env -> ANTHROPIC_API_KEY etc., non-overriding)

envfile.load_env(ROOT)

try:
    import config as labcfg  # noqa: E402  (harness/config.py — the one config loader)
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyyaml required: pip install pyyaml") from e


def load_config() -> dict:
    """Combined view {"global": …, "tasks": {…}} from bench/config.yaml +
    every task_configs/<T>.yaml (see harness/config.py)."""
    return labcfg.load_config(ROOT)


def _sh(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def _default_tag(task: str, split: str, model: str) -> str:
    base = model.rstrip("/").split("/")[-1] or "model"
    return f"{task}_{base}_{split}"


def _archetype(cfg, task: str) -> str:
    return cfg["tasks"][task].get("archetype", "qa")


def _integrity():
    """Import harness/integrity.py (kept out of top-level so `--help` never fails)."""
    sys.path.insert(0, str(ROOT / "harness"))
    try:
        import integrity
        return integrity
    finally:
        sys.path.pop(0)


def _verify_verifier(cfg) -> list[str]:
    """Verify the integrity VERIFIER itself: hash harness/integrity.py directly
    against bench/pins.json WITHOUT calling integrity.verify_pins — a neutered
    verifier must never get to vouch for itself. Returns [] when clean."""
    rel = "harness/integrity.py"
    pins_file = ROOT / cfg["global"].get("pins", "bench/pins.json")
    if not pins_file.exists():
        return []  # verify_pins reports the missing pins file itself
    try:
        pinned = json.loads(pins_file.read_text()).get("files", {})
    except (OSError, json.JSONDecodeError) as e:
        return [f"unreadable pins file {pins_file}: {e}"]
    want = pinned.get(rel)
    path = ROOT / rel
    if want is None:
        return [f"{rel}: not pinned in {pins_file.name} — re-freeze deliberately"]
    if not path.exists():
        return [f"{rel}: pinned but missing on disk"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        return [f"{rel}: sha256 mismatch — the integrity verifier itself was modified"]
    return []


def _verify_gate(allow_dirty: bool, cfg) -> bool:
    """Run the integrity gate before scoring. Returns True if scoring may proceed.

    judge_cli.py re-verifies independently (defense in depth); this early gate
    saves a Modal GPU eval when the surface has drifted. The verifier itself is
    checked first, by direct hash, so a tampered integrity.py cannot wave the
    rest of the check through.
    """
    mismatches = _verify_verifier(cfg) + _integrity().verify_pins(ROOT)
    if not mismatches:
        return True
    print("[integrity] benchmark surface does NOT match bench/pins.json:")
    for m in mismatches:
        print(f"  - {m}")
    if allow_dirty:
        print("[integrity] --allow-dirty: proceeding; results will be stamped DIRTY")
        return True
    print("[integrity] REFUSING. Re-freeze deliberately (`python bench.py freeze`) "
          "or pass --allow-dirty.")
    return False


def _apply_run_config(args, cfg, parser):
    """Config-driven runs: task.yaml defaults <- --config override <- CLI flags.

    Called once in main() for subcommands that carry --config (score/eval/judge).
    Fills every arg the user left unset from the resolved config's run:/judge:
    sections (falling back to the global judge pins), validates what ended up
    required, and snapshots the fully-resolved config to
    runs/<tag>/run_config.yaml — the one file that reproduces the run.
    """
    if not hasattr(args, "config"):
        return None  # train/rl/leaderboard/freeze/verify: not config-driven (yet)
    g = cfg["global"]
    rcfg = labcfg.resolve(ROOT, args.task, args.config or None)
    run = dict(rcfg.get("run") or {})
    jover = dict(rcfg.get("judge") or {})   # task/override-level judge settings

    def fill(field, value):
        if hasattr(args, field) and getattr(args, field) in (None, ""):
            setattr(args, field, value)

    fill("model", run.get("model", ""))
    fill("split", run.get("split", ""))
    fill("tag", run.get("tag", ""))
    fill("budgets", run.get("budgets", ""))
    fill("adapter", run.get("adapter", ""))
    if args.cmd in ("score", "judge"):
        # judge knobs only where a judge runs — `rollout` has its own --backend
        # (the POLICY backend) that must never inherit the judge's
        fill("judge_model", jover.get("model") or g["judge"]["model"])
        fill("backend", jover.get("backend") or g["judge"]["backend"])
        fill("n_votes", jover.get("n_votes"))
    # what each command truly requires, after config fill
    if hasattr(args, "model") and args.cmd in ("score", "eval") and not args.model:
        parser.error(f"{args.cmd}: --model required (CLI or run: model in --config)")
    if hasattr(args, "split") and not args.split:
        parser.error(f"{args.cmd}: --split required (CLI or run: split in --config)")
    if getattr(args, "split", "") and args.split not in g["splits"]:
        parser.error(f"invalid split {args.split!r} (choose from {g['splits']})")
    # snapshot the resolved run config beside the run's artifacts
    tag = args.tag or (_default_tag(args.task, args.split, args.model)
                       if getattr(args, "model", "") else "")
    if tag:
        resolved = dict(rcfg)
        resolved["run"] = {**run, "model": getattr(args, "model", ""),
                           "split": args.split, "tag": tag,
                           "judge_model": getattr(args, "judge_model", None),
                           "judge_backend": getattr(args, "backend", None),
                           "config_sha": labcfg.config_sha(ROOT, args.task)}
        labcfg.snapshot(resolved, ROOT / g["runs_dir"] / tag / "run_config.yaml")
    return rcfg


# ---------- subcommands ----------

def cmd_eval(args, cfg):
    """Run harness/eval.py (Modal) to produce candidates. QA archetype only."""
    if _archetype(cfg, args.task) != "qa":
        print(f"[eval] task {args.task!r} is agentic — use `bench.py rollout` "
              "(or `score`, which dispatches).")
        return 2, args.tag
    g = cfg["global"]
    tag = args.tag or _default_tag(args.task, args.split, args.model)
    budgets = args.budgets or str(g["eval_budget"])
    cmd = ["modal", "run", "harness/eval.py::expertise",
           "--task", args.task, "--model", args.model,
           "--budgets", budgets, "--split", args.split, "--tag", tag]
    if args.adapter:
        cmd += ["--adapter", args.adapter]
    if args.note_file:
        cmd += ["--note-file", args.note_file]
    if args.tp != 1:
        cmd += ["--tp", str(args.tp)]
    rc = _sh(cmd)
    if rc == 0:
        print(f"[eval] candidates -> runs/{tag}/budget_*/candidates.json")
    return rc, tag


def cmd_judge(args, cfg):
    """Run the canonical judge over existing candidates (integrity-gated). QA only."""
    if _archetype(cfg, args.task) != "qa":
        print(f"[judge] task {args.task!r} is agentic — the env's verifier scores "
              "it (`bench.py rollout`); there is no judge step.")
        return 2
    if not _verify_gate(getattr(args, "allow_dirty", False), cfg):
        return 2
    g = cfg["global"]
    budget = args.budget if args.budget is not None else g["eval_budget"]
    cmd = [sys.executable, "harness/judge_cli.py",
           "--task", args.task, "--tag", args.tag,
           "--split", args.split, "--budget", str(budget)]
    if getattr(args, "judge_model", None):
        cmd += ["--judge-model", args.judge_model]
    if getattr(args, "backend", None):
        cmd += ["--backend", args.backend]
    if getattr(args, "n_votes", None):
        cmd += ["--n-votes", str(args.n_votes)]
    if getattr(args, "limit", 0):
        cmd += ["--limit", str(args.limit)]
    if getattr(args, "no_record", False):
        cmd += ["--no-record"]
    if getattr(args, "allow_dirty", False):
        cmd += ["--allow-dirty"]
    # judge_cli prints the score line and appends the leaderboard row itself.
    return _sh(cmd)


def cmd_rollout(args, cfg):
    """Agentic eval+score: harness/rollout.py (env-verified, integrity-gated)."""
    if _archetype(cfg, args.task) != "agentic":
        print(f"[rollout] task {args.task!r} is archetype qa — use `bench.py score` "
              "(eval + judge).")
        return 2
    if not _verify_gate(getattr(args, "allow_dirty", False), cfg):
        return 2
    if not (args.model or getattr(args, "base_url", "") or
            getattr(args, "backend", "") == "mock"):
        print("[rollout] need a policy: --model <weights>, --base-url <endpoint>, "
              "or --backend mock (offline contract smoke).")
        return 2
    rcfg = labcfg.resolve(ROOT, args.task, getattr(args, "config", "") or None)
    tag = args.tag or _default_tag(
        args.task, args.split,
        args.model or getattr(args, "backend", "") or "student")
    # mock / already-served endpoints run locally; real weights run in the
    # task's Modal container (env deps + sglang serving baked in the image)
    local = (getattr(args, "backend", "") == "mock"
             or bool(getattr(args, "base_url", "")))
    if local:
        cmd = [sys.executable, "harness/rollout.py",
               "--task", args.task, "--split", args.split, "--tag", tag]
        if args.model:
            cmd += ["--model", args.model]
        if getattr(args, "base_url", ""):
            cmd += ["--base-url", args.base_url]
        if getattr(args, "backend", ""):
            cmd += ["--backend", args.backend]
    else:
        # sys.executable -m modal: the same interpreter running bench.py, so a
        # venv with the modal package needs no `modal` binary on PATH
        cmd = [sys.executable, "-m", "modal", "run",
               "harness/rollout_modal.py::rollout",
               "--task", args.task, "--split", args.split, "--tag", tag,
               "--model", args.model]
        if getattr(args, "tp", 1) != 1:
            cmd += ["--tp", str(args.tp)]
    if getattr(args, "config", ""):
        cmd += ["--config", args.config]
    if getattr(args, "limit", 0):
        cmd += ["--limit", str(args.limit)]
    if getattr(args, "no_record", False):
        cmd += ["--no-record"]
    if getattr(args, "allow_dirty", False):
        cmd += ["--allow-dirty"]
    # rollout.py prints the score line and appends the leaderboard row itself.
    return _sh(cmd)


def cmd_score(args, cfg):
    """End to end, dispatched on archetype: qa = eval -> judge -> record;
    agentic = rollout (env-verified). Integrity-gated up front either way."""
    if _archetype(cfg, args.task) == "agentic":
        # no judge on this path -> no judge-key pre-flight. Re-shape the args:
        # score's --backend is the JUDGE backend and must not leak into the
        # rollout policy backend.
        r_args = argparse.Namespace(
            cmd="rollout", task=args.task, split=args.split, tag=args.tag,
            model=args.model, config=getattr(args, "config", ""),
            base_url="", backend="", limit=0, tp=getattr(args, "tp", 1),
            no_record=getattr(args, "no_record", False),
            allow_dirty=getattr(args, "allow_dirty", False))
        return cmd_rollout(r_args, cfg)
    if not _verify_gate(getattr(args, "allow_dirty", False), cfg):
        return 2
    g = cfg["global"]
    # PRE-FLIGHT the judge credential BEFORE spending a GPU eval: the judge runs
    # last, so discovering a missing key after ~15 min of Modal time is pure waste.
    # (Candidates are still reusable via `bench.py judge`, but fail fast anyway.)
    backend = getattr(args, "backend", None) or g["judge"]["backend"]
    _key_for = {"api": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    need_key = _key_for.get(backend)
    if need_key and not os.environ.get(need_key):
        print(f"[score] judge pre-flight FAILED: backend is {backend!r} but {need_key} "
              "is not set (checked env and .env).")
        print("        Either:  cp .env.example .env  and add your key (canonical), or")
        print("        re-run with  --backend cli-claude  (non-canonical, stamped "
              "canonical:false — acceptable for dev self-eval).")
        return 2
    tag = args.tag or _default_tag(args.task, args.split, args.model)
    # eval (budget pinned to the canonical eval_budget for scoring)
    eval_args = argparse.Namespace(
        task=args.task, model=args.model, split=args.split, tag=tag,
        budgets=str(g["eval_budget"]), adapter=args.adapter,
        note_file=args.note_file, tp=args.tp)
    rc, tag = cmd_eval(eval_args, cfg)
    if rc != 0:
        print("[score] eval failed; aborting.")
        return rc
    # judge
    judge_args = argparse.Namespace(
        task=args.task, tag=tag, split=args.split, budget=g["eval_budget"],
        judge_model=args.judge_model, backend=args.backend,
        n_votes=getattr(args, "n_votes", None), limit=0,
        no_record=getattr(args, "no_record", False),
        allow_dirty=getattr(args, "allow_dirty", False))
    return cmd_judge(judge_args, cfg)


def cmd_freeze(args, cfg):
    """Write bench/pins.json — sha256 pins of the fixed benchmark surface."""
    integrity = _integrity()
    out = integrity.write_pins(ROOT)
    pins = json.loads(out.read_text())
    print(f"[freeze] pinned {len(pins['files'])} files + judge prompt + "
          f"eval_budget={pins['eval_budget']} -> {out}")
    return 0


def cmd_verify(args, cfg):
    """Report pin status (0 = clean, 1 = mismatch/missing)."""
    mismatches = _verify_verifier(cfg) + _integrity().verify_pins(ROOT)
    if not mismatches:
        print("[verify] OK — benchmark surface matches bench/pins.json")
        return 0
    print(f"[verify] {len(mismatches)} mismatch(es):")
    for m in mismatches:
        print(f"  - {m}")
    return 1


def cmd_leaderboard(args, cfg):
    lb = ROOT / cfg["global"]["leaderboard"]
    if not lb.exists():
        print(f"no leaderboard yet at {lb}")
        return 0
    rows = [json.loads(l) for l in lb.read_text().splitlines() if l.strip()]
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if not rows:
        print("no matching leaderboard rows")
        return 0
    rows.sort(key=lambda r: (r.get("task", ""), r.get("split", ""), -r.get("score", 0)))
    print(f"{'task':9} {'split':5} {'score':>7} {'ci95':>17} {'n':>3} {'2nd':>6} "
          f"{'fail':>5} {'backend':>10} {'integ':>5}  tag")
    print("-" * 104)
    for r in rows:
        ci = r.get("ci", [0, 0])
        cistr = f"[{ci[0]:.3f},{ci[1]:.3f}]"
        sec = r.get("secondary_mean")
        secstr = f"{sec:.3f}" if sec is not None else "-"
        fail = "FAIL" if r.get("failed") else "ok"
        prov = r.get("provenance", {})
        backend = r.get("backend") or prov.get("judge_backend") or "-"
        integ = r.get("integrity") or prov.get("integrity") or "-"
        print(f"{r.get('task',''):9} {r.get('split',''):5} {r.get('score',0):7.4f} "
              f"{cistr:>17} {r.get('n',0):3d} {secstr:>6} {fail:>5} "
              f"{backend:>10} {integ:>5}  {r.get('tag','')}")
    return 0


# ---------- argparse ----------

def build_parser(cfg) -> argparse.ArgumentParser:
    g = cfg["global"]
    tasks = list(cfg["tasks"])
    splits = g["splits"]
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    backends = ["api", "openai", "cli-claude", "cli-codex", "claude", "codex"]

    sc = sub.add_parser("score", help="eval -> judge -> record (end to end)")
    sc.add_argument("--task", required=True, choices=tasks)
    sc.add_argument("--config", default="",
                    help="run-override YAML (task.yaml schema; run:/judge: sections)")
    sc.add_argument("--model", default="", help="HF id or /out/models/<tag>/merged "
                    "(or run: model in --config)")
    sc.add_argument("--split", default="", help=f"one of {splits} (or run: split)")
    sc.add_argument("--tag", default="")
    sc.add_argument("--adapter", default="")
    sc.add_argument("--note-file", dest="note_file", default="")
    sc.add_argument("--tp", type=int, default=1)
    sc.add_argument("--judge-model", dest="judge_model", default=None,
                    help=f"default {g['judge']['model']} (pinned)")
    sc.add_argument("--backend", default=None, choices=backends,
                    help=f"default {g['judge']['backend']}; api/openai = canonical, "
                         "cli-* = non-canonical fallbacks")
    sc.add_argument("--n-votes", dest="n_votes", type=int, default=None,
                    help=f"self-consistency votes (default {g['judge'].get('n_votes', 3)})")
    sc.add_argument("--no-record", dest="no_record", action="store_true",
                    help="skip the LEADERBOARD append (smoke tests)")
    sc.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                    help="score despite pin mismatches (stamped integrity:DIRTY)")
    sc.set_defaults(func=cmd_score)

    ev = sub.add_parser("eval", help="produce candidates only (Modal agentic search)")
    ev.add_argument("--task", required=True, choices=tasks)
    ev.add_argument("--config", default="",
                    help="run-override YAML (task.yaml schema; run: section)")
    ev.add_argument("--model", default="", help="HF id (or run: model in --config)")
    ev.add_argument("--split", default="", help=f"one of {splits} (or run: split)")
    ev.add_argument("--tag", default="")
    ev.add_argument("--budgets", default="", help=f"default {g['eval_budget']}")
    ev.add_argument("--adapter", default="")
    ev.add_argument("--note-file", dest="note_file", default="")
    ev.add_argument("--tp", type=int, default=1)
    ev.set_defaults(func=lambda a, c: cmd_eval(a, c)[0])

    jd = sub.add_parser("judge", help="judge existing candidates -> score")
    jd.add_argument("--task", required=True, choices=tasks)
    jd.add_argument("--config", default="",
                    help="run-override YAML (task.yaml schema; run:/judge: sections)")
    jd.add_argument("--tag", required=True)
    jd.add_argument("--split", default="", help=f"one of {splits} (or run: split)")
    jd.add_argument("--budget", type=int, default=None, help=f"default {g['eval_budget']}")
    jd.add_argument("--judge-model", dest="judge_model", default=None,
                    help=f"default {g['judge']['model']} (pinned)")
    jd.add_argument("--backend", default=None, choices=backends,
                    help=f"default {g['judge']['backend']}; api/openai = canonical, "
                         "cli-* = non-canonical fallbacks")
    jd.add_argument("--n-votes", dest="n_votes", type=int, default=None,
                    help=f"self-consistency votes (default {g['judge'].get('n_votes', 3)})")
    jd.add_argument("--limit", type=int, default=0)
    jd.add_argument("--no-record", dest="no_record", action="store_true",
                    help="skip the LEADERBOARD append (smoke tests)")
    jd.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                    help="score despite pin mismatches (stamped integrity:DIRTY)")
    jd.set_defaults(func=cmd_judge)

    ro = sub.add_parser("rollout", help="agentic episodes, env-verified (no judge)")
    ro.add_argument("--task", required=True, choices=tasks)
    ro.add_argument("--config", default="",
                    help="run-override YAML (task.yaml schema; env:/agent: sections)")
    ro.add_argument("--split", default="", help=f"one of {splits} (or run: split)")
    ro.add_argument("--tag", default="")
    ro.add_argument("--model", default="",
                    help="weights to serve, or model name at --base-url")
    ro.add_argument("--base-url", dest="base_url", default="",
                    help="already-served OpenAI-compatible endpoint")
    ro.add_argument("--backend", default="", choices=["", "mock"],
                    help="mock = offline contract smoke (deterministic stub policy)")
    ro.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel size for in-container serving")
    ro.add_argument("--limit", type=int, default=0,
                    help="first N rows (smoke): implies --no-record, smoke_* artifacts")
    ro.add_argument("--no-record", dest="no_record", action="store_true")
    ro.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                    help="score despite pin mismatches (stamped integrity:DIRTY)")
    ro.set_defaults(func=cmd_rollout)

    lb = sub.add_parser("leaderboard", help="pretty-print runs/LEADERBOARD.jsonl")
    lb.add_argument("--task", default="", choices=[""] + tasks)
    lb.add_argument("--split", default="", choices=[""] + splits)
    lb.set_defaults(func=cmd_leaderboard)

    fz = sub.add_parser("freeze", help="write bench/pins.json (integrity lock)")
    fz.set_defaults(func=cmd_freeze)

    vf = sub.add_parser("verify", help="report integrity-pin status")
    vf.set_defaults(func=cmd_verify)

    return p


def main():
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()
    _apply_run_config(args, cfg, parser)
    rc = args.func(args, cfg)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
