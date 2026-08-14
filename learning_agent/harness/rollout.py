#!/usr/bin/env python3
"""Learning Agent agentic rollout — eval+score for `archetype: agentic` tasks.

QA path:      harness/eval.py (ReAct search -> candidates) -> judge_cli.py (LLM judge)
agentic path: THIS FILE — load scenario rows from tasks/<T>/<split>.json, build
              the submission policy (submission/agent.py build()), run each row
              through the task's env adapter (harness/adapters/<env>.py, see its
              __init__ for the contract), score with the env's OWN verifier.
              No LLM judge anywhere.

Artifacts deliberately MIRROR the judge's — same file names, same JSON shape —
so bench.py's score printer, the leaderboard, and the observatory work
unchanged:

    runs/<tag>/budget_<env.max_steps>/results_<split>.json     (budget = the
        agentic operating point: the pinned per-episode step cap)
    runs/<tag>/budget_<...>/episodes_<split>/<qid>_t<k>.json   (the
        verdicts_<split>/ analog: full per-episode transcripts for audit)
    runs/LEADERBOARD.jsonl append (refused when 0 rows scored)
    --limit N -> smoke_ prefixed artifacts + implied --no-record (a truncated
        run must never overwrite a full run or become a leaderboard row)

Scoring: env.num_trials episodes per row (trial seeds = env.seed + t);
per-row score = mean reward across trials; secondary metric = pass^k
(all-trials reward == 1.0) when num_trials > 1. A crashed episode marks the
row failed — never a silent 0. mean/CI are null when nothing scored.

Config-driven (harness/config.py): every knob comes from tasks/<T>/task.yaml,
overridable via --config <yaml>. CLI carries only run identity (task/split/
tag/model/endpoint) and safety flags.

Local run (mock/self-contained envs, or a student already served somewhere):
    python harness/rollout.py --task mock --split dev --backend mock \
        --limit 2 --allow-dirty
    python harness/rollout.py --task <T> --split dev --base-url <url> --model <m>
Envs with real dependencies (alfworld/webshop/tau2) run inside a Modal
container whose image bakes the pinned env — wired per pack; see bench.py
rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as labcfg           # noqa: E402
import envfile                    # noqa: E402
import integrity as I             # noqa: E402
from adapters import load_adapter  # noqa: E402
from judge_cli import _verify_verifier, bootstrap_ci95  # noqa: E402

envfile.load_env(ROOT)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_names(split: str, limit: int) -> tuple[str, str]:
    """(results filename, episodes dirname). smoke_ PREFIX for --limit runs —
    the observatory globs results_*.json, so a suffix would still surface an
    n=2 smoke as a scored row (same reasoning as judge_cli)."""
    if limit:
        return (f"smoke_results_{split}_limit{limit}.json",
                f"smoke_episodes_{split}_limit{limit}")
    return f"results_{split}.json", f"episodes_{split}"


def _validated(out: dict, qid: str) -> dict:
    """Enforce the adapter contract on one episode result."""
    if not isinstance(out, dict):
        raise ValueError(f"adapter returned {type(out).__name__}, not dict")
    r = out.get("reward")
    if not isinstance(r, (int, float)) or not (0.0 <= float(r) <= 1.0):
        raise ValueError(f"adapter reward {r!r} for {qid} not a float in [0,1]")
    return {"reward": round(float(r), 4), "steps": int(out.get("steps", 0)),
            "done": bool(out.get("done", False)),
            "transcript": out.get("transcript") or [],
            "info": out.get("info") or {}}


def _scored_row(rewards: list, steps: list, dones: list, k: int) -> dict:
    """Trials -> one judge-shaped per_question entry (mean reward + pass^k)."""
    score = round(sum(rewards) / len(rewards), 4)
    if k > 1:
        sec = {"kind": f"pass^{k}",
               "score": 1.0 if all(r >= 1.0 for r in rewards) else 0.0,
               "detail": {"k": k, "rewards": rewards}}
    else:
        sec = {"kind": "none", "score": None, "detail": {}}
    return {"claim_score": score, "rewards": rewards, "steps": steps,
            "done": dones, "secondary": sec, "failed": False}


def _failed_row(rewards, steps, dones, err: str) -> dict:
    return {"claim_score": None, "rewards": rewards, "steps": steps,
            "done": dones, "secondary": {"kind": "none", "score": None, "detail": {}},
            "failed": True, "error": err}


def rollout_rows(rows, adapter, agent, tcfg: dict,
                 log=lambda s: print(s, flush=True),
                 save_episode=lambda qid, trial, ep: None) -> dict:
    """The pure core: every row x trial -> per_question (judge-shaped).

    per_question[qid] = {"claim_score": mean reward | None, "rewards": [...],
                         "steps": [...], "done": [...],
                         "secondary": {"kind","score","detail"}, "failed": bool
                         [, "error"]}
    A trial that raises (or violates the adapter contract) fails the whole row:
    a partial mean would silently overstate confidence.

    Adapters with a NATIVE batch runner (tau2's run_domain handles trials,
    concurrency, and user simulation itself) may expose run_split(agent, rows,
    cfg) -> {qid: {"rewards": [...], "steps": [...], "done": [...][, "error"]}}
    instead of being driven per episode; scoring/failure doctrine stays here
    either way.
    """
    env = tcfg.get("env") or {}
    k = int(env.get("num_trials", 1))
    base_seed = int(env.get("seed", 0))
    per_question: dict[str, dict] = {}

    if callable(getattr(adapter, "run_split", None)):
        raw = adapter.run_split(agent, rows, tcfg)
        for row in rows:
            qid = str(row["id"])
            entry = raw.get(qid)
            if entry is None or entry.get("error"):
                err = (entry or {}).get("error", "adapter returned no result for row")
                per_question[qid] = _failed_row([], [], [], str(err)[:300])
                log(f"  {qid[:22]:22} FAILED ({str(err)[:80]})")
                continue
            rewards = entry.get("rewards") or []
            bad = [r for r in rewards
                   if not isinstance(r, (int, float)) or not 0.0 <= float(r) <= 1.0]
            if not rewards or bad:
                per_question[qid] = _failed_row([], [], [], f"bad rewards {rewards!r}")
                log(f"  {qid[:22]:22} FAILED (bad rewards)")
                continue
            rewards = [round(float(r), 4) for r in rewards]
            pq = _scored_row(rewards, entry.get("steps") or [],
                             entry.get("done") or [], k)
            per_question[qid] = pq
            for t, ep in enumerate(entry.get("episodes") or []):
                save_episode(qid, t, ep)
            sstr = "-" if pq["secondary"]["score"] is None else f"{pq['secondary']['score']:.0f}"
            log(f"  {qid[:22]:22} reward={pq['claim_score']:.4f}  pass^k={sstr}")
        return per_question

    for row in rows:
        qid = str(row["id"])
        rewards, steps, dones = [], [], []
        err = None
        for t in range(k):
            cfg = dict(tcfg)
            cfg["trial"], cfg["seed"] = t, base_seed + t
            try:
                ep = _validated(adapter.run_episode(agent, row, cfg), qid)
            except Exception as e:  # noqa: BLE001  crashed episode -> failed row
                err = f"trial {t}: {str(e)[:300]}"
                break
            save_episode(qid, t, {**ep, "trial": t, "seed": cfg["seed"]})
            rewards.append(ep["reward"])
            steps.append(ep["steps"])
            dones.append(ep["done"])
        if err is not None:
            per_question[qid] = _failed_row(rewards, steps, dones, err)
            log(f"  {qid[:22]:22} FAILED ({err[:80]})")
            continue
        pq = _scored_row(rewards, steps, dones, k)
        per_question[qid] = pq
        sstr = "-" if pq["secondary"]["score"] is None else f"{pq['secondary']['score']:.0f}"
        log(f"  {qid[:22]:22} reward={pq['claim_score']:.4f}  pass^k={sstr}  steps={steps}")
    return per_question


def build_provenance(root: Path, task: str, tcfg: dict, split: str,
                     integrity_status: str, limit: int, policy: dict) -> dict:
    """Judge-provenance key parity, filled for an env-verified run, plus the
    agentic pins (adapter/env/driver). judge_model 'env' = the env IS the judge."""
    env = tcfg.get("env") or {}
    acfg = tcfg.get("agent") or {}
    adapter_rel = tcfg["adapter"]
    return {
        "judge_model": "env", "pinned_judge_model": "env", "judge_backend": "env",
        "canonical": True,                      # the env verifier IS the canonical judge
        "n_votes": int(env.get("num_trials", 1)),   # trials play the vote role
        "judge_prompt_sha": "",                 # no judge prompt exists on this path
        "harness_sha": _sha256_file(root / "harness" / "rollout.py"),
        "sys_sha": _sha256_file(root / tcfg["sys"]),
        "gold_sha": _sha256_file(root / tcfg[split]),
        "config_sha": labcfg.config_sha(root, task),
        "corpus_pin": tcfg.get("corpus_version") or tcfg.get("corpus_commit") or "",
        "budget": int(env.get("max_steps", 0)), "limit": limit,
        "eval_temperature": float(acfg.get("temperature", 0.0)),
        "seed": int(env.get("seed", 0)),
        "integrity": integrity_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # agentic-specific pins
        "adapter": adapter_rel,
        "adapter_sha": _sha256_file(root / adapter_rel),
        "env_pin": env.get("pin", ""),
        "num_trials": int(env.get("num_trials", 1)),
        "max_steps": int(env.get("max_steps", 0)),
        "driver": acfg.get("driver", "react"),
        "user_llm": env.get("user_llm", ""),
        "policy_model": policy.get("model", ""),
        "policy_backend": policy.get("backend", ""),
    }


def build_results(task: str, tag: str, split: str, tcfg: dict,
                  per_question: dict, provenance: dict, limit: int = 0,
                  bootstrap_resamples: int = 10000, seed: int = 0) -> dict:
    """results_<split>.json with exact top-level key parity to judge_cli."""
    env = tcfg.get("env") or {}
    k = int(env.get("num_trials", 1))
    scores = [pq["claim_score"] for pq in per_question.values()
              if pq["claim_score"] is not None]
    failed = [qid for qid, pq in per_question.items() if pq["failed"]]
    n = len(scores)
    mean = round(sum(scores) / n, 4) if n else None
    ci = bootstrap_ci95(scores, resamples=bootstrap_resamples, seed=seed) if n else None
    sec_vals = [pq["secondary"]["score"] for pq in per_question.values()
                if pq["secondary"]["score"] is not None]
    sec_mean = round(sum(sec_vals) / len(sec_vals), 4) if sec_vals else None
    results = {
        "task": task, "tag": tag, "split": split,
        "budget": int(env.get("max_steps", 0)),
        "judge_model": "env", "backend": "env", "canonical": True,
        "grade_mode": "env_reward",
        "secondary_metric": f"pass^{k}" if k > 1 else "none",
        "mean": mean, "n": n, "bootstrap_ci95": ci,
        "secondary_mean": sec_mean,
        "failed": failed, "n_failed": len(failed),
        "all_failed": len(failed) == len(per_question) and len(per_question) > 0,
        "provenance": provenance,
        "per_question": per_question,
    }
    if limit:
        results["smoke"] = True
        results["limit"] = limit
    return results


def leaderboard_row(results: dict) -> dict:
    """Same keys as judge_cli's append."""
    return {"task": results["task"], "tag": results["tag"], "split": results["split"],
            "score": results["mean"], "ci": results["bootstrap_ci95"],
            "n": results["n"], "failed": results["all_failed"],
            "n_failed": results["n_failed"], "secondary_mean": results["secondary_mean"],
            "judge_model": "env", "backend": "env", "canonical": True,
            "integrity": results["provenance"]["integrity"],
            "provenance": results["provenance"]}


def persist(task: str, split: str, tag: str, tcfg: dict, g: dict,
            per_question: dict, episodes: list, integrity_status: str,
            limit: int, no_record: bool, policy: dict, root: Path | None = None) -> dict:
    """Write episode transcripts + results, append (or refuse) the leaderboard
    row, print the summary. Shared by the local CLI below and the Modal
    entrypoint (harness/rollout_modal.py) so artifacts are identical either way.
    `episodes` = [(qid, trial, episode_dict), ...]."""
    root = root or ROOT
    env = tcfg.get("env") or {}
    bdir = root / g["runs_dir"] / tag / f"budget_{int(env.get('max_steps', 0))}"
    results_name, episodes_name = artifact_names(split, limit)
    edir = bdir / episodes_name
    edir.mkdir(parents=True, exist_ok=True)
    for qid, trial, ep in episodes:
        (edir / f"{qid}_t{trial}.json").write_text(json.dumps(ep, indent=2))

    provenance = build_provenance(root, task, tcfg, split, integrity_status,
                                  limit, policy)
    jcfg = g["judge"]
    results = build_results(task, tag, split, tcfg, per_question, provenance,
                            limit=limit,
                            bootstrap_resamples=int(jcfg.get("bootstrap_resamples", 10000)),
                            seed=int(jcfg.get("bootstrap_seed", 0)))
    out_path = bdir / results_name
    out_path.write_text(json.dumps(results, indent=2))
    if limit:
        print(f"[rollout] --limit {limit}: smoke artifacts only "
              f"({out_path.name}, {edir.name}/); full-run artifacts left untouched")

    if no_record:
        print("[rollout] --no-record: leaderboard append SKIPPED")
    elif results["n"] == 0:
        print(f"[rollout] REFUSING leaderboard append: 0 rows scored "
              f"({results['n_failed']}/{len(per_question)} failed) — an all-failed "
              f"run has no score to record (see {out_path})")
    else:
        lb = root / g["leaderboard"]
        with lb.open("a") as f:
            f.write(json.dumps(leaderboard_row(results)) + "\n")
        print(f"[rollout] appended leaderboard row -> {lb}")

    mean = results["mean"]
    score_str = f"{mean:.4f}" if mean is not None else "n/a (all failed)"
    print(f"\n[rollout] {tag}/{split}: score={score_str}  "
          f"ci95={results['bootstrap_ci95']}  n={results['n']}  "
          f"failed={results['n_failed']}  pass^k={results['secondary_mean']}  "
          f"integrity={integrity_status}")
    print(f"[rollout] wrote {out_path}")
    return results


def integrity_gate(allow_dirty: bool, g: dict, label: str = "rollout") -> str:
    """judge_cli's doctrine, shared with rollout_modal: refuse on drift unless
    --allow-dirty, which stamps DIRTY. Returns the integrity status."""
    pins_file = ROOT / g.get("pins", "bench/pins.json")
    mismatches = _verify_verifier(ROOT, pins_file) + I.verify_pins(ROOT)
    if mismatches and not allow_dirty:
        msg = "\n".join(f"  - {m}" for m in mismatches)
        raise SystemExit(
            f"[integrity] REFUSING to {label}: benchmark surface does not match "
            f"bench/pins.json:\n{msg}\n"
            "If the change is deliberate, re-freeze (`python bench.py freeze`).\n"
            "To score anyway (stamped integrity:DIRTY), pass --allow-dirty.")
    if mismatches:
        print(f"[integrity] WARNING: scoring with {len(mismatches)} pin "
              "mismatch(es); results stamped DIRTY")
        return "DIRTY"
    return "OK"


def _build_agent(args):
    """The submission policy of the tree we run in — post-eval scores exactly
    what the contestant shipped because this import resolves in THEIR tree."""
    sub = str(ROOT / "submission")
    if sub not in sys.path:
        sys.path.insert(0, sub)
    from agent import build  # noqa: PLC0415
    if args.backend in ("mock", "cli-claude"):
        return (build(backend=args.backend, model=args.model),
                {"model": args.model or args.backend, "backend": args.backend})
    agent = build(weights=args.model if not args.base_url else "",
                  base_url=args.base_url, model=args.model)
    return agent, {"model": args.model or "student",
                   "backend": args.base_url or "serve.py"}


def main():
    cfg = labcfg.load_config(ROOT)
    g = cfg["global"]
    ap = argparse.ArgumentParser(description="Learning Agent agentic rollout (env-verified scoring).")
    ap.add_argument("--task", required=True, choices=labcfg.known_tasks(ROOT))
    ap.add_argument("--split", required=True, choices=g["splits"])
    ap.add_argument("--config", default="", help="run-override YAML (task.yaml schema)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="", help="weights to serve, or model name at --base-url")
    ap.add_argument("--base-url", default="", help="already-served OpenAI-compatible endpoint")
    ap.add_argument("--backend", default="", choices=["", "mock", "cli-claude"],
                    help="mock = offline contract smoke; cli-claude = frontier "
                         "reference baseline via the logged-in claude CLI")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N rows (smoke): implies --no-record, smoke_* artifacts")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.limit and not args.no_record:
        print(f"[rollout] --limit {args.limit} is a smoke test: truncated runs are "
              "never recorded (implying --no-record)")
        args.no_record = True

    # ---- INTEGRITY GATE (identical doctrine to judge_cli) ----
    integrity_status = integrity_gate(args.allow_dirty, g)

    tcfg = labcfg.resolve(ROOT, args.task, args.config or None)
    if tcfg.get("archetype") != "agentic":
        raise SystemExit(f"[rollout] task {args.task!r} is archetype "
                         f"{tcfg.get('archetype')!r} — QA tasks score via "
                         "bench.py score (eval + judge), not rollout")
    problems = labcfg.validate_task(tcfg)
    if problems:
        raise SystemExit(f"[rollout] invalid task config: {'; '.join(problems)}")

    adapter = load_adapter(ROOT, tcfg["adapter"])
    rows = adapter.load_split(ROOT / tcfg[args.split])
    if args.limit:
        rows = rows[: args.limit]

    env = tcfg.get("env") or {}
    tag = args.tag or f"{args.task}_{(args.model or args.backend or 'student').rstrip('/').split('/')[-1]}_{args.split}"

    agent, policy = _build_agent(args)
    print(f"[rollout] task={args.task} tag={tag} split={args.split} "
          f"adapter={tcfg['adapter']} driver={(tcfg.get('agent') or {}).get('driver', 'react')} "
          f"num_trials={env.get('num_trials', 1)} max_steps={env.get('max_steps')} "
          f"n={len(rows)} integrity={integrity_status}")

    episodes: list[tuple] = []
    per_question = rollout_rows(
        rows, adapter, agent, tcfg,
        save_episode=lambda qid, t, ep: episodes.append((qid, t, ep)))
    return persist(args.task, args.split, tag, tcfg, g, per_question, episodes,
                   integrity_status, args.limit, args.no_record, policy)


if __name__ == "__main__":
    main()
