"""rubric_eval — judge candidates against weighted-claim rubrics.

For each eval row ({id, question, gold_answer, rubric}) this sources a candidate answer —
from a precomputed {id: answer} file, or by generating from a served task model (oai_client:
--model / $TASK_MODEL over an OpenAI-compatible endpoint, or the claude-CLI fallback) —
judges it with judge_client (N-vote majority per claim -> weighted score), and aggregates a
mean + SEEDED bootstrap 95% CI (bit-identical for a fixed seed). The top-level "mean" is the
fitness the evolve runner reads.

Eval rows come from gen_eval.py OR task/dev.json, interchangeably: a JSON array or
JSONL of {id, question, gold_answer, rubric[, topic, evidence]}.

    python3 toolbox/eval_tool/rubric_eval.py --dev dev.jsonl --answers candidates.json --out dev_results.json
    python3 toolbox/eval_tool/rubric_eval.py --dev task/dev.json --model "$TASK_MODEL" \
        --base-url http://localhost:8000/v1 --task fav2 --out dev_results.json

No pip package is needed: generation and judging both go over urllib/subprocess via siblings.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Toolbox bootstrap (standard, identical in every CLI script): put the toolbox
# root on sys.path so absolute package imports resolve when run as a script.
_TOOLBOX_ROOT = next((str(p) for p in Path(__file__).resolve().parents if p.name in ("toolbox", "toolbox_bank")), None)
if _TOOLBOX_ROOT is None:
    raise SystemExit(f"toolbox bootstrap: no toolbox ancestor above {Path(__file__).resolve()}")
if _TOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, _TOOLBOX_ROOT)
from api_clients import judge_client as J  # noqa: E402  (sibling module)
from api_clients.oai_client import add_client_args, client_from_args, resolve_model  # noqa: E402


# --------------------------------------------------------- bootstrap CI (seeded)

def bootstrap_ci95(values: list[float], resamples: int = 10000, seed: int = 0) -> list[float]:
    """Seeded percentile bootstrap 95% CI on the mean — reproduces bit-identically
    for a fixed seed."""
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [round(values[0], 4), round(values[0], 4)]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples)]
    return [round(lo, 4), round(hi, 4)]


# ------------------------------------------------------- eval + answer loading

def load_rows(path: str) -> list[dict]:
    """Load eval rows from a JSON array (dev.json) OR JSONL (gen_eval.py output)."""
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected a JSON array or JSONL of eval rows")
    return rows


def load_answers_file(path: str) -> dict[str, str]:
    """candidates.json-style {id: answer} (or {id: {answer|text|content}})."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    for qid, v in raw.items():
        if isinstance(v, str):
            out[qid] = v
        elif isinstance(v, dict):
            out[qid] = v.get("answer") or v.get("text") or v.get("content") or ""
        else:
            out[qid] = str(v)
    return out


def generate_answers(rows: list[dict], client, system: str | None,
                     temperature: float, max_tokens: int, max_workers: int) -> dict[str, str]:
    """Closed-book answers from the served task model via oai_client (concurrent)."""
    batch = []
    for r in rows:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": r["question"]})
        batch.append(msgs)
    replies = client.batch_chat(batch, max_workers=max_workers,
                                temperature=temperature, max_tokens=max_tokens)
    return {r["id"]: (t or "").strip() for r, t in zip(rows, replies)}


# ------------------------------------------------------------------- eval

def _judge_one(r: dict, ans: str, task, n_votes, model, backend) -> dict:
    """One question -> per_question entry. Failure is recorded, never a silent 0."""
    try:
        jr = J.judge_claims(r["question"], ans, r["rubric"],
                            gold_answer=r.get("gold_answer"), task=task,
                            n_votes=n_votes, model=model, backend=backend)
    except Exception as e:  # noqa: BLE001
        return {"claim_score": None, "failed": True, "error": str(e)[:300]}
    return {"claim_score": jr["claim_score"], "failed": False,
            "per_claim": jr["per_claim"], "canonical": jr["canonical"],
            "_backend": jr["backend"]}


def run_eval(rows: list[dict], answers: dict[str, str], task: str | None,
             n_votes: int, model: str, backend: str, seed: int,
             resamples: int, retry_failed: int = 1) -> dict:
    per_question: dict[str, dict] = {}
    used_backend = None
    for r in rows:
        qid = r["id"]
        ans = answers.get(qid, "")
        if not ans.strip():
            per_question[qid] = {"claim_score": None, "failed": True,
                                 "reason": "empty/missing answer"}
            print(f"  {qid[:24]:24} FAILED (empty answer)", flush=True)
            continue
        entry = _judge_one(r, ans, task, n_votes, model, backend)
        used_backend = entry.pop("_backend", used_backend)
        per_question[qid] = entry
        if entry["failed"]:
            print(f"  {qid[:24]:24} FAILED (judge error: {entry['error'][:70]})", flush=True)
        else:
            print(f"  {qid[:24]:24} claim={entry['claim_score']:.4f}", flush=True)

    # Bounded re-judge of ERROR failures only (empty answers cannot improve).
    for attempt in range(retry_failed):
        retryable = [r for r in rows
                     if per_question[r["id"]].get("failed")
                     and "error" in per_question[r["id"]]]
        if not retryable:
            break
        print(f"[rubric_eval] retrying {len(retryable)} failed judgement(s) "
              f"(attempt {attempt + 1}/{retry_failed})", flush=True)
        for r in retryable:
            entry = _judge_one(r, answers[r["id"]], task, n_votes, model, backend)
            used_backend = entry.pop("_backend", used_backend)
            per_question[r["id"]] = entry
            if entry["failed"]:
                print(f"  {r['id'][:24]:24} FAILED again (judge error: {entry['error'][:70]})", flush=True)
            else:
                print(f"  {r['id'][:24]:24} claim={entry['claim_score']:.4f} (retry)", flush=True)

    scores = [e["claim_score"] for e in per_question.values() if not e.get("failed")]
    failed = [qid for qid, e in per_question.items() if e.get("failed")]
    n = len(scores)
    mean = round(sum(scores) / n, 4) if n else None
    ci = bootstrap_ci95(scores, resamples=resamples, seed=seed) if n else None
    return {
        "kind": "rubric_eval",
        "task": task,
        "judge_model": model,
        "judge_backend": used_backend or backend,
        "n_votes": n_votes,
        "mean": mean,          # <- fitness the evolve runner reads (gated on n_failed there)
        "n": n,
        "n_failed": len(failed),
        "failed": failed,
        "bootstrap_ci95": ci,
        "bootstrap_seed": seed,
        "per_question": per_question,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the LLM judge over eval data: score candidates, report mean + CI.")
    ap.add_argument("--dev", required=True, help="eval rows: gen_eval JSONL or dev.json array")
    ap.add_argument("--answers", default=None,
                    help="precomputed {id: answer} (candidates.json); else generate from --model")
    # served-task model generation (used only when --answers is absent)
    add_client_args(ap, default_temperature=0.0, default_max_tokens=2048)
    ap.add_argument("--sys-file", default=None, help="optional system prompt for generation")
    ap.add_argument("--max-workers", type=int, default=8, help="concurrency for generation")
    # judge
    ap.add_argument("--task", default=None, help="fav2 -> finance judge persona; else code")
    ap.add_argument("--n-votes", type=int, default=3,
                    help="self-consistency: judge each question this many times and majority-vote "
                         "(odd count breaks ties); mirrors bench/config.yaml judge.n_votes")
    ap.add_argument("--judge-model", default=None,
                    help="judge model id (or set $JUDGE_MODEL); none baked in")
    ap.add_argument("--judge-backend", default="auto",
                    choices=["auto", "openai", "api", "cli", "mock"],
                    help="mock = deterministic offline judge stub (no key/network, for tests)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (fixed -> reproducible)")
    ap.add_argument("--resamples", type=int, default=10000,
                    help="bootstrap resamples for the 95%% CI on the mean; mirrors "
                         "bench/config.yaml judge.bootstrap_resamples")
    ap.add_argument("--retry-failed", type=int, default=1,
                    help="re-judge questions that failed with a judge error, up to N extra passes; "
                         "1 recovers transient errors (429/5xx/parse) at bounded cost — a question "
                         "that still fails is marked failed, never silently scored 0")
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()

    rows = load_rows(args.dev)
    if args.answers:
        answers = load_answers_file(args.answers)
    else:
        resolve_model(args, ap)  # errors cleanly if neither --model nor $TASK_MODEL is set
        client = client_from_args(args)
        system = Path(args.sys_file).read_text() if args.sys_file else None
        print(f"[gen] generating {len(rows)} answers "
              f"(backend={client.backend} model={client.model})", flush=True)
        answers = generate_answers(rows, client, system, args.temperature,
                                   args.max_tokens, args.max_workers)

    judge_model = J.resolve_judge_model(args.judge_model, args.judge_backend)
    print(f"[eval] dev={args.dev} n={len(rows)} judge={judge_model} "
          f"backend={args.judge_backend} n_votes={args.n_votes}", flush=True)
    result = run_eval(rows, answers, args.task, args.n_votes, judge_model,
                      args.judge_backend, args.seed, args.resamples, args.retry_failed)

    mean_str = f"{result['mean']:.4f}" if result["mean"] is not None else "n/a (all failed)"
    print(f"\n[eval] mean={mean_str}  ci95={result['bootstrap_ci95']}  "
          f"n={result['n']}  failed={result['n_failed']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
