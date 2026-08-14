"""gen_eval — author dev-style eval items from a corpus.

Read real spans (grep or line-window over --corpus), then have the model author, grounded
in each span, one dev-style eval item: a closed-book question, a gold answer, and a
weighted-claim rubric. Records the source span as evidence. Emits JSONL rows:

    {"id","topic","question","gold_answer",
     "rubric":[{"claim_id","claim_type","weight","statement"}],   # weights sum to 100
     "evidence":[{"path","start_line","end_line"}]}

feed straight into rubric_eval.py. Model-agnostic (--model / $TASK_MODEL, OpenAI-compatible
endpoint or claude-CLI fallback); span sampling and the client are reused from the data
toolbox, so no pip package is needed and --help works with nothing installed.

  python3 toolbox/eval_tool/gen_eval.py --corpus task/corpus --glob '**/*.txt' \\
      --base-url http://localhost:8000/v1 --n 100 --out dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Toolbox bootstrap (standard, identical in every CLI script): put the toolbox
# root on sys.path so absolute package imports resolve when run as a script.
_TOOLBOX_ROOT = next((str(p) for p in Path(__file__).resolve().parents if p.name in ("toolbox", "toolbox_bank")), None)
if _TOOLBOX_ROOT is None:
    raise SystemExit(f"toolbox bootstrap: no toolbox ancestor above {Path(__file__).resolve()}")
if _TOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, _TOOLBOX_ROOT)
from api_clients.oai_client import (  # noqa: E402
    add_client_args, client_from_args, parse_json_object, resolve_model,
)
from data_tool.corpus_sampling import iter_spans, span_context  # noqa: E402  (shared span sampling)


GEN_SYS = ("You author rigorous evaluation items grounded STRICTLY in a provided source "
           "excerpt. Every question, gold answer, and rubric claim must be verifiable from "
           "the excerpt alone — never invent facts, APIs, numbers, or names absent from it.")

GEN_INSTRUCTIONS = """From the excerpt above, author ONE evaluation item as a single strict JSON object:
{
  "topic": "<short topic label>",
  "question": "<detailed closed-book question; include ids/names/dates so it is answerable without the excerpt>",
  "gold_answer": "<the complete, correct answer, fully supported by the excerpt>",
  "rubric": [
    {"claim_type": "fact", "weight": 40, "statement": "<one thing a correct answer must state, checkable against the excerpt>"}
  ]
}
Rules:
- 3 to 6 rubric claims, each independently checkable.
- Integer weights that SUM TO EXACTLY 100 (weigh the load-bearing claims more).
- claim_type is one of: fact, reasoning, decoy. Prefer including one "decoy" claim naming a
  plausible-but-wrong answer that a correct answer must NOT make.
- Output ONLY the JSON object and nothing else."""


# ------------------------------------------------------------- rubric finalize

def _normalize_weights(claims: list[dict]) -> None:
    """Rewrite claim weights in place to non-negative integers summing to 100
    (largest-remainder apportionment; equal split if the model gave no usable weights)."""
    n = len(claims)
    ws = [max(0.0, float(c.get("weight") or 0)) for c in claims]
    total = sum(ws)
    if total <= 0:
        base, rem = divmod(100, n)
        out = [base + (1 if i < rem else 0) for i in range(n)]
    else:
        scaled = [w / total * 100 for w in ws]
        out = [int(x) for x in scaled]
        rem = 100 - sum(out)
        order = sorted(range(n), key=lambda i: scaled[i] - out[i], reverse=True)
        for i in range(rem):
            out[order[i % n]] += 1
    for c, w in zip(claims, out):
        c["weight"] = w


def finalize_rubric(raw) -> list[dict]:
    """Coerce a model rubric into [{claim_id, claim_type, weight, statement}] with weights
    summing to 100. Drops entries with no statement; re-ids claims c1..cN."""
    claims: list[dict] = []
    if not isinstance(raw, list):
        return []
    for c in raw:
        if not isinstance(c, dict):
            continue
        stmt = str(c.get("statement", "")).strip()
        if not stmt:
            continue
        claims.append({
            "claim_type": (str(c.get("claim_type", "fact")).strip() or "fact"),
            "weight": c.get("weight", 0),
            "statement": stmt,
        })
    if not claims:
        return []
    _normalize_weights(claims)
    return [{"claim_id": f"c{i}", "claim_type": c["claim_type"],
             "weight": c["weight"], "statement": c["statement"]}
            for i, c in enumerate(claims, 1)]


def build_row(span: dict, obj: dict, rid: str) -> dict | None:
    """Assemble one dev-style row from a parsed model object + its source span."""
    q = str(obj.get("question", "")).strip()
    gold = str(obj.get("gold_answer", "")).strip()
    rubric = finalize_rubric(obj.get("rubric"))
    if not (q and gold and rubric):
        return None
    topic = str(obj.get("topic", "")).strip() or Path(span["path"]).stem
    return {
        "id": rid,
        "topic": topic,
        "question": q,
        "gold_answer": gold,
        "rubric": rubric,
        "evidence": [{"path": span["path"],
                      "start_line": span["start_line"],
                      "end_line": span["end_line"]}],
    }


# ------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate dev-style eval data (question + gold + weighted rubric) from a corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_client_args(ap, default_temperature=0.7, default_max_tokens=2048)
    ap.add_argument("--corpus", required=True, help="corpus root (e.g. task/corpus)")
    ap.add_argument("--glob", default="**/*",
                    help="file glob under corpus (fav2 '**/*.txt', dspy '**/*.py')")
    ap.add_argument("--grep", default=None,
                    help="only take spans around lines matching this regex (else random windows)")
    ap.add_argument("--n", type=int, default=100, help="number of spans / eval items")
    ap.add_argument("--span-lines", type=int, default=40, help="lines per evidence span")
    ap.add_argument("--per-span", type=int, default=1, help="eval items per span")
    ap.add_argument("--id-prefix", default="eval", help="row id prefix (ids are <prefix>-NNNN)")
    ap.add_argument("--top-p", type=float, default=None, help="nucleus sampling (fwd to vLLM/SGLang)")
    ap.add_argument("--top-k", type=int, default=None, help="top-k sampling (fwd to vLLM/SGLang)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print sampled spans and exit (no model calls, no server)")
    ap.add_argument("--out", default="", help="output JSONL path (required unless --dry-run)")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    import random
    rng = random.Random(args.seed)
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        raise SystemExit(f"corpus not found: {corpus}")

    spans = list(iter_spans(corpus, args.glob, args.grep, args.n, args.span_lines, rng))
    print(f"[gen-eval] {len(spans)} spans from {corpus} ({args.glob})")

    if args.dry_run:
        for sp in spans[:10]:
            print(f"\n=== {sp['path']}:{sp['start_line']}-{sp['end_line']} ===")
            print(sp["text"][:600])
        print(f"\n[dry-run] {len(spans)} spans total (showed up to 10). No model called.")
        return

    resolve_model(args, ap)
    client = client_from_args(args)

    jobs = [sp for sp in spans for _ in range(max(1, args.per_span))]
    batch = [[{"role": "system", "content": GEN_SYS},
              {"role": "user", "content": f"{span_context(sp)}\n\n{GEN_INSTRUCTIONS}"}]
             for sp in jobs]
    print(f"[gen-eval] authoring {len(batch)} items "
          f"(backend={client.backend} model={client.model})...")
    replies = client.batch_chat(batch, max_workers=args.max_workers,
                                temperature=args.temperature, max_tokens=args.max_tokens,
                                top_p=args.top_p, top_k=args.top_k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written, bad = 0, 0
    with out.open("w") as f:
        for sp, reply in zip(jobs, replies):
            obj = parse_json_object(reply or "")
            row = build_row(sp, obj, f"{args.id_prefix}-{written:04d}") if obj else None
            if row is None:
                bad += 1
                continue
            f.write(json.dumps(row) + "\n")
            written += 1
    print(f"[gen-eval] wrote {written} eval rows -> {out}  ({bad} unparseable/incomplete, dropped)")
    if written == 0:
        raise SystemExit(f"[gen-eval] nothing generated (0 rows written to {out}); "
                         "see errors above")
    print("  next: rubric_eval.py --dev "
          f"{out} --answers candidates.json (or --model $TASK_MODEL --base-url ...)")


if __name__ == "__main__":
    main()
