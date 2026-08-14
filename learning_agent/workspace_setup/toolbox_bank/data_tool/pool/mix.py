"""mix — token-balanced blend of finished pools into one training JSONL.

Offline (no model): reads already-generated --rows pools and repeatedly takes a chunk from
whichever pool has contributed the fewest tokens so far (relative to its --weights target),
so any prefix is itself a balanced blend. Run data_tool/dedup_decontam.py on each pool FIRST, then blend
the clean pools. Logic ported from the Art of Scaling study's blend assembler.

--normalize auto rewrites rows to the {messages:[...]} shape only when the input pools have
mixed shapes (e.g. QA {q,a} rows + doc-style rows), so the output file never silently half-flips
to trace mode; a single shared shape is kept as-is. Token counts use a whitespace word-count
proxy — only used to balance the blend, so the approximation is harmless.

  python3 toolbox/data_tool/mix.py --pools data/fav2_paraphrase.clean.jsonl,data/fav2_qa.clean.jsonl,\\
data/fav2_reasoning.clean.jsonl,data/fav2_implications.clean.jsonl --out data/fav2_mix.rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def approx_tokens(text: str) -> int:
    """Dependency-free token proxy: whitespace word count."""
    return len(text.split())


def row_text(row: dict) -> str:
    """All text in a row (used for token accounting)."""
    if "messages" in row:
        return " ".join(str(m.get("content", "")) for m in row["messages"])
    return f"{row.get('question', '')} {row.get('answer', '')}"


def row_tokens(row: dict) -> int:
    return approx_tokens(row_text(row))


def to_messages(row: dict) -> dict:
    """Normalize a row to the {messages:[...]} shape (loss-masked to assistant turns)."""
    if "messages" in row:
        return {"messages": row["messages"]}
    return {"messages": [{"role": "user", "content": str(row.get("question", ""))},
                         {"role": "assistant", "content": str(row.get("answer", ""))}]}


def load_pool(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and ("messages" in r or ("question" in r and "answer" in r)):
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Token-balanced interleave of finished pools -> one blended --rows JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pools", required=True,
                    help="comma-separated list of input JSONL pool files (one per method)")
    ap.add_argument("--weights", default="",
                    help="comma-separated relative token weights per pool (default: equal)")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="stop after ~this many (approx) tokens (0 = drain all pools)")
    ap.add_argument("--chunk-rows", type=int, default=64,
                    help="interleave granularity: rows taken per pool turn; 64 keeps the "
                         "per-pool token shares balanced across the blend while avoiding long "
                         "single-method runs that would locally skew the mixture")
    ap.add_argument("--normalize", choices=["auto", "messages", "keep"], default="auto",
                    help="auto = to messages iff pools have mixed shapes; messages = always; "
                         "keep = leave rows untouched (only safe if all pools share one shape)")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle each pool before interleaving (pools are usually pre-shuffled "
                         "by data_tool/dedup_decontam.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report pool sizes and planned shares, then exit (no file written)")
    ap.add_argument("--out", default="", help="output JSONL path (required unless --dry-run)")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    pool_paths = [p.strip() for p in args.pools.split(",") if p.strip()]
    if len(pool_paths) < 2:
        ap.error("--pools needs at least 2 files to blend")
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(pool_paths):
            ap.error(f"--weights has {len(weights)} values but --pools has {len(pool_paths)}")
        if min(weights) <= 0:
            ap.error("--weights must be positive")
    else:
        weights = [1.0] * len(pool_paths)

    rng = random.Random(args.seed)
    pools = []
    for path, w in zip(pool_paths, weights):
        rows = load_pool(path)
        if not rows:
            print(f"[mix] WARNING: no usable rows in {path}, skipping")
            continue
        if args.shuffle:
            rng.shuffle(rows)
        toks = sum(row_tokens(r) for r in rows)
        pools.append({"name": Path(path).name, "rows": rows, "weight": w,
                      "avail": toks, "cursor": 0, "contrib": 0})
    if len(pools) < 2:
        raise SystemExit("need at least 2 non-empty pools to blend")

    # shape detection
    shapes = set()
    for p in pools:
        for r in p["rows"]:
            shapes.add("messages" if "messages" in r else "qa")
    mixed = len(shapes) > 1
    normalize = (args.normalize == "messages") or (args.normalize == "auto" and mixed)

    print(f"[mix] {len(pools)} pools; shapes={sorted(shapes)} "
          f"normalize_to_messages={normalize}")
    for p in pools:
        print(f"  {p['name']:32s} rows={len(p['rows']):>8} approx_tokens={p['avail']:>12,} "
              f"weight={p['weight']}")

    if args.dry_run:
        wsum = sum(p["weight"] for p in pools)
        print("[dry-run] planned token shares:")
        for p in pools:
            print(f"  {p['name']:32s} target_share={p['weight'] / wsum:.3f}")
        return

    def take_chunk(p):
        n = min(args.chunk_rows, len(p["rows"]) - p["cursor"])
        if n <= 0:
            return []
        chunk = p["rows"][p["cursor"]:p["cursor"] + n]
        p["cursor"] += n
        return chunk

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tot = n_rows = 0
    with out.open("w") as f:
        while True:
            if args.target_tokens and tot >= args.target_tokens:
                break
            cands = [p for p in pools if p["cursor"] < len(p["rows"])]
            if not cands:
                break
            # least-contributed source next (relative to its weight)
            p = min(cands, key=lambda x: x["contrib"] / x["weight"])
            for row in take_chunk(p):
                t = row_tokens(row)
                p["contrib"] += t
                tot += t
                n_rows += 1
                out_row = to_messages(row) if normalize else row
                f.write(json.dumps(out_row) + "\n")

    print(f"[mix] wrote {n_rows} rows ~{tot:,} tokens -> {out}")
    for p in pools:
        print(f"  {p['name']:32s} contributed ~{p['contrib']:>12,} tokens "
              f"({p['contrib'] / max(tot, 1):.1%})")


if __name__ == "__main__":
    main()
