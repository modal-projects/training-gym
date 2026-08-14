"""dedup_decontam — finalize an internalization pool: dedup + 13-gram decontam + verbatim seeding.

Offline (no model). Run on every pool before training. Pipeline (ported from the Art of Scaling
study's finalize step):

  1. EXACT dedup — drop rows whose normalized train text repeats (md5 of lowercased words).
  2. NEAR dedup  — MinHash + LSH on word 5-gram shingles; drop Jaccard > --near-threshold
                   (default 0.8, 128 perms). Pure-Python, no third-party dep.
  3. DECONTAM    — drop any row sharing a 13-gram with the eval file's question /
                   gold_answer / rubric statements / answers / choices (protects the
                   held-out probes). Evidence excerpts are corpus text and are NOT
                   collected. Skipping this step requires an explicit --allow-no-eval.
  4. SHUFFLE     — global shuffle so any prefix is an unbiased nested subset.
  5. SEED        — inject one verbatim corpus copy every --seed-every tokens, first at 0.

Row shapes accepted: {question, answer}, {messages: [...]}, and DPO pairs
{prompt, chosen, rejected} (deduped/decontaminated on prompt+chosen+rejected; seeding
is refused for DPO pools). --eval-questions removes leakage ONLY (point it at
dev.json); never point --corpus at gold.
Seed rows are doc-style; if the input pool is plain {question, answer} and seeding is on, the
output is normalized to the messages shape so the --rows file never half-flips to trace mode.
Token counts use a whitespace word-count proxy (drives seeding cadence only).

  python3 toolbox/data_tool/pool/dedup_decontam.py --in data/fav2_paraphrase.rows.jsonl --out data/fav2_paraphrase.clean.jsonl \\
      --eval-questions task/dev2/dev.json --corpus tasks/fav2/corpus --glob '**/*.txt' \\
      --seed-every 1000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+")
MERSENNE = (1 << 61) - 1


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def norm_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def word_ngrams(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def is_dpo_row(row: dict) -> bool:
    return "chosen" in row and "rejected" in row


def train_text(row: dict) -> str:
    """The text a row contributes to loss: doc/messages -> assistant turns; QA -> answer;
    DPO pair -> chosen + rejected (both drive the DPO loss)."""
    if "messages" in row:
        asst = [m.get("content", "") for m in row["messages"] if m.get("role") == "assistant"]
        return " ".join(asst) if asst else " ".join(m.get("content", "") for m in row["messages"])
    if is_dpo_row(row):
        return f"{row.get('chosen', '')} {row.get('rejected', '')}"
    return str(row.get("answer", ""))


def full_text(row: dict) -> str:
    """All text in a row (for decontamination)."""
    if "messages" in row:
        return " ".join(str(m.get("content", "")) for m in row["messages"])
    if is_dpo_row(row):
        return (f"{row.get('system', '')} {row.get('prompt', '')} "
                f"{row.get('chosen', '')} {row.get('rejected', '')}")
    return f"{row.get('question', '')} {row.get('answer', '')}"


def approx_tokens(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# pure-python MinHash + LSH (word 5-gram shingles)
# --------------------------------------------------------------------------- #
def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def make_perms(num_perm: int, seed: int = 1):
    rng = random.Random(seed)
    return [(rng.randrange(1, MERSENNE), rng.randrange(0, MERSENNE)) for _ in range(num_perm)]


def minhash(words: list[str], perms, shingle: int = 5) -> tuple[int, ...]:
    shingles = word_ngrams(words, shingle) or set(words) or {""}
    base = [_hash64(s) for s in shingles]
    sig = []
    for a, b in perms:
        sig.append(min(((a * h + b) % MERSENNE) for h in base))
    return tuple(sig)


def choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) with bands*rows=num_perm whose LSH threshold (1/bands)^(1/rows)
    is closest to the requested Jaccard threshold."""
    best = None
    for b in range(1, num_perm + 1):
        if num_perm % b:
            continue
        r = num_perm // b
        t = (1.0 / b) ** (1.0 / r)
        err = abs(t - threshold)
        if best is None or err < best[0]:
            best = (err, b, r)
    return best[1], best[2]


def near_dedup(rows: list[dict], threshold: float, num_perm: int, shingle: int = 5):
    """Return (kept_rows, n_dropped). LSH banding over MinHash signatures."""
    perms = make_perms(num_perm)
    bands, rows_per_band = choose_bands(num_perm, threshold)
    seen_buckets = [set() for _ in range(bands)]
    kept = []
    dropped = 0
    for row in rows:
        sig = minhash(norm_words(train_text(row)), perms, shingle)
        keys = []
        is_dup = False
        for bi in range(bands):
            band = sig[bi * rows_per_band:(bi + 1) * rows_per_band]
            k = _hash64(f"{bi}:" + ",".join(map(str, band)))
            keys.append((bi, k))
            if k in seen_buckets[bi]:
                is_dup = True
        if is_dup:
            dropped += 1
            continue
        for bi, k in keys:
            seen_buckets[bi].add(k)
        kept.append(row)
    return kept, dropped


# --------------------------------------------------------------------------- #
# eval decontamination n-grams
# --------------------------------------------------------------------------- #
def _collect_strings(obj) -> list[str]:
    """Eval-side strings that must not leak into training text.

    Curated keys only: `evidence[].excerpt` is verbatim CORPUS text and must
    never be collected — training data legitimately derives from the corpus,
    and decontaminating against excerpts would drop it wholesale.
    """
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            out += _collect_strings(x)
    elif isinstance(obj, dict):
        for key in ("question", "answer", "prompt", "gold_answer"):
            if isinstance(obj.get(key), str):
                out.append(obj[key])
        for key in ("answers", "choices", "options", "gold", "targets"):
            v = obj.get(key)
            if isinstance(v, list):
                out += [str(x) for x in v if not isinstance(x, (dict, list))]
            elif isinstance(v, str):
                out.append(v)
        for c in obj.get("rubric") or []:
            if isinstance(c, dict) and isinstance(c.get("statement"), str):
                out.append(c["statement"])
        # wrapper shapes: {"questions": [...]} etc.
        for key in ("questions", "items", "rows", "data", "examples"):
            v = obj.get(key)
            if isinstance(v, list):
                out += _collect_strings(v)
    return out


def load_eval_grams(path: str, ngram: int) -> set[str]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    items = []
    # try JSONL first, then whole-file JSON (object or list)
    parsed_jsonl = True
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            parsed_jsonl = False
            break
    if not parsed_jsonl or not items:
        try:
            doc = json.loads(raw)
            items = doc if isinstance(doc, list) else [doc]
        except json.JSONDecodeError as e:
            raise SystemExit(f"could not parse --eval-questions {path}: {e}")
    grams = set()
    for it in items:
        for s in _collect_strings(it):
            grams |= word_ngrams(norm_words(str(s)), ngram)
    if not grams:
        raise SystemExit(
            f"--eval-questions {path}: no {ngram}-grams collected — the decontamination "
            f"gate would be a silent no-op. Check the file shape (rows need question/"
            f"gold_answer/rubric keys), or lower --decontam-ngram if the eval strings "
            f"are shorter than {ngram} words.")
    return grams


# --------------------------------------------------------------------------- #
# verbatim corpus seed
# --------------------------------------------------------------------------- #
def load_corpus_docs(corpus: Path, glob: str, max_chars: int) -> list[tuple[str, str]]:
    """(relative path, text) per doc — the rel path feeds the seed row's user turn."""
    docs = []
    for p in sorted(corpus.glob(glob)):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        if text.strip():
            docs.append((str(p.relative_to(corpus)), text))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Finalize a pool: exact+near dedup, 13-gram decontam, verbatim seeding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input pool JSONL (--rows shape)")
    ap.add_argument("--out", default="", help="output JSONL path (required unless --dry-run)")
    ap.add_argument("--eval-questions", default="",
                    help="JSON/JSONL of eval questions to decontaminate against (drop rows "
                         "sharing a 13-gram); REQUIRED unless --allow-no-eval is passed")
    ap.add_argument("--allow-no-eval", action="store_true",
                    help="explicitly skip decontamination (only when the track truly has "
                         "no eval file); without this flag, omitting --eval-questions is "
                         "a hard error")
    ap.add_argument("--decontam-ngram", type=int, default=13,
                    help="decontamination n-gram order; 13 is the standard contamination "
                         "order — long enough not to fire on common phrasing, short enough "
                         "to catch verbatim eval-question overlap")
    ap.add_argument("--near-threshold", type=float, default=0.8,
                    help="MinHash Jaccard threshold for near-dedup (report: 0.8)")
    ap.add_argument("--minhash-perms", type=int, default=128,
                    help="MinHash permutations; 128 is the standard accuracy/cost point for "
                         "Jaccard estimation (error ~ 1/sqrt(perms))")
    ap.add_argument("--shingle", type=int, default=5,
                    help="word-ngram size for MinHash shingles; 5 words is the standard "
                         "granularity (shorter over-connects unrelated text, longer misses near-dupes)")
    ap.add_argument("--no-near-dedup", action="store_true", help="skip MinHash near-dedup")
    # verbatim seeding
    ap.add_argument("--corpus", default="", help="corpus root for verbatim seeding (omit to skip)")
    ap.add_argument("--glob", default="**/*", help="file glob under --corpus for seed docs")
    ap.add_argument("--seed-every", type=int, default=0,
                    help="inject one verbatim corpus copy every N (approx) tokens, first at "
                         "position 0 (0 = no seeding; report used 100_000_000)")
    ap.add_argument("--seed-max-doc-chars", type=int, default=0,
                    help="truncate seed docs longer than this many chars (0 = whole doc)")
    ap.add_argument("--no-shuffle", action="store_true", help="do not shuffle the kept rows")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed")
    ap.add_argument("--dry-run", action="store_true",
                    help="print diagnostics only; do not write --out")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")

    # load pool
    rows = []
    for line in Path(args.inp).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and ("messages" in r
                                    or ("question" in r and "answer" in r)
                                    or ("prompt" in r and "chosen" in r and "rejected" in r)):
            rows.append(r)
    n_input = len(rows)
    if n_input == 0:
        raise SystemExit(f"no usable rows in {args.inp}")
    print(f"[hygiene] {n_input} input rows from {args.inp}")

    # (1) exact dedup on normalized train text
    seen, kept = set(), []
    for row in rows:
        key = hashlib.md5(" ".join(norm_words(train_text(row))).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    n_exact = n_input - len(kept)

    # (2) near dedup (MinHash + LSH)
    n_near = 0
    if not args.no_near_dedup:
        kept, n_near = near_dedup(kept, args.near_threshold, args.minhash_perms, args.shingle)

    # (3) decontamination
    n_decontam = 0
    if args.eval_questions:
        eval_grams = load_eval_grams(args.eval_questions, args.decontam_ngram)
        kept2 = []
        for row in kept:
            if word_ngrams(norm_words(full_text(row)), args.decontam_ngram) & eval_grams:
                n_decontam += 1
                continue
            kept2.append(row)
        kept = kept2
        print(f"[hygiene] decontam: {len(eval_grams)} eval {args.decontam_ngram}-grams, "
              f"dropped {n_decontam}")
    elif args.allow_no_eval:
        print("[hygiene] WARNING: decontamination skipped (--allow-no-eval)")
    else:
        raise SystemExit(
            "no --eval-questions given; decontamination is the leakage gate. Pass "
            "--eval-questions <dev.json or your eval file>, or opt out explicitly "
            "with --allow-no-eval if the track truly has no eval file.")

    # (4) shuffle
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(kept)

    # (5) seeding plan
    seed_docs = []
    if args.corpus and args.seed_every > 0:
        corpus = Path(args.corpus)
        if not corpus.is_dir():
            raise SystemExit(f"corpus not found: {corpus}")
        seed_docs = load_corpus_docs(corpus, args.glob, args.seed_max_doc_chars)
        if not seed_docs:
            print(f"[hygiene] WARNING: no seed docs under {corpus} ({args.glob}); skipping seeding")
    elif args.seed_every > 0:
        print("[hygiene] WARNING: --seed-every set but no --corpus given — skipping seeding")

    do_seed = bool(seed_docs) and args.seed_every > 0
    # a verbatim corpus copy = all docs as doc-style rows (user turn required by
    # chat templates; loss is masked to the assistant turn = the doc text)
    seed_rows = [{"messages": [
        {"role": "user", "content": f"Study this document from the corpus: {rel}"},
        {"role": "assistant", "content": d},
    ]} for rel, d in seed_docs]
    seed_copy_tokens = sum(approx_tokens(train_text(r)) for r in seed_rows)

    # DPO pairs cannot mix with doc-style seed rows — one file, one trainer contract.
    if do_seed and any(is_dpo_row(r) for r in kept):
        raise SystemExit("verbatim seeding applies to SFT pools only; this pool contains "
                         "{prompt, chosen, rejected} DPO rows — rerun without --seed-every")

    # shape normalization: if we inject messages seed rows into a plain-{q,a} pool, keep the
    # file consistent by normalizing every row to messages.
    has_plain = any("messages" not in r and not is_dpo_row(r) for r in kept)
    normalize = do_seed and has_plain

    def emit_row(row: dict) -> dict:
        if normalize and "messages" not in row and not is_dpo_row(row):
            return {"messages": [{"role": "user", "content": str(row.get("question", ""))},
                                 {"role": "assistant", "content": str(row.get("answer", ""))}]}
        return row

    n_kept = len(kept)
    print(f"[hygiene] kept {n_kept} rows  (exact_dups={n_exact} near_dups={n_near} "
          f"decontam_dropped={n_decontam})")
    if do_seed:
        print(f"[hygiene] seeding: {len(seed_rows)} doc rows (~{seed_copy_tokens:,} tokens) "
              f"per copy, every {args.seed_every:,} tokens (first at position 0)"
              + ("; normalizing pool to messages shape" if normalize else ""))

    if args.dry_run:
        print("[dry-run] no file written.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Cadence is driven by POOL tokens only. The injected seed (a whole corpus copy) must NOT
    # count toward the trigger, or — when the corpus is larger than --seed-every — every copy
    # would immediately re-cross the anchor and re-seed on the next row, drowning the pool.
    pool_tok = next_anchor = written = seed_copies = seed_written = 0
    with out.open("w") as f:
        for row in kept:
            if do_seed and pool_tok >= next_anchor:
                for sr in seed_rows:
                    f.write(json.dumps(sr) + "\n")
                    seed_written += 1
                seed_copies += 1
                next_anchor += args.seed_every
            f.write(json.dumps(emit_row(row)) + "\n")
            written += 1
            pool_tok += approx_tokens(train_text(row))
        # if seeding is on but the pool never reached the first boundary, still seed once at 0
        if do_seed and seed_copies == 0:
            for sr in seed_rows:
                f.write(json.dumps(sr) + "\n")
                seed_written += 1
            seed_copies = 1

    total = written + seed_written
    print(f"[hygiene] wrote {total} rows -> {out}  "
          f"(pool={written}, seed_copies={seed_copies}, seed_rows={seed_written}, "
          f"~{pool_tok:,} pool tokens)")


if __name__ == "__main__":
    main()
