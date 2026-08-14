# grounded_qa — two-step corpus-grounded QA generation

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

You want closed-book QA rows that are answerable from the task corpus — the
bread-and-butter SFT pool for knowledge-internalization tracks, and the seed
data for `react_trace_gen` (search traces) and rejection sampling (on-policy
rewrites).

## Contract

Output JSONL, one row per line:

    {"question": "...", "answer": "...",
     "evidence": [{"path": "<rel/path>", "start_line": 12, "end_line": 48}]}

`question`/`answer` is what trainers read; `evidence` (the span the QA came
from) is what lets `react_trace_gen` build real search traces later — record
it even if you don't need it yet. Generator model: whatever endpoint you
point at (the served task model or your own model) — see `data_tool/README.md`.

## Recipe

1. Sample REAL spans from `--corpus` (grep hits, fixed-size windows, whole
   short docs). `data_tool/corpus_sampling.py` has the helpers;
   `api_clients/oai_client.py` (`batch_chat`) is the client.
2. Step A: span -> a detailed, self-contained, closed-book question about it.
3. Step B: span + question -> the gold answer, grounded in the span only.
4. Record the span as `evidence` on the row.

Two steps beat one (ask-and-answer in one call drifts off-span). Prompting,
span policy, and answer format are yours to tune per task.

## Pitfalls

- Models love to wrap step-A output in JSON (`{"question": ...}`). Parse or
  reject it on BOTH steps — a non-empty step-B answer does not mean step A
  was clean; JSON blobs in the `question` field train the task model to expect
  garbage questions.
- Strip `Q:`/`Answer:` prefixes carefully: a naive case-insensitive
  `^(answer|a)[:\-]` also eats real answers that begin "A: ..." (option
  letters!).
- Distinguish "endpoint failed" from "model said unanswerable" in your drop
  accounting, or a flaky server reads as bad data.
- This format duplicates heavily. `pool/dedup_decontam.py` is not optional.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
