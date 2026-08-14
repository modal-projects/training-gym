# paraphrase — WRAP-style rephrasing of full documents

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

Doc-style internalization: the task model sees each corpus document restated in
several registers, which beats raw-doc repetition for recall (the WRAP
result). Cheap, deterministic to orchestrate, works on any text corpus.

## Contract

Output JSONL, doc-style rows:

    {"messages": [
        {"role": "user", "content": "Study this document from the corpus: <rel/path>"},
        {"role": "assistant", "content": "<the rewrite>"}]}

Chat templates require a user turn; loss is masked to the assistant turn, so
the rewrite is what gets internalized.

## Recipe

1. Read one full corpus doc into context per request.
2. Rewrite it in a spread of styles — e.g. easy (explain to a child), medium
   (encyclopedia entry), hard (dense technical prose) — with the hard
   constraint: preserve every fact, name, number, and date.
3. One row per (doc, style). Style mix and prompts are yours to tune;
   whether this format helps your corpus + model is for you to MEASURE
   (eval_tool), not assume.

## Pitfalls

- Truncation is the killer. A whole-doc rewrite of a long doc cannot fit the
  generation cap, and the shared client does not surface `finish_reason` —
  so VERIFY completeness yourself (length ratio vs source, terminal
  punctuation, chunk long docs). A mid-sentence half-paraphrase that
  "preserves every fact" of the first half is poison that looks like food.
- Strip conversational preambles ("Here is the rewritten document:") —
  prompts demanding "output only the rewrite" are routinely ignored.
- Docs that overflow the model's context fail per-request; if you silently
  drop them your pool skews toward short docs. Count and report.
- Then `pool/dedup_decontam.py`, like every pool.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
