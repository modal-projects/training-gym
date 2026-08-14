# reasoning — reasoning-then-Conclusions traces over full documents

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

Doc-style internalization with a reasoning register: the model works through
the document's facts/entities/quantities/dates in prose, then commits to a
"Conclusions:" section. Teaches the task model to reason over the corpus, not
just recite it.

## Contract

Output JSONL, doc-style rows (loss on the assistant turn):

    {"messages": [
        {"role": "user", "content": "Study this document from the corpus: <rel/path>"},
        {"role": "assistant", "content": "<step-by-step reasoning ...>\n\nConclusions:\n..."}]}

## Recipe

1. One full corpus doc in context per request.
2. Ask for step-by-step reasoning through the document's content — then a
   final `Conclusions:` section that states what was established, preserving
   every name/number/date.
3. One row per doc. Whether this register helps your corpus + model is for
   you to MEASURE (eval_tool), not assume.

## Pitfalls

- The payload comes LAST — which is exactly what generation-cap truncation
  eats first. A trace that rambles and never reaches `Conclusions:` is the
  single most likely failure. VALIDATE the section exists (a one-line
  `"Conclusions:" in text` check catches both truncation and format drift);
  drop rows that fail.
- Strip preambles; reject replies that are summaries without reasoning.
- Then `pool/dedup_decontam.py`, like every pool.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
