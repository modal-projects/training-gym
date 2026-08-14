# implications — declarative expansion of full documents

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

Doc-style internalization that goes one step past restating: the model lists
what the document ENTAILS — consequences, connections, restatements of
buried facts — one declarative statement per line. Complements `paraphrase`
(same facts, different transformation).

## Contract

Output JSONL, doc-style rows (loss on the assistant turn):

    {"messages": [
        {"role": "user", "content": "Study this document from the corpus: <rel/path>"},
        {"role": "assistant", "content": "<implication>\n<implication>\n..."}]}

## Recipe

1. One full corpus doc in context per request.
2. Ask for the implications entailed by the document, one per line,
   preserving every name/number/date. No meta-text, no numbering chatter.
3. One row per doc (or per doc-section for long docs).

## Pitfalls

- THE length-cap trap: models write unbounded lists that slam the generation
  cap and truncate mid-line. Enforce BOTH a max-lines and a max-chars gate
  and DROP over-long rows — and if you trim instead, re-check both gates
  after trimming (a naive line-slice on a row that was over on chars is a
  no-op that ships the truncation artifact anyway).
- The client doesn't surface `finish_reason`; a truncated reply that happens
  to land under your caps is invisible — prefer per-line completeness checks
  (does the last line end like a sentence?).
- Then `pool/dedup_decontam.py`, like every pool.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
