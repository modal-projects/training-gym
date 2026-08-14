# annotation — expert margin-note (gloss) examples from full documents

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

Doc-style internalization that pairs source text with an expert's reading of
it: quote the load-bearing passages, and gloss each one — what it means, why
it matters, how it connects. The task model internalizes text AND commentary.
(Locally-invented method, not from the Art of Scaling study — measure it.)

## Contract

Output JSONL, doc-style rows (loss on the assistant turn):

    {"messages": [
        {"role": "user", "content": "Study this document from the corpus: <rel/path>"},
        {"role": "assistant", "content": "\"<quoted passage>\"\nNote: <gloss>\n\n\"<quote>\"\nNote: ..."}]}

## Recipe

1. One full corpus doc in context per request.
2. Ask for an annotated reading: quote the important passages / terms /
   figures / claims verbatim, and after each quote write a concise, grounded
   note.
3. One row per doc.

## Pitfalls

- Validate the structure: a reply with ZERO `Note:` blocks means the model
  ignored the format and summarized — those rows don't teach the gloss
  behavior at all. A reply ending in a dangling quote with no note is the
  truncation signature. Drop both.
- This method reproduces MORE verbatim corpus text in assistant turns than
  any other — which is the point, but it makes `pool/dedup_decontam.py`
  (with `--eval-questions`) maximally load-bearing here. Never skip it.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
