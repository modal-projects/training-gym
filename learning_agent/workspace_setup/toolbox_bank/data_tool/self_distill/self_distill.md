# self_distill — SDFT rewriting of reference answers into the task model's register

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

You have `{question, answer}` rows whose answers came from somewhere other
than the task model (grounded_qa with a different generator, hand-written
references) and you want to SFT without distribution-gap forgetting. Method
from "Self-Distillation Bridges Distribution Gap in Language Model
Fine-Tuning" (sail-sg/sdft): the task model restates each reference answer in
its own words; training uses the rewrite, keeping the content while staying
on-policy. (For the same-model-with-context-in-window variant, see
`training_tool/self_distillation` — that's a trainer, not a data method.)

## Contract

Input: `{question, answer}` JSONL (e.g. a grounded_qa pool).
Output: same shape — answers replaced by the task model's rewrites. PRESERVE
extra fields (`evidence`, ids): downstream tools need them, and silently
dropping them breaks the grounded_qa -> self_distill -> react_trace_gen
pipeline.

## Recipe

1. For each row, show the task model the question + reference answer and ask it
   to restate the answer in its own words, preserving all content.
2. Emit the rewrite as the new `answer`. Drop empty rewrites — and COUNT the
   drops; a 40% silent drop rate looks like a smaller-but-fine pool.
3. Train the result with a plain SFT package (`training_tool/automodel` /
   `unsloth`).

## Pitfalls

- THE HARD RULE of this method: the rewriter MUST be the served task model.
  This is the one generator where "point it at any model" does not apply —
  a rewrite by any other model reintroduces the exact distribution gap the
  method exists to remove, and nothing in the output will look wrong.
  Verify your endpoint is the task model before the run, and fail loudly if the
  endpoint is unreachable rather than falling back to anything else.
- Rewriting collapses distinct reference answers into similar task model
  phrasings — NEW near-duplicates the input's earlier dedup never saw. Run
  `pool/dedup_decontam.py` on the OUTPUT, even if the input was clean.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
