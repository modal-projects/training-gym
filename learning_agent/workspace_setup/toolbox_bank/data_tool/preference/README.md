# preference — judge-ranked DPO pairs from the current task model

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

You can tell better answers from worse ones (a rubric + the pinned judge)
and want the task model to move toward its own good samples — DPO on
chosen/rejected pairs, fully on-policy.

## Contract

Output JSONL, one pair per line (see `dpo_pairs.example.jsonl`):

    {"system": "<optional — the task sys.txt>",
     "prompt": "<the question>",
     "chosen": "<the better answer>",
     "rejected": "<the worse answer>"}

This is axolotl's DPO input (`rl: dpo`, chatml.prompt_pairs —
`training_tool/axolotl/docs/rlhf.qmd`) and trl DPOTrainer's. Note the DPO
learning rate is ~1000x below SFT's — training starts at the reference
optimum. `pool/dedup_decontam.py` accepts this shape — pairs go through the
gate like every other pool.

## Recipe

1. Take eval-STYLE rows `{question, gold_answer?, rubric}` — authored with
   `eval_tool/gen_eval.py` or built from your synthetic pools. NEVER dev.json
   (see pitfalls).
2. Sample K completions per question from the served task model (on-policy).
3. Score EACH sample with the weighted-claim judge
   (`api_clients/judge_client.judge_claims`) against the row's rubric — the
   BIG pinned judge ranks, never the task model itself.
4. Pair chosen = highest-scoring, rejected = lowest-scoring. Keep the pair
   only when (chosen - rejected) >= a real margin (~0.25) AND chosen clears
   a floor (never teach a best-of-bad answer). One pair per question, max.
5. A question whose judging fails is SKIPPED and counted — never zero-scored
   into a pair.

## Pitfalls

- DO NOT build pairs from `task/<task>/dev.json`. The dev set is your
  steering instrument; training DPO on dev questions is dev-set leakage by
  construction, and inflated dev scores then corrupt every model-selection
  decision downstream. Author train-side questions with gen_eval instead.
- Endpoints that ignore `n=K` return one choice: chosen == rejected, and
  with a zero margin gate you'd emit identical-text pairs. Assert K samples;
  keep the margin gate above zero.
- Judge calls are seconds each and K x n_votes per question — write pairs
  incrementally so a mid-run failure doesn't discard everything paid for.
- Fail loudly (non-zero exit) when zero pairs survive; an empty pairs file
  with exit 0 ships silently.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
