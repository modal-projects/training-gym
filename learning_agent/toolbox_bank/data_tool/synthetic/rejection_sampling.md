# rejection_sampling — sample the task model K times, keep the best

Method card: you implement this (write your own generator under `data_tool/`),
the card tells you the contract, the recipe, and the traps.

## When to use

On-policy SFT data without a judge: sample K completions per question FROM
THE CURRENT TASK MODEL, keep the best one by a cheap quality filter, train on
the survivors. Training on the task model's own accepted outputs (instead of
foreign reference text) causes less forgetting — the on-policy rationale with a
selection signal on top.

## Contract

Input: `{question}` rows, optionally with a reference `answer`.
Output: `{question, answer}` JSONL where `answer` is the kept sample.

## Recipe

1. Sample K completions per question from the served task model (temperature
   up, e.g. 0.8–1.0 — identical samples carry no signal).
2. Filter, two starter options (no external judge needed):
   - consistency: keep the medoid — the sample most similar to its siblings
     by token-F1; fully reference-free.
   - gold-F1: keep the best sample by token-F1 against the reference answer;
     drop questions whose best is below a threshold.
   (Or use the pinned judge via `api_clients/judge_client` as the filter —
   costlier, sharper.)
3. Emit survivors; then `pool/dedup_decontam.py`.

## Pitfalls

- Verify the endpoint honors `n=K`. Some OpenAI-compatible servers silently
  return one choice; with K=1 a consistency filter degenerates into "keep
  everything" with no quality signal. Assert `len(choices) == K` (or make K
  separate requests).
- Fail LOUDLY on a dead endpoint: the natural failure mode of
  catch-retry-return-empty is a zero-row output file with exit 0, which
  downstream mixing silently ships. Zero kept rows = non-zero exit.
- Don't mix filters silently: if some rows lack a reference and you fall
  back from gold-F1 to consistency for just those, say so in the summary —
  or split the run.
- Use the shared `api_clients/oai_client.py` (`batch_chat`) rather than
  hand-rolling HTTP: it gets you retry hygiene, failure counts, mock-backend
  smoke tests, and `$TASK_MODEL` resolution for free.

Starting point, not a menu: vary it, combine it, or replace it with a better
method — new methods are scored like everything else, dev margin over base.
