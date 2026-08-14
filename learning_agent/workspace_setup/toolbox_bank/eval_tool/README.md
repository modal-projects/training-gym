# eval_tool — score the task model

- `gen_eval.py` — author a dev set from the corpus (`{id, question,
  gold_answer, rubric}` rows). Use it when your track ships no dev.json.
- `rubric_eval.py` — score answers against weighted-claim rubrics → mean +
  bootstrap confidence interval. Accepts a precomputed answers file, or
  generates from a served endpoint.

Rules:

1. Judge with gpt-5.6-luna: with `LEARNING_AGENT_JUDGE_URL` in `.env` (workspaces have
   it) the judge calls are canonical — the SAME judge that scores your
   submission. Use it freely wherever you need an LLM judge.
2. Evaluate through the task model's FULL harness (tools, search), exactly as
   `submission/eval.py` will answer — not bare chat completion.
3. The number that counts is the margin over the untrained base, not the
   absolute score. Score the base through the same harness once.
4. The dev set steers you; it must never enter training data.
