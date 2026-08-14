# harness_tool — the task model's answering program

The task agent answers through EXACTLY ONE harness; `submission/eval.py` is
that choice made executable. The starters seeded for your task are the
`.py` files in this folder: each file's docstring says what it is, and the
TOOLS.md catalog lists them.

Rules:

1. Pick one starter (or write your own), wire it into `submission/eval.py`,
   and evaluate through it — mixing harnesses mid-eval makes the margin
   unattributable.
2. Harness optimization is your own capability: rewrite the prompt, the
   strategy, the tool set — or design NEW tools for the task model.
   Score candidate harnesses on your dev set like any checkpoint.
3. No external LLM calls anywhere in the answer path (rule 6 in AGENTS.md).
