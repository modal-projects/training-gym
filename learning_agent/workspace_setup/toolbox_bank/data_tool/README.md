# data_tool — make training data

How to make data is YOUR capability, not the toolbox's. The method folders
hold METHOD CARDS (`.md`) — contract + recipe + pitfalls — not runnable
scripts: you write the generator, they tell you what has to be true.

<!-- if:sft -->
- `synthetic/` — cards for generators over the task corpus: grounded_qa,
  paraphrase, implications, reasoning, annotation,
  rejection_sampling. Output: `{question, answer}` or doc-style `{messages}`
  examples (JSONL, one per line).
- `agentic/` — react_trace_gen card: multi-turn grep/read traces →
  `{messages}` examples (teaches search behavior; loss masked to assistant
  turns).
<!-- endif:sft -->
<!-- if:rl,opd -->
- `rl/` — the RL data contract: prompts file shape + how to write the
  reward (its own README).
<!-- endif:rl,opd -->
<!-- if:opd -->
- `self_distill/` — the self-distillation data-transform card (its own md).
<!-- endif:opd -->
- `pool/` — dedup_decontam and mix, the two RUNNABLE tools here. They are
  the contract, not a method: run dedup_decontam on EVERY dataset before
  training.

The cards are starting points, not a menu. Explore: vary a method, combine
methods, research and invent new ones — the literature moves faster than
this folder. A new method needs no registration: measure it like everything
else (dev margin over the untrained base, through the full harness).

Rules:

1. The generator model is whatever endpoint you point at — the served
   task model or your own model; you may also write examples by hand.
2. Data derives from the task corpus and generations made inside this run.
   No external datasets.
3. Never train on dev questions — `pool/dedup_decontam` with
   `--eval-questions` is the gate, and it now hard-fails when skipped
   (explicit `--allow-no-eval` to opt out). It accepts `{question, answer}`
   and `{messages}` shapes.
4. Write your generators as Python files UNDER `data_tool/` (any name,
   docstring first line = what it does). A conforming file is automatically
   a tool: the run timeline picks up every executed toolbox path, so your
   methods stay visible and reproducible. Shared plumbing so you don't
   re-roll it: `api_clients/oai_client.py` (batch chat, retries, mock
   backend for offline smoke) and `data_tool/corpus_sampling.py` (doc
   listing/reading, span sampling).
