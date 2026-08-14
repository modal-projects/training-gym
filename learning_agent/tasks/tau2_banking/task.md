# Task: tau2_banking  (agentic — knowledge-grounded)

## Corpus
`tasks/tau2_banking/corpus/` is a pinned snapshot of Sierra Research's τ²-bench
`banking_knowledge` domain (tau2-bench@`aa74303c`) — the knowledge/RAG domain:

- `documents/` — **698 product-knowledge documents** (JSON, ~2.8 MB total): checking,
  savings, credit cards, business accounts, BNPL, "everyone pay" transfers, support
  codes, subscriptions. Filenames encode the topic
  (`doc_<category>_<product>_<NNN>.json`). This is the knowledge base to retrieve from.
- `db.json` (~268 KB) — customer/account records (the factual state a question may
  reference).

There is **no `policy.md`** — the "policy" is distributed across the 698 documents.

## What this task is
The knowledge-grounded τ² domain: the customer asks banking questions whose answers
live in the document base, and the right response is usually retrieval **plus an
account action** (apply for the card that matches the stated constraints, set up the
transfer the documents allow, …). The student handles the conversation, retrieves
via the domain's own search/read tools, and acts on the account — scored on the
resulting database state and required actions, like the other τ² domains, not on
prose. What changes is where the difficulty sits: grounding in 698 documents
instead of a single policy file.

## Environment + protocol (pinned)

- `tau2-bench @ aa74303c` (task.yaml `env.pin`) — a LATER pin than the
  transactional domains (this domain's data changed after `1901a301`; the pin is
  the commit the corpus snapshot was taken from). Run through tau2's NATIVE
  orchestrator via the same `harness/adapters/tau2.py` batch `run_split` seam.
- Retrieval: the domain's default variant (`alltools` — search/read tools over
  `documents/`; no embedding dependency).
- Protocol: `max_steps: 200`, `num_trials: 4` (secondary = **pass^4**),
  `seed: 300`; policy sampling = the Qwen3.5 thinking-mode recipe
  (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5).
- Customer simulator: **gpt-5.6-luna** (reasoning_effort low) — the SAME model
  for dev rollouts and official scoring (see tasks/tau2_airline/task.md for why).
  Reached keylessly via `LEARNING_AGENT_USER_SIM_URL`; `provenance.user_llm` records it per run.
- Reward per episode: tau2's check over final DB state + required actions,
  gated per scenario. Row score = mean reward over the 4 trials.

## Splits

tau2 ships **no** `split_tasks.json` for this domain (97 scenarios in
`tasks.json`), so the split is Learning Agent's, deterministic and recorded in `brief.md`:

```python
ids = [t["id"] for t in tasks.json]        # shipped order
random.Random(0).shuffle(ids)
dev, test = sorted(ids[:58]), sorted(ids[58:])   # 58 / 39, ≈60/40 like airline
```

- `dev.json`  — 58 scenarios (your iteration signal)
- `test.json` — 39 scenarios (the held-out number)

## Running it

Dev-time (in your workspace — same simulator as scoring, 1 trial instead of 4):

```bash
python bench.py rollout --task tau2_banking --split dev \
    --model /out/models/<tag>/merged --config tasks/tau2_banking/dev.yaml
```

Scoring (operator, pinned simulator + 4 trials):

```bash
python bench.py score --task tau2_banking --model <weights> --split dev
# = bench.py rollout (env-verified; no LLM judge). Headline = margin over the
# untrained base floor (--model Qwen/Qwen3.5-9B), as everywhere in Learning Agent.
```

Base floor on this protocol (Qwen3.5-9B, dev, 58 scenarios x 4 trials,
gpt-5.6-luna simulator, 2026-08-09): **0.0776** mean reward
(CI [0.03, 0.13]), **pass^4 = 0.0** — the untrained student essentially
floors: it cannot navigate the 698-document knowledge base and land the
right account actions. Frontier reference (GLM-5.2-FP8, provider sampling
per dev/glm52_baseline.yaml): 0.3664 / pass^4 0.1552 — hard for everyone,
but a ~0.29 mean-reward gap. The widest margin headroom in the tau2 suite.

Requires the agentic-shell sandbox binaries wherever episodes run (srt +
ripgrep, plus bubblewrap + socat on Linux) — the default `alltools`
retrieval variant refuses to start without them; the Modal image bakes them
in (harness/rollout_modal.py `_tau2_image(sandbox=True)`).

Artifacts are judge-shaped: `runs/<tag>/budget_200/results_<split>.json` +
`episodes_<split>/<id>_t<k>.json` full conversation transcripts.

## Answer / action format
The student answers the customer **grounded in the retrieved documents** and
completes the conversation by **calling tools** (retrieval + account actions;
schemas supplied by tau2 at run time). State specific figures, eligibility
rules, and conditions from the knowledge base; do not invent fees, limits, or
policies. The system prompt is `sys.txt`.

## Data
- `brief.md` — acquisition manifest (source + pinned commit) for the hard track,
  plus the Learning Agent split recipe above.
- `dev.json` / `test.json` — scenario id splits (Learning Agent split, above).
