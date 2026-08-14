# Task: tau2_retail  (agentic)

## Corpus
`tasks/tau2_retail/corpus/` is a pinned snapshot of Sierra Research's **τ²-bench**
`retail` domain (tau2-bench@`1901a301`):

- `policy.md` (~6.7 KB) — the retail customer-service policy: orders, cancellations,
  returns, exchanges, refunds, address/payment changes, identity checks.
- `db.json` (~2.7 MB) — the domain database the tools read and **write**: users,
  orders, products, and their variants/inventory. Live state, not reference prose.

Retail is the most tool/DB-heavy τ² domain. Ground every action in the policy and in
the current database state (do not invent orders, items, prices, or fees).

## What this task is
A **procedural** task: success is *doing*, not describing. The student acts as the
retail agent across a **multi-turn conversation** with a simulated customer, issuing
tool calls that read/modify a copy of `db.json`, and is scored on the resulting
database state, the actions it took, and what it communicated — never on prose.

## Environment + protocol (pinned)

- `tau2-bench @ 1901a301` (task.yaml `env.pin`), run through its NATIVE
  orchestrator — user simulator, tool loop, and reward are tau2's own
  (`harness/adapters/tau2.py` exposes the batch `run_split` seam).
- Leaderboard protocol: `max_steps: 200`, `num_trials: 4` (secondary =
  **pass^4**), `seed: 300`; policy sampling = the Qwen3.5 thinking-mode recipe
  (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5).
- Customer simulator: **gpt-5.6-luna** (reasoning_effort low) — the SAME model
  for dev rollouts and official scoring (see tasks/tau2_airline/task.md for why,
  and why Learning Agent numbers are not comparable to the public τ² leaderboard). Reached
  keylessly via `LEARNING_AGENT_USER_SIM_URL`; `provenance.user_llm` records it per run.
- Reward per episode: tau2's check over final DB state + required actions +
  required communication, gated per scenario. Row score = mean reward over the
  4 trials.

## Splits

Copied verbatim from the pinned repo's shipped
`data/tau2/domains/retail/split_tasks.json`:

- `dev.json`  — its `train` ids (74 scenarios; your iteration signal)
- `test.json` — its `test` ids (40 scenarios; the held-out number)

(Most retail scenarios ship without a `purpose` description — those rows carry
`purpose: null`.)

## Running it

Dev-time (in your workspace — same simulator as scoring, 1 trial instead of 4):

```bash
python bench.py rollout --task tau2_retail --split dev \
    --model /out/models/<tag>/merged --config tasks/tau2_retail/dev.yaml
```

Scoring (operator, pinned simulator + 4 trials):

```bash
python bench.py score --task tau2_retail --model <weights> --split dev
# = bench.py rollout (env-verified; no LLM judge). Headline = margin over the
# untrained base floor (--model Qwen/Qwen3.5-9B), as everywhere in Learning Agent.
```

Base floor on this protocol (Qwen3.5-9B, dev, 74 scenarios x 4 trials,
gpt-5.6-luna simulator, 2026-08-09): **0.7973** mean reward
(CI [0.73, 0.86]), **pass^4 = 0.5405**. Frontier reference (GLM-5.2-FP8,
provider sampling per dev/glm52_baseline.yaml): 0.7770 / pass^4 0.4865 — the
untrained student already matches the frontier model here, so margins on
retail will be earned in the pass^4 tail, not the mean.

Artifacts are judge-shaped: `runs/<tag>/budget_200/results_<split>.json` +
`episodes_<split>/<id>_t<k>.json` full conversation transcripts.

## Answer / action format
The student answers by **calling tools**, not by writing prose: the retail
domain's tool set (schemas supplied by tau2 at run time). Follow the policy
exactly, confirm required details before any write, and refuse actions the
policy prohibits. The system prompt is `sys.txt`.

## Data
- `brief.md` — acquisition manifest (source + pinned commit) for the hard track.
- `dev.json` / `test.json` — scenario id splits (shipped tau2 splits, above).
