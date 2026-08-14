# Task: tau2_airline  (agentic)

## Corpus
`tasks/tau2_airline/corpus/` is a pinned snapshot of Sierra Research's **τ²-bench**
`airline` domain (tau2-bench@`1901a301`):

- `policy.md` (~7.6 KB) — the airline customer-service policy the agent must follow:
  booking, changes, cancellations, baggage, refunds, escalation, identity checks.
- `db.json` (~7 MB) — the domain database the tools read and **write**: flights,
  reservations, and users. This is *live state*, not reference prose — the agent
  acts on it.

The corpus is the environment definition. Ground every action in the policy and in
the current database state (do not invent flights, reservations, or fares).

## What this task is
Unlike the recall tasks (`dspy` / `fav2` / `openclaw` / `maud`), this is a
**procedural** task: success is measured by *doing*, not describing. The student
acts as the airline agent across a **multi-turn conversation** with a simulated
customer, issuing tool calls that read/modify a copy of `db.json`, and is scored
on the resulting database state, the actions it took, and what it communicated —
never on written prose.

## Environment + protocol (pinned)

- `tau2-bench @ 1901a301` (task.yaml `env.pin`), run through its NATIVE
  orchestrator — user simulator, tool loop, and reward are tau2's own
  (`harness/adapters/tau2.py` exposes the batch `run_split` seam; see
  harness/adapters/__init__.py).
- Leaderboard protocol: `max_steps: 200`, `num_trials: 4` (secondary =
  **pass^4**), `seed: 300`; policy sampling = the Qwen3.5 thinking-mode recipe
  (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5).
- Customer simulator: **gpt-5.6-luna** (reasoning_effort low) — SETTLED, and
  the SAME model for your dev rollouts and for official scoring. (This departs
  from τ²-bench's published protocol, which simulates the customer with
  gpt-5.4-mini, so Learning Agent numbers are deliberately not comparable to that public
  leaderboard; Learning Agent scores margin over its own base floor.) The customer is part of the
  environment, so simulating it with something weaker would make your dev
  numbers mean something different from your score. You reach it without
  holding any API key: `LEARNING_AGENT_USER_SIM_URL` in your `.env` points at the
  operator's user-simulator service, which pins the model and budgets each
  session. `provenance.user_llm` records it on every run.
- Reward per episode: tau2's check over final DB state + required actions +
  required communication, gated per scenario. Row score = mean reward over the
  4 trials.

## Splits

Copied verbatim from the pinned repo's shipped
`data/tau2/domains/airline/split_tasks.json`:

- `dev.json`  — its `train` ids (30 scenarios; your iteration signal)
- `test.json` — its `test` ids (20 scenarios; the held-out number)

## Running it

Dev-time (in your workspace — same simulator as scoring, 1 trial instead of 4):

```bash
python bench.py rollout --task tau2_airline --split dev \
    --model /out/models/<tag>/merged --config tasks/tau2_airline/dev.yaml
```

Scoring (operator, pinned simulator + 4 trials):

```bash
python bench.py score --task tau2_airline --model <weights> --split dev
# = bench.py rollout (env-verified; no LLM judge). Headline = margin over the
# untrained base floor (--model Qwen/Qwen3.5-9B), as everywhere in Learning Agent.
```

Base floor on this protocol (Qwen3.5-9B, dev, 30 scenarios x 4 trials, the
settled gpt-5.6-luna simulator): **0.7583** mean reward (CI [0.63, 0.88]),
**pass^4 = 0.5667** — 17/30 scenarios solved on all four trials, 3 never.
Mean reward gives partial credit, so pass^4 is the more discriminating number
to train against.

For reference, the same student under the old gpt-5.4-mini simulator scored
0.7167 / pass^4 0.50. Swapping the simulator moved the floor by roughly one
scenario's worth in each direction (4 of 30 scenarios shifted by >=0.5, one of
them downward) — small, and well inside the CI, but it is why numbers from the
two simulators must not be mixed.

Frontier reference (GLM-5.2-FP8, provider sampling per dev/glm52_baseline.yaml,
2026-08-09): **0.7667** / pass^4 0.5333 — statistically indistinguishable from
the untrained 9B floor. Airline dev rewards policy compliance more than
capability; expect margins here to be small.

Artifacts are judge-shaped: `runs/<tag>/budget_200/results_<split>.json` +
`episodes_<split>/<id>_t<k>.json` full conversation transcripts.

## Answer / action format
The student answers by **calling tools**, not by writing prose: the airline
domain's tool set (schemas supplied by tau2 at run time). Follow the policy
exactly, confirm required details before any write, and refuse actions the
policy prohibits. The system prompt is `sys.txt`.

## Data
- `brief.md` — acquisition manifest (source + pinned commit) for the hard track.
- `dev.json` / `test.json` — scenario id splits (shipped tau2 splits, above).
