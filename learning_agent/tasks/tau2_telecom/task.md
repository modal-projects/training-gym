# Task: tau2_telecom  (agentic — dual-control)

## Corpus
`tasks/tau2_telecom/corpus/` is a pinned snapshot of Sierra Research's **τ²-bench**
`telecom` domain (tau2-bench@`1901a301`). This is the flagship **dual-control**
domain: the customer can also act on their own device/line, so the agent must
coordinate and re-check state, not just issue writes.

- `main_policy.md` (~5.7 KB) — the primary agent policy (dual-control variant).
- `tech_support_workflow.md` (~16 KB) — the step-by-step troubleshooting workflow.
- `tech_support_manual.md` (~18 KB) — the tech-support knowledge base (device/line
  behavior, error conditions, fixes).
- `db.toml` (~9.6 KB) — the agent-side domain database (accounts, lines, devices).
- `user_db.toml` (~0.9 KB) — the user-side state the customer controls (dual-control).
- `main_policy_solo.md`, `tech_support_workflow_solo.md` — the *solo*-mode variants
  (no user agent); kept for fidelity, unused by the pinned configuration.

Note: telecom's DB is **TOML**, not JSON. Ground every action in the policy,
workflow, manual, and current database state (do not invent plans, device
behavior, or steps).

## What this task is
A **procedural** task in a **dual-control** setting: the student acts as the
telecom support agent across a **multi-turn conversation**, both issuing tool
calls against a copy of `db.toml` and directing the simulated customer to
perform steps that change `user_db.toml`. Scored on resulting state + required
actions + communication, never on prose.

## Environment + protocol (pinned)

- `tau2-bench @ 1901a301` (task.yaml `env.pin`), run through its NATIVE
  orchestrator — user simulator (with user-side tools), tool loop, and reward
  are tau2's own (`harness/adapters/tau2.py` exposes the batch `run_split` seam).
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
`data/tau2/domains/telecom/split_tasks.json`:

- `dev.json`  — its `train` ids (74 scenarios; your iteration signal)
- `test.json` — its `test` ids (40 scenarios; the held-out number)

(The repo also ships `small`/`full` variants — 20 and 2285 scenarios — which Learning Agent
does not use.)

## Running it

Dev-time (in your workspace — same simulator as scoring, 1 trial instead of 4):

```bash
python bench.py rollout --task tau2_telecom --split dev \
    --model /out/models/<tag>/merged --config tasks/tau2_telecom/dev.yaml
```

Scoring (operator, pinned simulator + 4 trials):

```bash
python bench.py score --task tau2_telecom --model <weights> --split dev
# = bench.py rollout (env-verified; no LLM judge). Headline = margin over the
# untrained base floor (--model Qwen/Qwen3.5-9B), as everywhere in Learning Agent.
```

Base floor on this protocol (Qwen3.5-9B, dev, 74 scenarios x 4 trials,
gpt-5.6-luna simulator, 2026-08-09): **0.8277** mean reward
(CI [0.78, 0.87]), **pass^4 = 0.4865**. Frontier reference (GLM-5.2-FP8,
provider sampling per dev/glm52_baseline.yaml): 0.9730 / pass^4 0.8919 —
near ceiling. Telecom has the clearest frontier-vs-student gap of the
transactional domains (~0.15 mean, ~2x pass^4): real headroom for training.

Artifacts are judge-shaped: `runs/<tag>/budget_200/results_<split>.json` +
`episodes_<split>/<id>_t<k>.json` full conversation transcripts.

## Answer / action format
The student answers by **calling tools** and by **instructing the customer** —
not by writing a report. Tool schemas are supplied by tau2 at run time. Follow
the policy and workflow exactly; verify identity/details before any write;
decline actions the policy prohibits. The system prompt is `sys.txt`.

## Data
- `brief.md` — acquisition manifest (source + pinned commit) for the hard track.
- `dev.json` / `test.json` — scenario id splits (shipped tau2 splits, above).
