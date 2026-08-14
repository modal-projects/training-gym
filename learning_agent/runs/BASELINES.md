# Agentic-task baselines (dev split)

Measured 2026-08-09/10. Two models per task: the designated student
**Qwen3.5-9B** untrained — the base floor every submission's margin is
measured against — and **GLM-5.2-FP8** (the team endpoint) as a frontier
reference of what a much larger model gets with no task-specific training.

## Reading the numbers

Every cell is **mean reward / pass^4**.

- **Mean reward** — the environment's own verifier score per episode
  (tau2: final DB state + required actions + required communication, gated
  per scenario, partial credit possible; alfworld: 1 if the household task
  was completed, else 0), averaged over all scenarios × trials. "How much
  it gets right on average."
- **pass^4** — fraction of scenarios solved (reward 1.0) on **all four**
  independent trials. Punishes luck: a policy that solves a scenario one
  time in four contributes 0. "How reliably it solves." Always ≤ the mean;
  the gap between them is inconsistency. alfworld runs one trial per game
  (deterministic env), so it has no pass^k — its single number is mean
  reward.

## Results

| task | Qwen3.5-9B (base floor) | GLM-5.2-FP8 (reference) |
|---|---|---|
| tau2_airline | 0.7583 / 0.5667 | 0.7667 / 0.5333 |
| tau2_retail | 0.7973 / 0.5405 | 0.7770 / 0.4865 |
| tau2_telecom | 0.8277 / 0.4865 | 0.9730 / 0.8919 |
| tau2_banking | 0.0776 / 0.0000 | 0.3664 / 0.1552 |
| alfworld | 0.06 (react) · 0.04 (tools) | 0.12 (tools) |

What this says about headroom:

- **banking** — the student floors (it cannot navigate the 698-document
  knowledge base and land the right account actions); even the frontier
  model only reaches 0.37. The widest training margin in the suite.
- **telecom** — the clearest frontier gap among the transactional domains
  (~0.15 mean, ~2x pass^4): real headroom, and the frontier shows it is
  reachable.
- **airline / retail** — the untrained 9B already matches or beats
  GLM-5.2; margins here live in the pass^4 tail, not the mean.
- **alfworld** — hard for everyone under the pinned hidden-admissible /
  40-step protocol: GLM shows the same 27% env-rejected action rate and
  budget exhaustion as the student (Opus 4.8 reference on the react
  driver: 0.92). What training must supply is search strategy.

## Protocol / provenance

- Split: **dev** for every row. tau2: 4 trials, seed 300, max_steps 200,
  customer simulator gpt-5.6-luna (pinned; part of the environment).
  alfworld: 1 trial, 40 steps, admissible actions hidden.
- Qwen floors ran the pinned task sampling (Qwen3.5 thinking recipe) via
  `bench.py rollout --model Qwen/Qwen3.5-9B` on Modal; rows are on
  `LEADERBOARD.jsonl` (tags `tau2_*_qwen35-9b_dev`; airline's is
  `tau2_airline_qwen35-9b_luna_dev`, alfworld's `alfworld_qwen35-9b*_dev`).
- GLM references ran provider-recommended sampling
  (`dev/glm52_baseline.yaml`: temp 1.0, top_p 0.95, no penalties) against
  the served endpoint, `--no-record` per the frontier-reference convention
  (rule 5: GLM can never be a submission). Full artifacts:
  `runs/tau2_*_glm52_dev/`, `runs/alfworld_glm52_dev/`.
- All runs stamped `integrity: DIRTY` (pin drift on the measuring machine —
  the known corpora-"absent" pins issue), env-verified rewards, no LLM
  judge anywhere in the reward path.
- Per-task detail and the same numbers in context: each
  `tasks/<task>/task.md` "Base floor" note.
