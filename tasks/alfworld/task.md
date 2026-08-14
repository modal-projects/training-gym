# Task: alfworld (agentic)

Train the student to act in **ALFWorld** — household goals in the TextWorld
text environment (pick & place, heat/cool/clean, examine-in-light, two-object
placements). The first real env task on the agentic path; the measurement
contract it exercises is the one every env task uses (adapter + rollout).

## Environment (pinned)

- `alfworld==0.4.2`, TEXT mode only (no THOR/vision); game data baked into the
  eval image by `alfworld-download` at build (see `harness/rollout_modal.py`).
- Episode: initial observation + goal line; one text command per step;
  `env.max_steps: 40` per episode; sparse reward — **1.0 iff the env reports
  the goal achieved (`won`), else 0.0**. The env, never the model, decides.
- Protocol choice (pinned via task.yaml `env.show_admissible`): **admissible-
  actions list HIDDEN** (`false`, the default). Measured with base
  Qwen3.5-9B: hidden → floor **0.04–0.06**, the 8.6→86 literature regime, so
  the margin gets the full dynamic range; shown → **0.90** on the first 10 dev
  rows, near ceiling with almost no headroom. Flipping this knob changes the
  benchmark — re-freeze deliberately and never compare across the settings.

## Base floors (Qwen3.5-9B, dev, 50 games, admissible hidden)

| driver | score | wins | env-rejected actions | median steps |
|---|---|---|---|---|
| `tools` (default) | **0.04** — CI [0.00, 0.10] | 2/50 | 535/1950 = **27%** | 40 (budget) |
| `react` | **0.06** — CI [0.00, 0.14] | 3/50 | 975/1801 = **54%** | 40 (budget) |

Frontier reference, `react` driver only (the claude CLI cannot emit tool
calls): **Opus 4.8 = 0.92**, 46/50, median 17 steps.

Frontier reference, `tools` driver (the pinned protocol), GLM-5.2-FP8 via the
team endpoint (2026-08-09): **0.12** — CI [0.04, 0.22], 6/50, median steps 40
(budget), env-rejected 27%. Same rejection rate as the base student, same
budget exhaustion: under the hidden-admissible protocol even a frontier model
fails without task-specific search strategy — that headroom is the benchmark.

Reading these: the typed tool schema **halves** the rate of actions the
environment rejects — requiring both arguments on `take`/`put` structurally
prevents a class of malformed actions — yet the success rate is unchanged
(the two CIs overlap almost entirely). Action *formation* was never the
bottleneck; the base model exhausts its 40-step budget exploring, while Opus
finishes in 17. What training has to supply is search strategy and task
planning, not better-formed commands.
- **Action interface: native tool calls** (`agent.driver: tools`). The env's
  verbs are an OpenAI function schema (`ALFWORLD_TOOLS` in
  `harness/adapters/alfworld.py`: `go_to`, `take`, `put`, `open_receptacle`,
  `heat`/`cool`/`clean`, `use`, `examine`, `look`, `inventory`) with typed
  arguments, rendered back to ALFWorld's text commands by the adapter. This is
  the interface deployed agents actually use and the one the student's
  tool-use post-training optimized. **The endpoint must be served with a
  tool-call parser** or every call arrives as prose (the driver detects this
  and says so instead of scoring 0).
- The `react` driver (`ACTION: <command>` per turn,
  `toolbox/agentic_toolbox/react_env_agent.py`) remains available over the
  SAME action space, for text-only policies such as a CLI reference baseline.
  Numbers from the two drivers are not interchangeable — `provenance.driver`
  records which produced a run.
- The submission may replace driver internals; the `build()`/`act()` contract
  must survive (submission/README.md).

## Splits

Seeded samples generated once by
`modal run harness/rollout_modal.py::alfworld_splits` (seed 0) and pinned:

- `dev.json`  — 50 games from upstream **valid_seen** (rooms seen in training
  data; the agent's iteration signal)
- `test.json` — 50 games from upstream **valid_unseen** (unseen rooms; the
  held-out generalization number)

Rows: `{"id", "game_file" (relative to $ALFWORLD_DATA), "task_type"}`.

## Scoring

```bash
python bench.py score --task alfworld --model <weights> --split dev
# = python bench.py rollout ... (env-verified; no LLM judge)
```

Headline = margin over the untrained base floor
(`--model Qwen/Qwen3.5-9B`), as everywhere in Learning Agent. Artifacts are judge-shaped:
`runs/<tag>/budget_40/results_<split>.json` + `episodes_<split>/` transcripts.

## Training signal (the actual task)

Dev-split rollouts are the fitness signal (`bench.py rollout --task alfworld
--split dev --model <ckpt>`). Build training data however you like — expert
traces from the env's own planner data, self-play filtering, paraphrased
trajectories — but every acting model must be the designated student base or a
fine-tune of it (learning_agent.md rule 6).
