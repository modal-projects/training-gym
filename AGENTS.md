# AGENTS.md — working on the Learning Agent repo

Guidance for anyone (human or AI agent) doing **development or operator work on
this repository**.

> This is **not** the benchmark task spec the *evaluated* agent receives — that
> is [`learning_agent.md`](learning_agent_workspace/learning_agent.md). `learning_agent.md` and the rest
> of the pinned integrity surface (see [`README.md`](README.md#integrity)) must
> not be edited without a deliberate re-pin (`python bench.py freeze`).

## Modal environment

All learning-agent benchmark runs land in the shared Modal environment
**`lab-dev`** (workspace `modal-labs`) — so weights, run records, and the viewer
are shared with the whole team. This is driven by one knob,
`MODAL_ENVIRONMENT=lab-dev` in the repo-root `.env`:

- Agent runs and `bench.py train / eval / score / rl` pick it up automatically
  (`agents/run.sh` exports it from `.env`; `bench.py` loads `.env` into the
  process environment), so their `modal` jobs target `lab-dev` regardless of
  your machine's Modal default.
- Your personal Modal default (e.g. `leon-dev`) is intentionally left untouched
  — bare `modal …` commands you type by hand still use it. Pass `-e lab-dev`
  when you want a one-off manual command on the shared env.

The full Modal inventory — apps, volumes, secrets, environments, the deployed
viewer URL, costs — lives in [`dev/MODAL.md`](dev/MODAL.md).
The operator playbook (preflight → launch → score) is in [`launch/`](launch/).
