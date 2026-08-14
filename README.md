# Learning Agent

Every frontier agent can think and use tools. This project asks whether one can **learn**. Drop
an agent into a domain it has never seen, and it must come back with a model
that has mastered it. Nothing is delegated: the agent acquires the knowledge,
manufactures its own training signal, trains a designated student model,
invents the evaluation it steers by, and engineers the harness its model
answers through. No verifiable reward, no human in the loop. The only number
that counts is the margin: how much better the trained student answers expert
questions than the untrained one.

The benchmark is an autonomy ladder, and every rung removes a crutch (see
[Tracks](#tracks)). On `easy` the agent gets the corpus plus graded dev
questions. On `medium` it gets the corpus alone; if it wants a compass, it
must build one. On `hard` it gets a one-page brief naming the primary sources,
and it goes out and fetches the raw corpus itself. The top of the ladder is a
universal learning agent: aim it at any domain, someday at its own model, and
get back something measurably better.

The judgment never moves. The submission is the agent's own working copy of
this repo, trained student wired into `submission/eval.py`. The operator runs
that file on held-out questions the agent has never seen and scores the
answers with an LLM rubric judge. Learning that does not survive unseen
questions does not count.

Three actors:

1. **Agent under evaluation** — a CLI agent (Claude Code, Codex, ...) given its
   own workspace (the runners seed `learning_agent_workspace/` + the one
   task) and the task spec
   ([learning_agent.md](learning_agent_workspace/learning_agent.md)), one task
   assigned. It generates data, trains by driving the pinned packages in
   `toolbox/training_tool/`, measures itself on the dev split, and wires its
   best system into `submission/eval.py`. Everything in the workspace is its to change;
   the answering models must stay fine-tunes of the designated base, with no
   external LLM APIs in the answer path.
2. **Reference instrument** — `harness/` serves a checkpoint as a search agent
   (ReAct grep/glob/read over the task corpus) and judges answers against
   weighted-claim rubrics with a pinned Anthropic model (temperature 0, forced
   structured output, majority vote). Hash-pinned; the agent's dev-signal
   baseline until its own harness diverges from it.
3. **Operator** — holds the held-out questions and gold (physically absent from
   every workspace), runs `python submission/eval.py --input questions.json
   --output answers.json` in the finished workspace, judges the output with the
   rubric judge, and records the official number in `runs/LEADERBOARD.jsonl`.
   On agentic tasks there is no questions file: `harness/rollout.py` imports
   `submission/agent.py`'s `build()` and drives it against the pinned
   environment, whose verifier sets the reward.
   Dev-vs-test calibration is reported too: with no verifiable reward, building
   a trustworthy eval is part of what is measured.

## Tasks

Two archetypes. `qa`: the student answers held-out questions about the corpus
and the rubric judge scores the answers. `agentic`: the student acts in a
pinned environment and the environment's own verifier sets the reward — no
LLM judge in the reward path, and the secondary metric is pass^k over
`num_trials` episodes per scenario.

| task | archetype | corpus | student answers as |
|---|---|---|---|
| `openclaw` | qa | OpenClaw TypeScript framework (`da228660`) | TypeScript + explanation |
| `fav2` | qa | SEC EDGAR snapshot: 2,415 filings, 41 issuers, 2023–2026 | analyst prose |
| `maud` | qa | 152 MAUD merger agreements | legal prose |
| `alfworld` | agentic | ALFWorld household envs (`alfworld==0.4.2`) | native tool calls, 40-step episodes |
| `tau2_airline` | agentic | τ²-bench airline policy + db (`1901a301`) | tool calls, multi-turn conversation |
| `tau2_retail` | agentic | τ²-bench retail policy + db (`1901a301`) | tool calls, multi-turn conversation |
| `tau2_telecom` | agentic | τ²-bench telecom policies, workflow + dbs (`1901a301`) | tool calls + customer instructions (dual-control) |
| `tau2_banking` | agentic | τ²-bench banking knowledge base, 698 docs + db (`aa74303c`) | grounded answers + account actions via tools |

Each task ships `task.yaml` (the per-task config — a task exists iff this file
exists), `dev.json` (agent-visible), `test.json` (hidden, operator-only),
`sys.txt`, `task.md`, and `brief.md` (the hard-track acquisition manifest)
under `tasks/<task>/`. Corpora are distributed separately (gitignored). The
tau2 tasks run the τ²-bench native orchestrator — user simulator, tool loop,
and reward are the environment's own, pinned by commit in `task.yaml`
(`env.pin`); the customer simulator is one pinned model for dev and scoring
alike, reached keylessly through the operator's user-sim service. Measured
base floors (untrained student) and frontier references for the agentic
tasks: [runs/BASELINES.md](runs/BASELINES.md).

## Tracks

Every run is launched on one of three tracks; each hands the agent less.

| track | corpus | dev gold | the agent must additionally |
|---|---|---|---|
| `easy` | given | given | — |
| `medium` | given | — | author its own dev questions and gold to steer by |
| `hard` | — (only `tasks/<task>/brief.md`, naming the primary sources) | — | acquire and normalize the corpus itself, then everything `medium` requires |

`agents/run_sandbox_modal.sh --track <easy|medium|hard> ...` seeds the workspace
accordingly (default `easy`); on `medium`/`hard` the gold dev sets are
physically absent from the workspace, exactly like `test.json`. Held-out
scoring is identical on every track, so a score gap between tracks prices
exactly the input that was removed.

## Setup

```bash
# python 3.10+, pyyaml; Modal CLI (authenticated) for GPU eval/training
cp .env.example .env      # add OPENAI_API_KEY for the canonical judge
# training packages are materialized at learning_agent_workspace/toolbox/
# training_tool/ (pinned in toolbox/repos.yaml; clone_repos.py fetches them)
```

Keyless machines can judge via the logged-in `claude` CLI
(`--backend cli-claude`); such scores are stamped `canonical: false`.

## Usage

```bash
# score any model/checkpoint: fixed eval -> judge -> leaderboard record
python bench.py score --task fav2 --model <hf-id-or-checkpoint> --split dev --tag mytag

# run an agent against the benchmark: prepares a sandbox copy of the repo
# (held-out test.json physically absent), assembles the prompt, enforces the
# budget, captures the trace, audits for hidden-test access. Agents never run
# in this repo itself — run.sh refuses to launch outside a prepared sandbox.
agents/run_sandbox_modal.sh codex_kimi3 fav2 24                      # Modal container
agents/run_sandbox_modal.sh --track medium codex_kimi3 fav2 24       # see Tracks
agents/run_sandbox_docker.sh codex_kimi3 fav2 24                     # local Docker

# when the run ends, score the submission in the agent's workspace
cd agents/_runs/ws_<...>/workspace
python submission/eval.py --input <held-out questions.json> --output answers.json
python toolbox/eval_tool/rubric_eval.py --dev <held-out gold.json> \
    --answers answers.json --task fav2 --out results.json

# integrity: re-pin after a deliberate benchmark change / verify nothing drifted
python bench.py freeze
python bench.py verify
```

## Integrity

`bench/pins.json` holds sha256 pins of everything that defines the instrument:
harness code, `bench.py`, `bench/config.yaml`, `learning_agent.md`, each task's
`task.yaml`/`sys.txt`/`dev.json`/`test.json`, secondary-metric data files, and
each corpus as a content tree-hash. `score`/`judge` verify pins first and refuse to run on
drift. Every result records provenance (judge model, hashes, seeds, integrity
status) so any two numbers can be checked for comparability. Runs are audited
for hidden-test access (`agents/lib/audit_trace.py`).

## Layout

```
learning_agent_workspace/   the seeded agent surface — what a workspace IS:
  learning_agent.md         task spec template (placeholders: <TASK>, <STUDENT_MODEL>)
  toolbox/                  agent-side tool bank (TOOLS.md is the catalog)
  submission/               the scoring surface the agent rewrites (serve.py, agent.py, eval.py)
  runs/                     the agent's ledgers (seeded empty)
bench.py            operator CLI: score / eval / judge / rollout / freeze / verify / leaderboard
bench/              config.yaml (global pins) + pins.json (integrity lock) + tracks/
harness/            fixed instrument: config loader, eval, judge, judge service, integrity
tasks/<task>/       task.yaml (per-task config), corpus/, dev.json, test.json (hidden),
                    sys.txt, task.md, ROUNDS.md — the ONE task is injected at seed time
agents/             contestant runner (run_sandbox.sh -> run.sh) + one scaffold dir per CLI agent
runs/               operator ledgers: LEADERBOARD.jsonl, per-tag results
```

The line that matters: what the operator holds — the held-out questions and gold,
the external rubric judge, and the `submission/eval.py` I/O contract — is the
measurement. Everything inside the agent's workspace is its to change. Pins keep
the template and the reference instrument versioned so dev scores stay comparable
across runs. See [learning_agent_workspace/toolbox/TOOLS.md](learning_agent_workspace/toolbox/TOOLS.md).

## Where things live

The project is local-first: an agent's workspace and everything it produces stay on
the operator's machine unless deliberately pushed to Modal. Team sharing is
one knob, not a different workflow — point `MODAL_ENVIRONMENT` (see
`.env.example`) at a shared Modal environment instead of a personal one.

| what | local (default) | shared with the team |
|---|---|---|
| agent run workspaces | `agents/_runs/`, disposable, never shared directly | the ingested run record (row below) |
| containerized agent sessions | Docker: `agents/_container_runs/<task>/<session>/{workspace,logs}` | Modal volume `lab-agent-workspace`: `<task>/<session>/{workspace,logs}` (app `lab-agent`, one container per session) |
| trained weights | Modal volume `lab-out` at `/out/models/<tag>/merged` — already shared infrastructure | same |
| run records (traces/results/telemetry) | a local directory via the observatory's `--data-dir` | Modal volume `lab-observatory` |
| viewer | `python3 observatory/app.py --data-dir <dir>` | `modal deploy observatory/app.py` (team URL) |

The agent itself can execute in three places, all sharing one launch shape and
one seeding routine (`workspace_setup/prepare_workspace.sh`): on the operator
machine (`agents/run_sandbox.sh`), in a container under the Modal app
`lab-agent` (`agents/run_sandbox_modal.sh` — workspace and CLI session logs
persist on `lab-agent-workspace`; containers carry a Modal token so
`bench.py train` and the live watcher work inside), or in local Docker
(`agents/run_sandbox_docker.sh`).

Per-run learning activity (data/train/eval/evolve actions) and the track
(easy/medium/hard) an agent ran under are visible in the observatory — see
[observatory/README.md](observatory/README.md).

### Team runs: one launch, everything shared

> **Our team's shared target is the `lab-dev` Modal environment** (workspace
> `modal-labs`) — already the `.env` default and usable by everyone on the team
> (Modal environments are visible to all workspace members). Its viewer is live
> at https://modal-labs-lab-dev--lab-observatory-web.modal.run. Account-specific
> Modal details — environments, volumes, secrets, deployed URLs — live in
> [`dev/MODAL.md`](dev/MODAL.md).

Once per team: create (or pick) a shared Modal environment, put its name in
every member's `.env` as `MODAL_ENVIRONMENT=<team-env>`, and deploy the viewer
there once (`modal deploy observatory/app.py`). Then a shared run is:

```bash
agents/run_sandbox.sh --watch --track easy claude_reprompt fav2 24
```

`--watch` starts a detached observatory watcher next to the run. While the
agent works, teammates follow it live at the deployed viewer URL; when it
finishes, the run record carries the full trace, judge results, learning
timeline, and `raw/workspace.tar.gz` — the submission folder itself (minus
corpus/venvs), downloadable from the run page's raw artifacts.

Where each artifact lands, all in the shared environment: weights on
`lab-out` (the agent's training jobs inherit `MODAL_ENVIRONMENT` from `.env`
— `agents/run.sh` exports it at launch), trace + results + workspace archive
on `lab-observatory`, and the viewer reads that volume. Nothing else to wire.
```bash
# pull a teammate's finished workspace from the volume, score it locally
modal volume get lab-observatory runs/<run_id>/raw/workspace.tar.gz .
mkdir ws && tar -xzf workspace.tar.gz -C ws && cd ws   # then the submission contract
```
