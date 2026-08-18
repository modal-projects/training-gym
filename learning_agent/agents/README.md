# `agents/` — the contestant runner layer

This is how Learning Agent puts a CLI coding agent through a scored run and produces an auditable
record. It is modeled on PostTrainBench's `run_task.sh` + `agents/` scaffold matrix,
adapted to Learning Agent.

## Run one

Two containerized runners share one shape (`[--config <yaml>] [--track ...]
<scaffold> <task> [hours] [model]`), one seeding routine
(`workspace_setup/prepare_workspace.sh`), and one in-container entry script
(`lib/container_entry.sh` — points `HOME` inside the session's `logs/` so
CLI-native session state persists, starts the observatory watcher from a
read-only seed copy, then runs `run.sh`):

```bash
agents/run_sandbox_modal.sh  modal_glm52 fav2 23.5   # container under Modal app `lab-agent`
agents/run_sandbox_docker.sh modal_glm52 fav2 24     # local Docker container

agents/run_sandbox_modal.sh --config task_configs/fav2.yaml   # the task's own session defaults
```

The runners prepare the agent's own sandbox copy of the repo — committed tree
only, every `tasks/*/test.json` physically absent, corpus/dev seeded, fresh git
history — then run `run.sh` inside it. The finished workspace is the submission:
score it with `python submission/eval.py --input <held-out questions> --output
answers.json` from the workspace root.

Agents never run in the seed repo. The agent edits code (data, training, harness,
submission), so a launch here would let it modify the benchmark itself; `run.sh`
(the inner primitive: prompt, timer, trace, audit) refuses to start unless the
`.learning_agent_sandbox` marker written at seeding is present
(`LEARNING_AGENT_ALLOW_IN_PLACE=1` overrides, for operator smoke tests only).

Modal sessions live on the `lab-agent-workspace` volume as
`<task>/<session>/{workspace,logs}` (session = `<scaffold>_<student>_<stamp>`);
the whole session — task, corpus, and materialized training packages riding
inside the workspace — uploads per session (no shared state between sessions).
Containers carry the `lab-agent-modal-token` secret so
the agent's own `bench.py train`/`score` jobs and live dashboard ingest work
from inside. Docker sessions mirror the same layout under
`agents/_container_runs/`. Live observability is always on for container runs —
no `--watch` flag. Modal budgets are clamped to 23.75 h (platform timeout
ceiling is 24 h and run.sh's grace kill must fire first).

## The scaffold matrix

Ported 1:1 from PostTrainBench's `agents/` (adapted where PTB assumes a per-run
container; Learning Agent's sandboxes share the host, so effort/auth overrides are isolated
per scaffold and never touch the operator's real `~/.codex` or `~/.claude` config):

| scaffold | CLI | notes |
|---|---|---|
| `claude` | Claude Code | API key (`ANTHROPIC_API_KEY`) |
| `claude_reprompt` | Claude Code | + re-prompt loop: resumes the session with time remaining until <30 min left |
| `claude_non_api` | Claude Code | subscription OAuth token, effort=high |
| `claude_non_api_max` | Claude Code | subscription OAuth token, effort=max (Opus 4.6+) |
| `codex` | Codex CLI | API key (`OPENAI_API_KEY`) |
| `codexlow` / `codexhigh` / `codex_xhigh` | Codex CLI | reasoning-effort ablations (`-c model_reasoning_effort=…`) |
| `codex_xhigh_reprompt` | Codex CLI | effort=xhigh + re-prompt loop (`codex exec resume --last`) |
| `codex_non_api` (+ `_high`, `_xhigh`, `_reprompt`, `_high_reprompt`, `_xhigh_reprompt`) | Codex CLI | ChatGPT Pro subscription auth via isolated `CODEX_HOME` |
| `gemini` | Gemini CLI | `GEMINI_API_KEY` |
| `opencode` | OpenCode | multi-provider (Anthropic/OpenAI/Z.AI) — runs GLM-5 etc. |
| `modal_glm52` | OpenCode → Modal | team's own SGLang endpoint in `lab-dev` (GLM-5.2-FP8, 1M ctx, public/unauthenticated — see `dev/MODAL.md`); no API key needed |
| `glm5` | Claude Code → Z.AI | Anthropic-compatible endpoint, needs `ZAI_API_KEY` (Coding Plan) |
| `qwen3max` | Claude Code → DashScope | Anthropic-compatible endpoint, needs `DASHSCOPE_API_KEY` |

### Subscription-auth setup (non-API scaffolds)

Credentials live in the scaffold's own directory and are gitignored.

```bash
# Claude Max subscription
claude setup-token                                   # browser prompt
echo "sk-ant-..." > agents/claude_non_api/oauth_token       # (or claude_non_api_max)

# ChatGPT Pro subscription
codex login --device-auth                            # browser prompt
cp ~/.codex/auth.json agents/codex_non_api/          # (and any codex_non_api_* dir you use)
```

The re-prompt variants counter the dominant early-quit failure mode: when the CLI
session ends with budget left, the scaffold reads `timer.sh` and resumes the same
session with "you still have Xh Ym remaining" until fewer than 30 minutes remain.
The hard wall-clock kill in `run.sh` still bounds the whole thing.

The runner:

1. **Assembles the prompt** — `AGENTS.md` with `<TASK>` resolved, plus a short preamble
   stating the assignment, the working directory, the budget, and the stop condition.
2. **Writes `timer.sh`** into the repo root (the agent runs `bash timer.sh` to see time
   left) and sets a hard wall-clock kill 5 minutes past budget.
3. **Runs the scaffold** (`<scaffold>/solve.sh`), capturing the full stream-json trace.
4. **Parses the trace** to a human-readable transcript (`<scaffold>/human_readable_trace.py`).
5. **Audits the trace** deterministically for hidden-test access (see below).
6. **Diffs `runs/CHECKPOINTS.jsonl`** to record exactly what the agent submitted.

All artifacts land under `agents/_runs/<scaffold>_<task>_<stamp>/` (gitignored):
`prompt.txt`, `trace.jsonl`, `trace.txt`, `audit.json`, `submitted.jsonl`,
`solve_status.txt`.

The runner exits non-zero if the audit finds contamination, so CI can gate on it.

## The sandbox

Each run gets a fresh copy of the repo under `agents/_runs/ws_*/workspace/`,
prepared by the runners:

- `git archive` of HEAD — the sandbox has no git history to mine, and the agent's
  own `git init` starts at run start.
- `tasks/*/test.json` never enter the sandbox: the held-out data is physically
  absent, not merely forbidden.
- The gitignored inputs the agent needs are seeded: the task corpus (copy-on-write
  clone where the filesystem supports it), `dev.json`, `.env`.
- `toolbox/training/_vendor (in-workspace)` is symlinked beside the sandbox so `bench.py train`/`rl`
  work unchanged; Modal auth and the HF cache are host-level and unaffected.

## The audit — deterministic, not an LLM judge

PTB needs an LLM "contamination judge" because its forbidden thing ("did you train on
GSM8K?") is fuzzy. Learning Agent's forbidden thing is exact — the hidden test set is one known file
— so `lib/audit_trace.py` is a deterministic scan that **cannot be talked out of a
finding**. It flags:

- any reference to `tasks/<task>/test.json` in the trace,
- any verbatim hidden-test question string appearing in the trace,
- any `--split test` invocation (hidden-split scoring is operator-only).

Any hit ⇒ `integrity: CONTAMINATED` and a non-zero exit. It also emits a descriptive
behavior summary (corpus tool calls, train/rl/score launches, checkpoint registrations)
— useful signal, not pass/fail.

## Adding a scaffold

A scaffold is a directory with two files:

- **`solve.sh`** — launches the CLI in headless/streaming mode, reading `$PROMPT` and
  `$AGENT_CONFIG` (the model) from the environment, streaming events to stdout.
- **`human_readable_trace.py`** — converts that CLI's stream format to a transcript.
  (The `codex` parser auto-detects and copies through verbatim when the format is
  unknown, so it is a safe default to copy for a new CLI.)

Variants are just more directories — `claude`, `claude_reprompt`, `codex_xhigh`, … —
each comparable because only `solve.sh` differs; the prompt, task, and instrument are
identical across them.

## After a run — operator scoring

The agent only self-scores on **dev**. The official number comes from running the
submission in the finished sandbox against the held-out questions, then judging
the answers externally:

```bash
cd agents/_runs/ws_<...>/workspace
python submission/eval.py --input <held-out questions.json> --output answers.json
python toolbox/eval_tool/rubric_eval.py --dev <held-out gold.json> \
    --answers answers.json --task <task> --out results.json
```
