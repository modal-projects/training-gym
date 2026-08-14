# Learning Agent Observatory

One place to see every Learning Agent experiment — agent-run traces, judge results,
workspace snapshots, and GPU/system telemetry — modeled on
posttrainbench.com/traces. Local runs are normalized and uploaded to a Modal
volume; a small FastAPI app (also on Modal) renders them read-only. The
observatory never writes into pinned benchmark surfaces. Architecture, volume
layout, and field-level contracts live in [DESIGN.md](DESIGN.md) and
[schema.py](schema.py).

Local-first by default: run the viewer against a local `--data-dir` for solo
work. Team sharing is just pointing `MODAL_ENVIRONMENT` at a shared Modal
environment instead of a personal one — see [Where things
live](../README.md#where-things-live) in the root README.

## Quickstart

```bash
cp .env.example .env    # once, at the repo root

uv venv observatory/.venv && uv pip install -r observatory/requirements.txt
# (or plain pip: python3 -m venv observatory/.venv && observatory/.venv/bin/pip install -r observatory/requirements.txt)

# ingest a finished run (--archive-workspace adds raw/workspace.tar.gz —
# the full submission folder minus corpus/.git/venvs; --data-dir D also
# stages into D/runs/<id> for a local `app.py --data-dir D` viewer)
python3 -m observatory.cli ingest agents/_runs/ws_<...> --archive-workspace

# live-watch a running one (the runners start their own watcher)
python3 -m observatory.cli watch agents/_runs/ws_<...> --interval 20

# meter GPU-hours from Modal sandbox history and overlay them per run
# (control-plane truth vs the agent's self-reported GPU_LOG.jsonl;
#  `gpu hours (metered)` on the run page, tooltip on the index column)
python3 -m observatory.gpu_metering            # add --no-upload to inspect

# deploy the viewer
MODAL_ENVIRONMENT=junlin-dev modal deploy observatory/app.py

# local dev against a staged dir (no Modal round-trips)
python3 observatory/app.py --data-dir /tmp/obs_stage --port 8900
```

Env (repo-root `.env`, see `.env.example`): `MODAL_ENVIRONMENT`,
`MODAL_OBS_VOLUME`, and optional `OBS_DATA_DIR` as the default `--data-dir`
for local dev.

## Volume layout (`lab-observatory`)

```
runs/<run_id>/
  record.json           # RunRecord (schema.py) — everything the run view needs
  workspace.json        # WorkspaceSnapshot — file tree, small text inlined
  status.json           # Status — {state, updated_at, num_events, …} tiny, polled
  raw/                  # verbatim source artifacts (trace.jsonl, audit.json,
                        #   prompt.txt, solve.err, solve_status.txt, …)
```

`run_id` = run-dir basename. No global index file: the app lists `runs/*/`
and reads each `record.json`'s `index_row` (cached per mtime).

## API

| Route | Returns |
| --- | --- |
| `GET /api/runs` | `[index_row]`, status.json overlay, newest first |
| `GET /api/runs/{run_id}` | `record.json` verbatim |
| `GET /api/runs/{run_id}/status` | `status.json` (light poll for live view) |
| `GET /api/runs/{run_id}/workspace` | `workspace.json` |
| `GET /api/runs/{run_id}/raw/{name}` | raw artifact, `text/plain` |
| `GET /healthz` | `{"ok": true}` |
| `GET /` | index page (heatmap + runs table) |
| `GET /how` | the end-to-end benchmark walkthrough (stages, schematics, smoke-test commands) — same page on the Modal deployment and the local server, so the whole team sees it at the deployed URL |
| `GET /run?id=<run_id>` | run view |

All JSON; 404 `{detail}` on unknown run. Unknown/invalid `run_id` and `name`
are rejected (charset `[A-Za-z0-9._-]+`, no `..`).

## GPU sampler (opt-in)

`observatory/sampler.py` is a stdlib-only drop-in for any Modal GPU job —
three lines inside the function:

```python
from observatory.sampler import start_sampler
stop = start_sampler("/out/obs/<tag>/system_monitor.jsonl")  # daemon thread
# ... training ... stop.set() at the end (optional; daemon dies with the job)
```

Samples land on the job's volume only after `volume.commit()`; for long jobs
commit periodically, or accept Modal's function-exit commit.

The observatory does NOT modify `pipeline/` or `harness/`. Wiring the sampler
into `pipeline/train.py` is a user decision, not something this package does.
`harness/` is pin-locked — changing it requires a deliberate `bench.py freeze`.

## Fixtures

`fixtures/make_demo_run.py` regenerates the demo run dir
(`fixtures/demo/ws_claude_dspy_20260717T090000/`) plus the already-normalized
`sample_record.json` / `sample_workspace.json` / `sample_status.json`. The
demo trace content is sample data from a public PostTrainBench trajectory.

## Security

The deployed URL is public and unauthenticated — by decision, v1. fav2 golds
and rubrics are proprietary: think before ingesting fav2 runs whose
workspaces embed gold answers, because everything on the volume becomes
world-readable through the viewer. To flip to workspace-only access, set
`requires_proxy_auth=True` on the `web` function in `app.py` and redeploy.

Workspace snapshots always exclude runtime dotenv files (`.env`, `.env.*`,
and `.envrc`); documentation-only templates such as `.env.example` remain
visible. Raw traces are uploaded verbatim, so an agent that explicitly prints
a secret can still put it in the trace. Do not print credentials, and use
authenticated deployment or trace redaction when ingesting untrusted runs.

## Dependencies

`requirements.txt` pulls `modal`, `fastapi[standard]`, `uvicorn` for the
viewer + deploy only. `normalize/`, `cli.py`, and `sampler.py` stay
stdlib-only.
