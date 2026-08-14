# Learning Agent Observatory — design

One place to see every experiment: agent-run traces, judge results, workspace
snapshots, and GPU/system telemetry, modeled on posttrainbench.com/traces.
Read-only over the benchmark (never writes into pinned surfaces); writes only
to this directory and the Modal volume.

## Topology

```
local machine                          Modal (env: junlin-dev)
─────────────                          ───────────────────────
agents/_runs/ws_*/...   ──obs ingest──▶  Volume lab-observatory
runs/<tag>/budget_*/... ──obs watch───▶    runs/<run_id>/record.json
                                           runs/<run_id>/workspace.json
                                           runs/<run_id>/status.json
                                           runs/<run_id>/raw/…
                                                 │
                                       app lab-observatory (FastAPI, public URL)
                                           GET /            index (heatmap + table)
                                           GET /run?id=…    run view
```

- `obs ingest <dir>` — normalize one finished run dir and upload.
- `obs watch <dir>` — live mode: re-ingest every N seconds while the run is
  alive (trace.jsonl still growing / solve_status.txt absent), sampling local
  CPU/mem into `system_monitor`; finalize (workspace snapshot + state=finished)
  when the run ends.
- `observatory/sampler.py` — drop-in GPU sampler for Modal jobs
  (`start_sampler(path)` daemon thread → `system_monitor.jsonl`).
- The web app only reads the volume. `vol.reload()` keeps it fresh.

## Volume layout (`lab-observatory`)

```
runs/<run_id>/
  record.json           # RunRecord (schema.py) — everything the run view needs
  workspace.json        # WorkspaceSnapshot — file tree, small text inlined
  status.json           # Status — {state, updated_at, num_events, …} tiny, polled
  raw/                  # verbatim source artifacts (trace.jsonl, audit.json,
                        #   prompt.txt, solve.err, solve_status.txt, …)
```

`run_id` = run-dir basename (e.g. `claude_dspy_20260717T120301`), the unique
key everywhere. No global index file: the app lists `runs/*/` and reads each
`record.json`'s `index_row` (cached per mtime) — correct under concurrent
writers at research scale.

## Run record (see schema.py for field-level truth)

posttrainbench-compatible top level, extended for Learning Agent:

- `index_row` — flat row for the index page (task, scaffold, model, state,
  best_dev_score, canonical, integrity, audit, cost, turns, …).
- `meta` — identity: run_id, scaffold, task, base_model, trace_format,
  time_budget_h, launched_at/finished_at, exit_code.
- `summary` — agent-level: models seen, tools, num_turns, duration_ms,
  total_cost_usd, usage_total, stop_reasons, final_result_text, sessions.
- `sessions[]` — one per session init.
- `events[]` — normalized trace: `{i, ts, type, subtype, session_id,
  session_idx, parent_tool_use_id, turn, blocks[], usage, model}` with content
  hoisted into four block shapes (thinking / text / tool_use / tool_result —
  see schema.py Event docstring). Every scaffold format maps into these:
  - claude stream-json: message.content blocks hoisted verbatim.
  - codex: `agent_reasoning`→thinking, `agent_message`→text,
    `exec_command_begin/end`→tool_use/tool_result pair, `token_count`→usage.
  - gemini / opencode: best effort via the same mapping; unknown events pass
    through as `{type:"system", subtype:"unknown"}` with trimmed raw preserved.
  `turn` increments on each assistant event that starts a new assistant block
  after a user/system event; used for the jump-to-turn box.
  Timestamps: Learning Agent pipes solve.sh stdout raw, so trace lines carry none. The
  live watcher records arrival times per line into a sidecar at
  `<run_dir>/.obs/line_ts.jsonl` (`{"line": <1-based>, "ts": ISO8601}`),
  uploaded as `raw/line_ts.jsonl`; the normalizer joins on line number.
  Post-hoc ingests of old runs get `ts: null` (frontend falls back to event
  index for the timeline). The watcher also drops its local CPU/mem samples
  in `<run_dir>/.obs/system_monitor.jsonl`.
- `scores` — Learning Agent-specific: `checkpoints[]` (runs/CHECKPOINTS.jsonl rows),
  `leaderboard[]` (runs/LEADERBOARD.jsonl rows for tags this run produced),
  `results[]` (each runs/<tag>/budget_<b>/results_<split>.json found in the
  run's workspace, with per_question verdicts/votes kept for drill-down, plus
  eval_meta tool_calls/completion_tokens joined in when present).
  NEVER coerce `claim_score: null` / failed questions to 0.
- `judgements` — `audit` = audit.json verbatim (integrity CLEAN/CONTAMINATED,
  findings, access_counts, behavior).
- `system_monitor[]` — samples `{ts, gpu:{util_pct, mem_used_mib, mem_total_mib,
  temp_c, power_w}|null, cpu_load_1m, mem_used_gib, mem_total_gib, …}`.
  Local watcher samples CPU/mem; Modal sampler adds GPU.

Provenance is first-class in the UI: `canonical:false`, `integrity:DIRTY`,
audit `CONTAMINATED`, and judge-pin SHAs must be visible badges, not footnotes.

## Web app

`observatory/app.py`, Modal app name `lab-observatory`, deployed with
`modal deploy observatory/app.py` (env from `MODAL_ENVIRONMENT`). Public URL
(decision: no auth in v1). FastAPI + static files baked into the image.

API (all JSON; 404 on unknown run):

```
GET /api/runs                       → [index_row]  (sorted newest first)
GET /api/runs/{run_id}              → record.json
GET /api/runs/{run_id}/status       → status.json  (light poll for live view)
GET /api/runs/{run_id}/workspace    → workspace.json
GET /api/runs/{run_id}/raw/{name}   → raw artifact (text/plain)
GET /healthz                        → {ok: true}
GET /                               → static index.html
GET /run                            → static run.html (?id=<run_id>)
```

Frontend (`observatory/static/`): vanilla JS + Chart.js, posttrainbench-style
three-zone run page — left overview rail / center tabs (Trace · Judge ·
Scores · Workspace) / right system-metrics rail with expand-all modal.
Index page: task × scaffold heatmap (best dev score) + filterable runs table
(state, task, scaffold, canonical/integrity/audit badges). Live runs poll
`/status` and re-fetch the record when `updated_at` changes.

## Env (.env at repo root; see .env.example)

```
MODAL_ENVIRONMENT=junlin-dev      # modal env for volume + app
MODAL_OBS_VOLUME=lab-observatory  # volume holding all observability data
OBS_DATA_DIR=                     # local dir override for app dev (skip volume)
```

## File ownership (build phase)

- schema.py, DESIGN.md, fixtures/ — contracts (hand-written first).
- normalize/, cli.py, volume_io.py, local_monitor.py — ingestion builder.
- static/ — frontend builder.
- app.py, sampler.py, README.md, ../.env.example — app builder.

## Non-goals in v1

Experiment registry / launching runs from the UI (v2), auth, ingesting the
ReAct eval rollout transcripts (harness discards them today — needs a
deliberate harness change + re-freeze; see repo map).
