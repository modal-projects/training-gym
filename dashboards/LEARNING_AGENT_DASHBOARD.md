# Plan: learning-agent view in the Training Gym dashboard

Experiment branch: `lab-dashboard-experiment` in this fork
(`dashboard_experiment/training-gym`). Nothing under `learning_agent_bench/`
is modified by this plan; the observatory and its ingestion pipeline keep
working as they are today.

## Status (2026-08-13): BUILT AND DEPLOYED

Implemented as planned, deployed to Modal env `lab-dev` as app
`lab-learning-dashboard`. What exists:

- `modal_training_gym/_learning_agent_routes.py` — `GET /api/learning-runs` +
  `GET /api/learning-runs/{run_id}` reading the `lab-observatory` volume.
  One recursive `iterdir` per refresh (VolumeListFiles is rate-limited —
  never list per run), index_row/status cached per file mtime, 15s list TTL.
  Detail response drops `events`/`system_monitor` (~850 KB → ~17 KB) and
  strips `per_question` from results.
- `_dashboard.py` — two small diffs: `register_learning_agent_routes(web)` and
  LEARNING_AGENT_OBS_VOLUME/LEARNING_AGENT_OBS_URL env passthrough into the image.
- `common/dashboard.py` — `DASHBOARD_APP_NAME` overridable via
  `TRAINING_GYM_DASHBOARD_APP_NAME` (deploy-time collision avoidance).
- Frontend — nav item "Learning agent" (FlaskConical), `/learning` list page
  (summary cards + searchable runs table), `/learning/<run_id>` detail page
  (research-log timeline with checkpoint/submission/note pills, what/why,
  artifact chips, dev-score LineChart, eval-results table, deep-link to the
  observatory run view, links to gym training runs when a checkpoint's
  tag/path matches). `lib/learning.js` holds shared helpers.
- Observatory-parity panels — the run page is a 3-column grid mirroring the
  observatory run view: left rail (overview + results), center (score chart,
  research log, trace, workspace, evals), right rail (system telemetry).
  `GET .../monitor` serves record.json's system_monitor downsampled to ~400
  samples (cpu load, mem; gpu when present); cumulative gpu-hours chart
  derives from scores.gpu_log. `GET .../workspace` serves the snapshot tree
  with contents STRIPPED (raw workspace.json is ~27 MB with inlined file
  bodies) and `GET .../workspace/file?path=` returns one file's content;
  both come from a single-slot parsed cache keyed by workspace.json mtime.
  The workspace UI is a filterable tree + file viewer.
- ATIF export — `GET /api/learning-runs/{run_id}/trajectory[?download=true]`
  converts the full trace to a Harbor ATIF-v1.7 trajectory
  (harborframework.com/docs/agents/trajectory-format) via
  `modal_training_gym/_learning_agent_atif.py`: assistant events → agent steps
  (message/reasoning_content/tool_calls/metrics), tool-result user events →
  observations folded onto the calling step (resolved by (session_idx,
  call_id) first — codex resumes reuse item ids across sessions), text user
  events → user steps, system/result events → system steps; step ids
  renumbered sequentially per the ATIF validator. "Download ATIF trajectory"
  button on the trace section serves it as
  `<run_id>.trajectory.json`.
- Agent trace — `GET /api/learning-runs/{run_id}/events?offset=&limit=`
  pages the trace (`events` in record.json; tail by default, ≤500/page,
  single-slot parsed cache keyed by record mtime). The detail page renders
  it in a collapsed "Agent trace" section: tool_use command blocks,
  expandable tool_result outputs (20k char cap per block), thinking blocks,
  "Load earlier" pagination, and automatic tail-append while the run is
  live. Note: the gym's own rollout traces (ConversationView on training-run
  pages) are unrelated — they only exist for runs trained through the gym's
  RL frameworks and appear there natively.

Redeploy:

    MODAL_ENVIRONMENT=lab-dev \
    TRAINING_GYM_DASHBOARD_APP_NAME=lab-learning-dashboard \
    LEARNING_AGENT_OBS_VOLUME=lab-observatory \
    LEARNING_AGENT_OBS_URL=https://modal-labs-lab-dev--lab-observatory-web.modal.run \
    uv run modal deploy dashboards/app.py

## Fit assessment

Yes, this works, and the integration is small. Reasons:

1. The gym dashboard is already the shape we want: a Modal ASGI FastAPI app
   (`modal_training_gym/_dashboard.py`) serving a Svelte SPA built at image
   time (`dashboards/frontend`), with JSON APIs that read summary JSON from a
   Modal volume on demand. The observatory (`learning_agent_bench/observatory`)
   is the same pattern with a plainer frontend.
2. The learning-agent data already exists in a dashboard-ready form. The
   observatory ingests each run into `lab-observatory` volume as
   `runs/<run_id>/{record.json, workspace.json, status.json, raw/*}`, and
   `record.json` carries `scores.learning_log` — the LEARNING_LOG.jsonl rows
   verbatim (`{ts, kind, what, why, dev_score|result, artifacts}`). No new
   ingestion is needed; the dashboard only needs to read a second volume.
3. When the agent trains through the gym SDK, those runs already appear in
   this dashboard natively (Training runs page). The learning page is the
   missing piece that ties the agent's research log to those runs.

"Only the learning research log needs to be added" is right for the first
cut: one list page + one detail page. Traces, GPU telemetry, workspace
snapshots, and judge results stay in the observatory; the detail page links
out to the observatory run view for deep dives.

## Phase 1 — backend (new module, minimal diff to `_dashboard.py`)

1. New file `modal_training_gym/_learning_agent_routes.py`: a FastAPI router that reads
   the `lab-observatory` volume (name from env `LEARNING_AGENT_OBS_VOLUME`, default
   `lab-observatory`), mounted read-only in the dashboard app.
   - `GET /api/learning-runs` — list `runs/*/`, return each `record.json`'s
     `index_row` overlaid with `status.json` (that file is tiny and designed
     for polling), newest first. Cache per record mtime, same as observatory.
   - `GET /api/learning-runs/{run_id}` — the fields the detail page needs:
     `index_row`, `scores.learning_log`, `scores.results` summary, `status`.
     Validate `run_id` against `[A-Za-z0-9._-]+` like the observatory does.
2. Gate everything: if the volume doesn't exist or the env var is unset, the
   router returns empty lists and the nav item hides. Upstream behavior is
   unchanged, which keeps the fork rebasable on upstream main.
3. Auth: the new routes go behind the same optional dashboard password as the
   read routes (do NOT add them to `PASSWORD_EXEMPT_PATHS` — those are for
   token-authenticated write endpoints). Net security improvement over the
   public observatory URL; note the existing caveat that fav2 workspaces can
   embed gold answers — this plan exposes only the learning log and scores,
   not workspace snapshots.

Rejected alternative: having the learning agent write into
`training-gym-metadata` in the gym's TrainingRun format. That would touch the
bench pipeline and contort a LAB run into a training-run shape. Reading the
observatory volume keeps one writer per volume and zero changes to the bench.

## Phase 2 — frontend (two new pages, existing components)

1. `App.svelte`: add nav item `learning` ("Learning agent", `Book` icon —
   already imported), path `/learning`, entries in `pageMeta`/`pagePaths`/
   `pageFromPath`, and run-id extraction for `/learning/<run_id>`.
2. `src/pages/LearningPage.svelte`: runs table — run id, agent, task, state
   (StatusPill), best dev score, started (TimeAgo), duration. Reuse
   MinimalTable + FilterBar patterns from TrainingPage. 5s auto-refresh comes
   for free from the existing `load()` interval once wired in.
3. `src/pages/LearningRunDetailPage.svelte`: the research log —
   - chronological timeline of learning-log entries, one card per row: kind
     pill (checkpoint / submission / note), `what`, `why`, result/dev_score,
     artifact paths;
   - dev-score-over-time line chart from checkpoint-kind entries (reuse
     LineChart.svelte);
   - cross-links: for checkpoint entries whose tag or model_path matches a
     gym training run (from the already-fetched `/api/runs`), link to
     `/training/<training_run_id>`; plus one external link to the observatory
     run view for trace/GPU/workspace deep dive.
4. `src/lib/api.js`: `fetchLearningRuns()` / `fetchLearningRun(id)`.

## Phase 3 — verify locally, then deploy separately

1. Local dev: run `vite` dev server with a proxy to a small local stub that
   serves fixture JSON from
   `learning_agent_bench/observatory/fixtures/` (sample_record.json etc. —
   read-only use of existing fixtures). Screenshot both new pages before
   calling it done.
2. Experimental deploy: to `lab-dev` (or a personal env) with an overridden
   app name (e.g. `training-gym-dashboard-lab-exp`) so it cannot collide
   with any existing gym dashboard deployment. `modal deploy dashboards/app.py`
   from the fork.
3. Point it at the real `lab-observatory` volume and check a real fav2 run
   renders (needs a run whose LEARNING_LOG is non-empty — the seeded
   workspace copy is empty by design; ingested runs on the volume have the
   real ones).

## Effort

Backend ~200 lines in one new module + ~10-line diff to `_dashboard.py`;
frontend two pages built from existing components (~500 lines) + ~30 lines of
routing/nav. Roughly a day including local verification, plus the deploy.

## Open questions

1. Volume/environment: dashboard deploy must run in the same Modal
   environment as the `lab-observatory` volume (obs is on `lab-dev` per
   current notes) — confirm which environment the experimental deploy targets.
2. Whether the learning page should also surface eval `results` (canonical
   scores) as a column/section now, or wait until the log page proves useful.
