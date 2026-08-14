# launch/ — the operator playbook

Everything needed to launch, watch, and score a benchmark run, written so it
can be executed top-to-bottom without any other context. Account-specific
names (Modal environments, secrets, deployed URLs) live in
[dev/MODAL.md](../dev/MODAL.md); this playbook reads them from
`.env` and never hardcodes them.

The five stages behind these commands are explained with schematics at the
deployed viewer's `/how` page (or `observatory/static/how.html` in this repo).

---

## 0 · One-time setup (per operator machine)

```bash
cp .env.example .env         # then edit:
#   MODAL_ENVIRONMENT=<team env>        # see dev/MODAL.md for the current one
#   MODAL_OBS_VOLUME=lab-observatory
#   ANTHROPIC_API_KEY=<real key>        # OPERATOR-ONLY: used for the canonical
#                                       # judge at END-of-run scoring (step 5).
#                                       # run_sandbox.sh strips API keys from the
#                                       # .env it seeds — the agent evaluates
#                                       # itself during the run (its own judgment
#                                       # or the CLI-backed judge instruments).
#                                       # Leave the line out if you have no key —
#                                       # scoring then uses the claude CLI and
#                                       # stamps canonical:false. NEVER leave a
#                                       # placeholder value: it breaks auto-fallback.
```

- `toolbox/training/_vendor/{axolotl,slime}` must exist as a sibling checkout.
- `modal token` authenticated; the chosen environment must contain the
  `huggingface-secret` (needed by training and eval serving — see
  dev/MODAL.md for which envs have it).
- Task corpora on disk under `tasks/<task>/corpus/` (distributed separately).
- The team viewer, once: `modal deploy observatory/app.py`
  (uses `MODAL_ENVIRONMENT` from your shell; `-e <env>` to override).

## 1 · Preflight (free, ~1 min) — run before EVERY launch

```bash
bash launch/preflight.sh fav2        # or dspy / openclaw
```

Green means: pins verified, all test suites pass, task inputs present,
Modal reachable in the configured environment, trainers linked, judge path
resolves (API key or CLI fallback). The script exits nonzero on anything
that would waste a run.

## 2 · Dress rehearsal (pennies, ~10 min) — before the first run of a series

A 3-minute budget through the ENTIRE chain — prepare, launch, trace, watcher
upload, viewer, kill, audit:

```bash
bash agents/run_sandbox.sh --watch --track easy claude_reprompt fav2 0.05
```

Then open the viewer: the run should appear on the index within a minute,
state `running`, trace growing; after ~8 minutes it flips to finished with
an audit verdict and a workspace archive. If all that happened, the full
pipeline works.

## 3 · The real run

```bash
# fav2, easy track, Claude agent with re-prompting, 24h budget — DETACHED:
bash launch/detach_run.sh --watch --track easy claude_reprompt fav2 24
```

- **Always launch multi-hour runs via `launch/detach_run.sh`** — it starts the
  run in its own session + process group (`setsid` semantics; macOS ships no
  setsid binary, the script uses python's `start_new_session`). A run launched
  as a plain foreground/background job dies with the terminal, the SSH
  connection, or a session-scoped task manager — this killed a real 24h run
  1h39m in on 2026-07-21. The script prints the pid, the log path, and the
  exact kill command.
- `--track medium|hard` for the harder variants (see README "Tracks").
- Scaffolds live under `agents/` (claude_reprompt, codex_xhigh_reprompt, …);
  `claude*` scaffolds use the machine's logged-in `claude` CLI.
- The hard kill fires at budget+5 min regardless of how it was launched.
- `--watch` starts the detached observatory watcher: live trace at the
  viewer while the run goes, final ingest + workspace archive when it ends.
  Its log: `agents/_runs/ws_<...>/obs_watch.log`.

## 4 · While it runs (all read-only)

```bash
tail -f agents/_runs/ws_<...>/workspace/agents/_runs/*/trace.jsonl   # raw trace
modal volume ls lab-out models/                                      # checkpoints landing
cat agents/_runs/ws_<...>/obs_watch.log                              # watcher heartbeat
```

Or just watch the viewer: index row → run page → Trace / Learning /
Scores tabs update live. Kill a runaway run by killing the run_sandbox
process group; the watcher exits on its own.

## 5 · Scoring (after the run ends)

```bash
cd agents/_runs/ws_<scaffold>_<task>_<stamp>/workspace

# 1. the submission contract, against the HELD-OUT questions (operator-only):
python submission/eval.py --input tasks/fav2/test.json --output answers.json

# 2. judge it (same judge pins as dev; canonical needs ANTHROPIC_API_KEY):
python toolbox/eval_tool/rubric_eval.py --dev tasks/fav2/test.json \
    --answers answers.json --task fav2 \
    --judge-model claude-opus-4-20250514 --n-votes 3 --out results_test.json

# 3. the untrained-base floor on the same split (the margin's denominator):
python bench.py score --task fav2 --model Qwen/Qwen3.5-9B --split test --tag base_floor

# 4. the audit verdict rides with the run (written at run end):
cat agents/_runs/ws_<...>/audit.json | python3 -m json.tool | head
```

The number that matters = submission mean − base-floor mean, on test.
Compare it with the agent's own dev claims (CHECKPOINTS.jsonl) for the
dev-vs-test calibration story. Note step 1 is the FIRST time held-out data
touches the workspace: run it only after the agent is done and audited.

## 6 · Troubleshooting

| symptom | cause / fix |
|---|---|
| judge errors `backend 'api' needs ANTHROPIC_API_KEY` | placeholder or missing key in `.env`. Remove the line (CLI fallback, canonical:false) or put a real key |
| training dies with image-processor / HF download errors | `huggingface-secret` missing in this Modal environment, or `lab-hf-cache` was wiped (it repopulates automatically, ~18 GB once) |
| `refusing to launch: not a prepared sandbox` | you ran `agents/run.sh` directly in the seed repo — always launch via `run_sandbox.sh` |
| weights/records land in the wrong Modal env | `MODAL_ENVIRONMENT` not in `.env` (runs) or not exported in your shell (manual commands) |
| viewer shows nothing | watcher not started (`--watch` missing)? check `obs_watch.log`; ingest manually: `python3 -m observatory.cli ingest agents/_runs/ws_<...> --archive-workspace` |
| run won't die | kill the process group: `kill -- -<pid of run_sandbox>`; the +5 min hard kill also reaps children |
