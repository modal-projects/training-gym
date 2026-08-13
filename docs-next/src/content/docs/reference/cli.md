---
title: CLI Reference
description: Command-line interface for the Training Gym SDK.
---

The `training-gym` CLI is installed automatically with the package.

```bash
pip install git+https://github.com/modal-projects/training-gym.git@main
```

Run `training-gym --help` or `training-gym <command> --help` to see the
available commands and options.

## Dashboard configuration

The dashboard commands require a Modal account with `modal token set` already
configured.

### `training-gym setup`

Deploy the observability dashboard to your Modal account:

```bash
training-gym setup
```

The command builds a Modal image (Node.js + Svelte frontend + FastAPI backend),
deploys it as a persistent web app, saves its URL locally, and prints the URL.

Use one of the authentication flags to explicitly control Modal proxy
authentication:

```bash
training-gym setup --proxy-auth
training-gym setup --no-proxy-auth
```

### `training-gym open`

Open the deployed dashboard in your default browser:

```bash
training-gym open
```

### `training-gym set-password`

Set or clear Basic Auth for the dashboard and redeploy it:

```bash
# Prompt securely for a password
training-gym set-password

# Set a password non-interactively
training-gym set-password --password "$DASHBOARD_PASSWORD"

# Clear the password
training-gym set-password --password ""
```

### `training-gym set-proxy-auth`

Interactively save or replace the `MODAL_KEY` and `MODAL_SECRET` used to call
served endpoints that were deployed with `unauthenticated=False`:

```bash
training-gym set-proxy-auth
```

## Training runs

The `training-gym run` commands inspect runs through your deployed dashboard.
Commands that support `-j` / `--json` emit machine-readable JSON.

### `training-gym run list`

List runs, sorted by most recently updated:

```bash
training-gym run list
training-gym run list --status failed --since 24h
training-gym run list --model Qwen/Qwen3-4B --group nightly -j
```

Options:

- `--status STATUS`: filter by run status.
- `--model MODEL`: filter by model.
- `--dataset DATASET`: filter by dataset.
- `--recipe RECIPE`: filter by training recipe.
- `--group GROUP`: filter by run group.
- `--since TIME`: include only runs created or updated since epoch seconds, an
  ISO 8601 timestamp, or a relative time such as `30m`, `2h`, or `7d`.
- `--limit N`: set the maximum number of runs returned (default: 50).
- `-j`, `--json`: emit machine-readable JSON.

### `training-gym run get`

Show status and top-level metadata for one run:

```bash
training-gym run get RUN_ID
training-gym run get RUN_ID --verbose
training-gym run get RUN_ID --json
```

Options:

- `--verbose`: include reward history and rollout data.
- `-j`, `--json`: emit machine-readable JSON.

### `training-gym run params`

Show the framework training recipe recorded for one run:

```bash
training-gym run params RUN_ID
training-gym run params RUN_ID --json
```

Options:

- `-j`, `--json`: emit machine-readable JSON.

### `training-gym run logs`

Show the latest 100 log lines for a run, or stream new logs:

```bash
training-gym run logs RUN_ID
training-gym run logs RUN_ID --follow
training-gym run logs RUN_ID --since 30m --tail 500
training-gym run logs RUN_ID --search "checkpoint"
```

Options:

- `-f`, `--follow`: stream logs until interrupted or the run stops.
- `--since START` / `--until END`: select a timestamp or relative time window.
- `-n`, `--tail N`: return the newest N lines (maximum 20,000).
- `--search TEXT`: filter logs by text.
- `-j`, `--json`: emit JSON; when following, emit one JSON object per event.

`--follow` cannot be combined with `--since`, `--until`, or `--tail`.

### `training-gym run trace`

Download rollout traces for a run. `--out` specifies a parent directory; files
are written beneath it in a directory named after the run:

```bash
training-gym run trace RUN_ID --out ./traces
training-gym run trace RUN_ID --out ./traces --step 1,4,9
training-gym run trace RUN_ID --out ./traces --step 4-100:2 --dry-run
```

Options:

- `--out DIR`: set the parent output directory (required).
- `--step STEP`: select steps using comma-separated values or start-inclusive,
  end-exclusive ranges with an optional stride, such as `4-100:2`.
- `--dry-run`: report the expected sample count and size without downloading.
- `-y`, `--yes`, `--force`: skip the confirmation prompt.
- `-j`, `--json`: emit machine-readable JSON.

## Cleanup

### `training-gym cleanup`

Delete metadata for failed or cancelled runs older than seven days:

```bash
# Preview without deleting
training-gym cleanup --dry-run

# Delete terminal runs older than 30 days
training-gym cleanup --older-than-days 30
```

Cleanup removes run metadata and rollout data; it does not delete checkpoints.
