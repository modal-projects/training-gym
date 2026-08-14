#!/bin/bash
# Runs INSIDE a container (Modal or Docker) — one Learning Agent agent session end to end.
# Both container runners execute this same script so a session behaves
# identically regardless of where it is hosted.
#
#   container_entry.sh <scaffold> <task> <hours> <model> <track> <ws_dir> <logs_dir> [seed_dir]
#
#   <ws_dir>   the prepared workspace (mounted from the session's volume subdir /
#              bind mount — everything the agent does in it persists)
#   <logs_dir> session log home. HOME is pointed INSIDE it so the CLI's own
#              session state (OpenCode's ~/.local/share/opencode storage,
#              Claude Code's ~/.claude session jsonl, codex $CODEX_HOME)
#              persists next to the run record instead of dying with the container.
#   <seed_dir> read-only seed copy of observatory/ (+ this script), provided by
#              the runner — the live watcher runs from it so the agent cannot
#              rewrite its own reporter.
set -uo pipefail

USAGE="usage: container_entry.sh <scaffold> <task> <hours> <model> <track> <ws_dir> <logs_dir> [seed_dir]"
SCAFFOLD="${1:?$USAGE}"; TASK="${2:?$USAGE}"; HOURS="${3:?$USAGE}"
MODEL="${4:-}"; TRACK="${5:-easy}"; WS="${6:?$USAGE}"; LOGS="${7:?$USAGE}"; SEED="${8:-}"

[ -f "$WS/.learning_agent_sandbox" ] || { echo "refusing: $WS is not a prepared sandbox (no .learning_agent_sandbox)" >&2; exit 2; }

# CLI-native session state persists with the run record
export HOME="$LOGS/home"
mkdir -p "$HOME"
# Scaffolds that keep CLI state OUTSIDE $HOME (codex_glm52's isolated
# CODEX_HOME) persist it under the session's logs dir via this pointer.
export LEARNING_AGENT_LOGS_DIR="$LOGS"
git config --global user.name  "lab-agent"          2>/dev/null || true
git config --global user.email "lab-agent@localhost" 2>/dev/null || true
git config --global --add safe.directory '*'         2>/dev/null || true

export LEARNING_AGENT_TRACK="$TRACK"

# Drop a copy of the stitched task spec next to the logs for easy inspection
# (the workspace AGENTS.md is already concrete — stitched at seeding).
cp "$WS/AGENTS.md" "$LOGS/AGENTS.filled.md" 2>>"$LOGS/entry.err" || true

# Live observability from the seed copy (agent-tamper-proof). The watcher waits
# for the inner run dir, ingests to the shared observatory volume, archives the
# workspace on its final pass, and exits by itself on solve_status.txt.
WATCH_PID=""
if [ -n "$SEED" ] && [ -d "$SEED/observatory" ]; then
    (
        cd "$SEED"
        for _ in $(seq 1 120); do
            if ls -d "$WS/agents/_runs"/*/ >/dev/null 2>&1; then
                exec python3 -m observatory.cli watch "$(dirname "$WS")" \
                    --interval 60 --archive-workspace
            fi
            sleep 5
        done
        echo "obs watch: no run dir appeared within 10 min — giving up" >&2
    ) >"$LOGS/obs_watch.log" 2>&1 &
    WATCH_PID=$!
fi

cd "$WS"
bash agents/run.sh "$SCAFFOLD" "$TASK" "$HOURS" "$MODEL"
RC=$?

# Give the watcher up to 4 min for its final ingest pass, then stop waiting.
if [ -n "$WATCH_PID" ]; then
    for _ in $(seq 1 24); do
        kill -0 "$WATCH_PID" 2>/dev/null || break
        sleep 10
    done
    kill "$WATCH_PID" 2>/dev/null || true
fi
exit "$RC"
