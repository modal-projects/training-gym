#!/bin/bash
# Learning Agent workspace runner — Docker container edition. Same shape as run_sandbox.sh:
#
#   agents/run_sandbox_docker.sh [--prepare-only] [--track easy|medium|hard] <scaffold> <task> [hours] [model]
#
# One detached container per session; live observability is always on (the
# container runs the watcher itself). Sessions live on the host, mirroring the
# Modal volume layout:
#
#   agents/_container_runs/<task>/<session>/workspace   mounted at /workspace
#   agents/_container_runs/<task>/<session>/logs        mounted at /logs
#
# Also mounted read-only: the seed repo's observatory/ + container_entry.sh at
# /seed (the watcher must not be agent-rewritable), and ~/.modal.toml so the
# agent's own `modal run` GPU jobs work from inside. Trainers arrive inside the
# workspace itself (toolbox/training_tool/<pkg>, materialized by prepare_workspace.sh).
#
# session id: <scaffold>_<student-slug>_<stamp> — identical to the Modal runner.
set -euo pipefail

USAGE="usage: run_sandbox_docker.sh [--config <yaml>] [--prepare-only] [--track easy|medium|hard] [<scaffold> <task> [hours] [model]]"

PREPARE_ONLY=0
TRACK=""
CONFIG_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --prepare-only) PREPARE_ONLY=1; shift ;;
        --track) TRACK="${2:?$USAGE}"; shift 2 ;;
        --track=*) TRACK="${1#--track=}"; shift ;;
        --config) CONFIG_FILE="${2:?$USAGE}"; shift 2 ;;
        --config=*) CONFIG_FILE="${1#--config=}"; shift ;;
        --) shift; break ;;
        -*) echo "unknown flag '$1' — $USAGE" >&2; exit 2 ;;
        *) break ;;
    esac
done

SCAFFOLD="${1:-}"
TASK="${2:-}"
HOURS="${3:-}"
MODEL="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/agents/lib/session_config.sh"

# config-driven session: unset fields come from the YAML's task:/session: block
# (task_configs/<T>.yaml IS a valid session config — the task's launch defaults)
if [ -n "$CONFIG_FILE" ]; then
    [ -f "$CONFIG_FILE" ] || { echo "no config at $CONFIG_FILE" >&2; exit 2; }
    TASK="${TASK:-$(yaml_top "$CONFIG_FILE" task)}"
    SCAFFOLD="${SCAFFOLD:-$(lab_yaml_session "$CONFIG_FILE" scaffold)}"
    HOURS="${HOURS:-$(lab_yaml_session "$CONFIG_FILE" hours)}"
    MODEL="${MODEL:-$(lab_yaml_session "$CONFIG_FILE" model)}"
    TRACK="${TRACK:-$(lab_yaml_session "$CONFIG_FILE" track)}"
fi
[ -n "$SCAFFOLD" ] && [ -n "$TASK" ] || { echo "$USAGE" >&2; exit 2; }
HOURS="${HOURS:-24}"
TRACK="${TRACK:-easy}"
case "$TRACK" in easy|medium|hard) ;; *) echo "unknown track '$TRACK' (must be easy|medium|hard)" >&2; exit 2 ;; esac
lab_task_known "$ROOT" "$TASK" || { echo "unknown task '$TASK' (no task_configs/$TASK.yaml)" >&2; exit 2; }
[ -f "$ROOT/agents/$SCAFFOLD/solve.sh" ] || { echo "no scaffold at agents/$SCAFFOLD/solve.sh" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "docker not installed / not on PATH" >&2; exit 2; }

# trailing `|| :`: an absent key makes grep exit 1, which under `set -e` +
# pipefail aborts the launch at the assignment instead of falling back (see
# run_sandbox_modal.sh — it failed exactly that way).
_envget() { grep -E "^$1=" "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'" || :; }
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-$(_envget MODAL_ENVIRONMENT)}"
MODAL_OBS_VOLUME="${MODAL_OBS_VOLUME:-$(_envget MODAL_OBS_VOLUME)}"

STUDENT_SLUG="$(python3 -c "
import yaml
print(yaml.safe_load(open('$ROOT/bench/config.yaml'))['global']['base_model'].split('/')[-1].lower())")"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${SCAFFOLD}_${STUDENT_SLUG}_${STAMP}"
SESSION_DIR="$ROOT/agents/_container_runs/$TASK/$SESSION"
mkdir -p "$SESSION_DIR/logs"

echo "== lab-agent (Docker) =="
echo "  session : $TASK/$SESSION"
echo "  dir     : $SESSION_DIR"
echo "  track   : $TRACK   budget: ${HOURS}h"

# 1) seed the workspace (shared routine — identical to the local/Modal runners)
source "$ROOT/workspace_setup/prepare_workspace.sh"
prepare_workspace "$ROOT" "$SESSION_DIR" "$TRACK" "$SCAFFOLD" "$TASK" "$HOURS"

if [ "$PREPARE_ONLY" = 1 ]; then
    echo "prepare-only: no container launched. Workspace at $SESSION_DIR/workspace"
    exit 0
fi

# 2) image (cached after the first build)
docker build -q -t lab-agent:latest "$ROOT/agents/docker" >/dev/null

# 3) launch detached; container_entry.sh runs the watcher + agents/run.sh
DOCKER_ARGS=(
    -d --name "lab_${SESSION}"
    -v "$SESSION_DIR/workspace":/workspace
    -v "$SESSION_DIR/logs":/logs
    -v "$ROOT/observatory":/seed/observatory:ro
    -v "$ROOT/agents/lib/container_entry.sh":/seed/container_entry.sh:ro
    -e MODAL_ENVIRONMENT="$MODAL_ENVIRONMENT"
    -e MODAL_OBS_VOLUME="$MODAL_OBS_VOLUME"
)
# the agent's own `modal run` GPU jobs need credentials (HOME is /logs/home in-container)
[ -f "$HOME/.modal.toml" ] && DOCKER_ARGS+=(-v "$HOME/.modal.toml":/logs/home/.modal.toml:ro) \
    || echo "warn: no ~/.modal.toml — in-container modal jobs and obs ingest will fail" >&2

CID="$(docker run "${DOCKER_ARGS[@]}" lab-agent:latest \
    bash /seed/container_entry.sh "$SCAFFOLD" "$TASK" "$HOURS" "$MODEL" "$TRACK" /workspace /logs /seed)"

echo "== container spawned =="
echo "  container: ${CID:0:12} (lab_${SESSION})"
echo "  logs     : docker logs -f lab_${SESSION}"
echo "  artifacts: $SESSION_DIR"
echo "  stop     : docker stop lab_${SESSION}"
echo "  viewer   : https://modal-labs-lab-dev--lab-observatory-web.modal.run"
