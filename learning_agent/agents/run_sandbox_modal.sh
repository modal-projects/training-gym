#!/bin/bash
# Learning Agent workspace runner — Modal container edition:
#
#   agents/run_sandbox_modal.sh [--prepare-only] [--track easy|medium|hard] <scaffold> <task> [hours] [model]
#
# One Modal app (`lab-agent`), one container per session; live observability is
# always on (the container runs the watcher itself — no --watch flag). The
# session lives on the shared volume $MODAL_AGENT_VOLUME (default
# lab-agent-workspace) in environment $MODAL_ENVIRONMENT:
#
#   <task>/<session>/workspace   the agent's copy (its edits persist); the task
#                                and its corpus ride inside it at task/
#   <task>/<session>/logs        CLI session state, watcher log, filled spec
#
# session id: <scaffold>_<student-slug>_<stamp>, e.g. modal_glm52_qwen3.5-9b_20260728_151208
#
# Steps:
#   1. seed a complete staging workspace via workspace_setup/prepare_workspace.sh
#      (spec filled, task + corpus at task/, packages materialized)
#   2. ensure the pinned task-model snapshot is on lab-hf-cache (one-time)
#   3. upload the session dir to the volume
#   4. deploy the lab-agent app (idempotent) and spawn run_session detached
#
# --prepare-only stops after step 3 (session staged on the volume, no container).
set -euo pipefail

USAGE="usage: run_sandbox_modal.sh [--config <yaml>] [--prepare-only] [--track easy|medium|hard] [<scaffold> <task> [hours] [model]]"

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
HOURS="${HOURS:-23.5}"
TRACK="${TRACK:-easy}"
case "$TRACK" in easy|medium|hard) ;; *) echo "unknown track '$TRACK' (must be easy|medium|hard)" >&2; exit 2 ;; esac
lab_task_known "$ROOT" "$TASK" || { echo "unknown task '$TASK' (no task_configs/$TASK.yaml)" >&2; exit 2; }
[ -f "$ROOT/agents/$SCAFFOLD/solve.sh" ] || { echo "no scaffold at agents/$SCAFFOLD/solve.sh" >&2; exit 2; }

# Modal's function-timeout ceiling is 24 h; run.sh's own grace kill (budget + 5 min)
# must fire before the platform kills the container mid-write.
HOURS="$(python3 -c "import sys; h=float(sys.argv[1]); print(min(h, 23.75))" "$HOURS")"

# team env + volume names: process env wins, then repo .env. The trailing `|| :`
# is load-bearing: an absent key makes grep exit 1, which under `set -e` +
# pipefail killed the whole launch at the assignment below — silently, before a
# single line of output, so the documented default never got a chance to apply.
_envget() { grep -E "^$1=" "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'" || :; }
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-$(_envget MODAL_ENVIRONMENT)}"
[ -n "$MODAL_ENVIRONMENT" ] || { echo "MODAL_ENVIRONMENT not set (env or .env)" >&2; exit 2; }
export MODAL_ENVIRONMENT
VOLUME="${MODAL_AGENT_VOLUME:-$(_envget MODAL_AGENT_VOLUME)}"
VOLUME="${VOLUME:-lab-agent-workspace}"
export MODAL_AGENT_VOLUME="$VOLUME"
# first use creates the volume (the CLI's put/ls do not).
modal volume ls "$VOLUME" >/dev/null 2>&1 || modal volume create "$VOLUME" --version 2

STUDENT_SLUG="$(python3 -c "
import yaml
print(yaml.safe_load(open('$ROOT/bench/config.yaml'))['global']['base_model'].split('/')[-1].lower())")"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${SCAFFOLD}_${STUDENT_SLUG}_${STAMP}"

echo "== lab-agent (Modal) =="
echo "  session : $TASK/$SESSION"
echo "  volume  : $VOLUME (env $MODAL_ENVIRONMENT)"
echo "  track   : $TRACK   budget: ${HOURS}h"

# 1) seed the COMPLETE workspace into a local staging dir (shared routine;
#    the corpus rides inside the session upload at task/corpus)
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
source "$ROOT/workspace_setup/prepare_workspace.sh"
prepare_workspace "$ROOT" "$STAGE" "$TRACK" "$SCAFFOLD" "$TASK" "$HOURS"

# (No separate trainer upload: the staged workspace already carries the
# pinned packages — prepare_workspace materializes them via clone_repos.)

# 3) upload the complete session (workspace already carries task/ + corpus)
echo "== uploading session =="
modal volume put "$VOLUME" "$STAGE" "$TASK/$SESSION"

# 4) download the task model INTO the workspace (cloud-to-cloud — the weights
#    never route through the launch machine): a Modal sandbox mounts the
#    session's workspace and pulls the pinned snapshot to workspace/model/.
BASE_MODEL="$(python3 -c "import yaml; print(yaml.safe_load(open('$ROOT/bench/config.yaml'))['global']['base_model'])")"
BASE_REV="$(python3 -c "import yaml; print(yaml.safe_load(open('$ROOT/bench/config.yaml'))['global'].get('base_model_revision',''))")"
echo "== downloading task model into the workspace: $BASE_MODEL @ ${BASE_REV:-main} =="
python3 - "$VOLUME" "$TASK/$SESSION" "$BASE_MODEL" "$BASE_REV" <<'PYEOS'
import sys
import modal

volume_name, session, model, rev = sys.argv[1:5]
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface_hub", "hf_transfer")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
vol = modal.Volume.from_name(volume_name)
app = modal.App.lookup("lab-model-cache", create_if_missing=True)
cmd = ("from huggingface_hub import snapshot_download; "
       f"snapshot_download({model!r}, revision={rev!r} or None, "
       f"local_dir='/ws/{session}/workspace/model')")
sb = modal.Sandbox.create("python3", "-c", cmd, image=image,
                          volumes={"/ws": vol},
                          secrets=[modal.Secret.from_name("huggingface-secret")],
                          timeout=3600, app=app)
sb.wait()
if sb.returncode != 0:
    raise SystemExit(f"model download failed:\n{sb.stderr.read()}")
print("[model] weights at workspace/model/")
PYEOS

echo "== session staged: $VOLUME:$TASK/$SESSION =="

if [ "$PREPARE_ONLY" = 1 ]; then
    echo "prepare-only: no container launched. Inspect with:"
    echo "  modal volume ls $VOLUME $TASK/$SESSION --env $MODAL_ENVIRONMENT"
    exit 0
fi

# 4) deploy (idempotent — also refreshes the image) and spawn the session container
modal deploy "$ROOT/agents/modal_runner.py" >/dev/null
CALL_ID="$(python3 - "$MODAL_ENVIRONMENT" "$TASK" "$SESSION" "$SCAFFOLD" "$HOURS" "$MODEL" "$TRACK" <<'EOF'
import sys
import modal
env, task, session, scaffold, hours, model, track = sys.argv[1:8]
fn = modal.Function.from_name("lab-agent", "run_session", environment_name=env)
call = fn.spawn(task=task, session=session, scaffold=scaffold,
                hours=float(hours), model=model, track=track)
print(call.object_id)
EOF
)"

echo "== container spawned =="
echo "  call id  : $CALL_ID"
echo "  logs     : modal app logs lab-agent --env $MODAL_ENVIRONMENT"
echo "  artifacts: modal volume ls $VOLUME $TASK/$SESSION --env $MODAL_ENVIRONMENT"
echo "  stop     : python3 -c \"import modal; modal.FunctionCall.from_id('$CALL_ID').cancel()\""
echo "  viewer   : https://modal-labs-lab-dev--lab-observatory-web.modal.run"
