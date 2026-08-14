#!/bin/bash
# Learning Agent agent runner — put a CLI agent through one scored benchmark run.
#
#   agents/run.sh [--config <yaml>] [<scaffold> <task> [hours] [model]]
#     scaffold : a dir under agents/ holding solve.sh (+ human_readable_trace.py)
#                e.g. claude, claude_reprompt, codex, codex_xhigh, gemini, opencode,
#                glm5, qwen3max, *_non_api* (see agents/README.md for the full matrix)
#     task     : the workspace's task (task/, copied in at seed time)
#     hours    : wall-clock budget (default 24)
#     model    : model id/alias passed to the scaffold as $AGENT_CONFIG
#                (defaults per scaffold family, see below)
#     --config : task.yaml-schema YAML supplying any field the CLI left unset
#                (task: + session: scaffold/track/hours/model) — a task's own
#                task_configs/<T>.yaml is a valid session config
#
#   env LEARNING_AGENT_TRACK : easy|medium|hard (default easy) — which `instructions/data_access/<track>.md`
#                   block gets assembled into the prompt. Not a
#                   positional (keeps the 4-arg signature); run_sandbox.sh exports it
#                   before invoking this script inside the prepared workspace.
#
# Runs IN-PLACE against whatever repo root it lives under. For a scored run use
# agents/run_sandbox.sh, which first gives the agent its own workspace copy (the
# workspace is the submission) and then invokes this script inside it.
#
# What it does, mirroring PostTrainBench's run_task.sh but adapted for Learning Agent:
#   1. assemble the prompt (AGENTS.md + <TASK> + eval preamble)
#   2. write a timer.sh the agent can query; enforce a hard wall-clock kill
#   3. run the scaffold, capturing a timestamped stream-json trace
#   4. parse the trace to human-readable
#   5. DETERMINISTICALLY audit the trace for hidden-test access
#   6. diff runs/LEARNING_LOG.jsonl to report what the agent recorded
#
# Everything lands under agents/_runs/<scaffold>_<task>_<stamp>/ (gitignored).
set -uo pipefail

USAGE="usage: run.sh [--config <yaml>] [<scaffold> <task> [hours] [model]]"

CONFIG_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
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
cd "$ROOT"
source "$ROOT/agents/lib/session_config.sh"

if [ -n "$CONFIG_FILE" ]; then
    [ -f "$CONFIG_FILE" ] || { echo "no config at $CONFIG_FILE" >&2; exit 2; }
    TASK="${TASK:-$(yaml_top "$CONFIG_FILE" task)}"
    SCAFFOLD="${SCAFFOLD:-$(lab_yaml_session "$CONFIG_FILE" scaffold)}"
    HOURS="${HOURS:-$(lab_yaml_session "$CONFIG_FILE" hours)}"
    MODEL="${MODEL:-$(lab_yaml_session "$CONFIG_FILE" model)}"
    # LEARNING_AGENT_TRACK (exported by run_sandbox.sh) wins; config fills only when unset
    LEARNING_AGENT_TRACK="${LEARNING_AGENT_TRACK:-$(lab_yaml_session "$CONFIG_FILE" track)}"
fi
[ -n "$SCAFFOLD" ] && [ -n "$TASK" ] || { echo "$USAGE" >&2; exit 2; }
HOURS="${HOURS:-24}"

# Agents run ONLY in prepared sandboxes, never in the seed repo: the agent edits
# code (data, training, harness, submission), so launching it here would let it
# modify the benchmark itself. run_sandbox.sh writes the marker checked below.
# LEARNING_AGENT_ALLOW_IN_PLACE=1 overrides for operator smoke tests only.
if [ ! -f "$ROOT/.learning_agent_sandbox" ] && [ "${LEARNING_AGENT_ALLOW_IN_PLACE:-0}" != "1" ]; then
    echo "refusing to launch: $ROOT is not a prepared sandbox (no .learning_agent_sandbox marker)." >&2
    echo "use: agents/run_sandbox.sh <scaffold> <task> [hours] [model]" >&2
    exit 2
fi

SCAFFOLD_DIR="$ROOT/agents/$SCAFFOLD"
[ -f "$SCAFFOLD_DIR/solve.sh" ] || { echo "no scaffold at $SCAFFOLD_DIR/solve.sh" >&2; exit 2; }
[ -f "$ROOT/task/task.md" ] || lab_task_known "$ROOT" "$TASK" \
    || { echo "no task in this workspace (task/task.md missing)" >&2; exit 2; }

TRACK="${LEARNING_AGENT_TRACK:-easy}"
case "$TRACK" in easy|medium|hard) ;; *) echo "unknown LEARNING_AGENT_TRACK '$TRACK' (must be easy|medium|hard)" >&2; exit 2 ;; esac

# GPU jobs (bench.py train/rl -> `modal run`) read MODAL_ENVIRONMENT from the
# process env, NOT from .env — export it here so the agent's weights land in
# the environment .env names (the team-shared one), not the operator's profile
# default. A value already in the environment wins.
if [ -z "${MODAL_ENVIRONMENT:-}" ] && [ -f "$ROOT/.env" ]; then
    _menv="$(grep -E '^MODAL_ENVIRONMENT=' "$ROOT/.env" | tail -1 | cut -d= -f2-)"
    _menv="${_menv%\"}"; _menv="${_menv#\"}"; _menv="${_menv%\'}"; _menv="${_menv#\'}"
    if [ -n "$_menv" ]; then
        export MODAL_ENVIRONMENT="$_menv"
        echo "  modal env: $MODAL_ENVIRONMENT (from .env)"
    fi
fi

if [ -z "$MODEL" ]; then
    case "$SCAFFOLD" in
        claude*)   MODEL="opus" ;;
        codex_kimi3) MODEL="auto" ;;                  # resolved from the endpoint at launch
        codex_glm52) MODEL="zai-org/GLM-5.2-FP8" ;;   # before codex*: glob would eat it
        codex*)    MODEL="gpt-5.1-codex" ;;
        gemini*)   MODEL="gemini-3.1-pro" ;;
        opencode*) MODEL="zai/glm-5" ;;          # opencode wants provider/model
        modal_glm52) MODEL="modal-glm/zai-org/GLM-5.2-FP8" ;;
        glm5)      MODEL="glm-5" ;;
        qwen3max)  MODEL="qwen3-max" ;;
        *)         MODEL="default" ;;
    esac
fi

# Preemption resume: Modal restarts a preempted function with the same input,
# and the session's logs dir survives on the volume. Anchor the run identity
# and the deadline there, so a restarted container continues the SAME run on
# the SAME budget instead of minting a fresh one with a fresh clock.
RESUMED=0
STATE_FILE="${LEARNING_AGENT_LOGS_DIR:+$LEARNING_AGENT_LOGS_DIR/run_state.env}"
if [ -n "$STATE_FILE" ] && [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090 — sets RUN_NAME and DEADLINE
    source "$STATE_FILE"
    RESUMED=1
    RUN_DIR="$ROOT/agents/_runs/$RUN_NAME"
    mkdir -p "$RUN_DIR"
    BUDGET_SEC=$(( DEADLINE - $(date +%s) ))
    [ "$BUDGET_SEC" -ge 60 ] || BUDGET_SEC=60
else
    STAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_NAME="${SCAFFOLD}_${TASK}_${STAMP}"
    RUN_DIR="$ROOT/agents/_runs/$RUN_NAME"
    mkdir -p "$RUN_DIR"
    BUDGET_SEC=$(python3 -c "import sys;print(int(float(sys.argv[1])*3600))" "$HOURS")
    DEADLINE=$(( $(date +%s) + BUDGET_SEC ))
    if [ -n "$STATE_FILE" ]; then
        printf 'RUN_NAME=%s\nDEADLINE=%s\n' "$RUN_NAME" "$DEADLINE" > "$STATE_FILE"
    fi
fi

echo "== Learning Agent run =="
echo "  scaffold : $SCAFFOLD (model=$MODEL)"
echo "  task     : $TASK"
echo "  track    : $TRACK"
echo "  budget   : ${HOURS}h"
echo "  run dir  : $RUN_DIR"
[ "$RESUMED" = 1 ] && echo "  RESUMED after container restart — $(( BUDGET_SEC / 60 )) min left on the original deadline"

# 1) prompt
PROMPT="$(python3 agents/lib/make_prompt.py --task "$TASK" --root "$ROOT" --hours "$HOURS" --track "$TRACK")"
printf '%s' "$PROMPT" > "$RUN_DIR/prompt.txt"
export PROMPT AGENT_CONFIG="$MODEL"

# 2) timer + hard deadline (float-safe: HOURS may be fractional, e.g. 0.05 for
#    a smoke); on resume the deadline is the original one from run_state.env
bash agents/lib/make_timer.sh "$DEADLINE" "$ROOT/timer.sh"
echo "  wrote timer.sh (deadline epoch $DEADLINE, ${BUDGET_SEC}s)"

# snapshot submissions to diff afterwards (keep the original across restarts)
if [ ! -s "$RUN_DIR/learning_log.before" ]; then
    cp -f runs/LEARNING_LOG.jsonl "$RUN_DIR/learning_log.before" 2>/dev/null || : > "$RUN_DIR/learning_log.before"
fi

# 3) run scaffold with a portable process-group timeout (no GNU `timeout` on macOS).
#    `set -m` (job control) makes the backgrounded solve its OWN process group, so the
#    negative-PID kill below reaches the agent AND every child it spawns (modal, python,
#    the CLI's subprocesses). Without it, the child shares our group and the group-kill
#    silently misses — a runaway run would never die. Verified on this platform.
TRACE="$RUN_DIR/trace.jsonl"
# +5 min past budget: backstop kill only. The agent self-stops at its own timer.sh
# deadline (BUDGET_SEC); the grace lets an in-flight submit / log flush finish before
# we escalate to TERM then KILL, so a clean finisher is never truncated at the wire.
GRACE=$(( BUDGET_SEC + 300 ))
echo "== launching agent (hard kill in $((GRACE/60)) min) =="

set -m
# LEARNING_AGENT_RUN_ID reaches gpu_launcher, which tags every sandbox it creates so the
# observatory's GPU metering (observatory/gpu_metering.py) can attribute
# control-plane usage to this run without relying on time windows.
# Append (not truncate): a resumed run keeps the trace it already earned.
LEARNING_AGENT_RUN_DIR="$RUN_DIR" LEARNING_AGENT_RUN_ID="$(basename "$RUN_DIR")" \
    bash "$SCAFFOLD_DIR/solve.sh" >> "$TRACE" 2>>"$RUN_DIR/solve.err" &
SOLVE_PID=$!
( sleep "$GRACE"; kill -TERM -"$SOLVE_PID" 2>/dev/null; sleep 30; kill -KILL -"$SOLVE_PID" 2>/dev/null ) &
WATCH_PID=$!
set +m

BEGIN=$(date +%s)
wait "$SOLVE_PID"; SOLVE_EXIT=$?
kill -- -"$WATCH_PID" 2>/dev/null   # agent finished on its own; cancel the killer + its sleep
END=$(date +%s)

printf 'exit=%s\nseconds=%s\n' "$SOLVE_EXIT" "$((END-BEGIN))" > "$RUN_DIR/solve_status.txt"
rm -f "$ROOT/timer.sh"
echo "== agent stopped (exit $SOLVE_EXIT, $(( (END-BEGIN)/60 )) min) =="

# 4) parse trace to human-readable (verbatim fallback if no parser)
PARSER="$SCAFFOLD_DIR/human_readable_trace.py"
if [ -f "$PARSER" ]; then
    python3 "$PARSER" "$TRACE" -o "$RUN_DIR/trace.txt" 2>/dev/null \
        || cp "$TRACE" "$RUN_DIR/trace.txt"
else
    cp "$TRACE" "$RUN_DIR/trace.txt"
fi

# 5) deterministic contamination audit
echo "== auditing trace for hidden-test access =="
python3 agents/lib/audit_trace.py --trace "$TRACE" --task "$TASK" --root "$ROOT" \
    --out "$RUN_DIR/audit.json"
AUDIT_EXIT=$?

# 6) what did the agent record?
cp -f runs/LEARNING_LOG.jsonl "$RUN_DIR/learning_log.after" 2>/dev/null || : > "$RUN_DIR/learning_log.after"
NEW=$(comm -13 <(sort "$RUN_DIR/learning_log.before") <(sort "$RUN_DIR/learning_log.after") 2>/dev/null)
printf '%s\n' "$NEW" > "$RUN_DIR/submitted.jsonl"
NCK=$(grep -c . "$RUN_DIR/submitted.jsonl" 2>/dev/null || true)  # grep -c prints 0 itself on no match
if [ "$NCK" -eq 0 ]; then
    echo "WARNING: learning log is empty — the run recorded no experiments (runs/LEARNING_LOG.jsonl)" >&2
fi

echo
echo "================= RUN SUMMARY ================="
echo "  agent exit     : $SOLVE_EXIT"
echo "  integrity      : $([ $AUDIT_EXIT -eq 0 ] && echo CLEAN || echo CONTAMINATED)"
echo "  log entries new : $NCK"
echo "  artifacts      : $RUN_DIR"
echo "  next (operator): cd $ROOT && python submission/eval.py --input <held-out questions.json> --output answers.json"
echo "==============================================="

exit "$AUDIT_EXIT"
