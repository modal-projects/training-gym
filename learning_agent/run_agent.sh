#!/bin/bash
# run_agent — THE launch entrypoint: one task config in, one detached run out.
#
#   ./run_agent.sh task_configs/fav2.yaml           # everything from the config
#   ./run_agent.sh task_configs/fav2_rl.yaml 6      # wall-clock hours override
#
# The config supplies the task identity and its session defaults (scaffold,
# track, hours, model). The rest follows automatically: workspace seeding
# (workspace_setup/prepare_workspace.sh composes AGENTS.md, task/, toolbox/)
# and the detached Modal launch (agents/run_sandbox_modal.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${1:?usage: run_agent.sh <task_configs/<task>.yaml> [hours] [scaffold] [model]}"
HOURS="${2:-}"
SCAFFOLD="${3:-}"
MODEL="${4:-}"
exec "$ROOT/agents/run_sandbox_modal.sh" --config "$CFG" "$SCAFFOLD" "" "$HOURS" "$MODEL"
