#!/bin/bash
# Learning Agent launch preflight — every FREE check that predicts whether a paid run
# will survive. Usage: bash launch/preflight.sh [task]   (default fav2)
#
# Exit 0 = safe to launch. Exit 1 = something below would waste the run.
# Soft warnings (canonical:false judging) do not fail the preflight.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TASK="${1:-fav2}"
FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=1; }

echo "== Learning Agent preflight (task: $TASK) =="

# 1. instrument integrity
if python3 bench.py verify >/dev/null 2>&1; then ok "pins verify (benchmark surface intact)"
else bad "bench.py verify — graded surface drifted; freeze or fix before running"; fi

# 2. test suites
if python3 -m pytest observatory/tests/ -q >/dev/null 2>&1; then
    ok "all test suites"
else bad "test suites — run: python3 -m pytest observatory/tests/ -q"; fi

# 3. task inputs
[ -d "tasks/$TASK/corpus" ] && ok "corpus on disk (tasks/$TASK/corpus)" \
    || bad "corpus missing at tasks/$TASK/corpus (distributed separately)"
[ -f "tasks/$TASK/dev.json" ] && ok "dev.json present" || bad "tasks/$TASK/dev.json missing"
[ -f "tasks/$TASK/test.json" ] && ok "test.json present (operator-side scoring input)" \
    || warn "tasks/$TASK/test.json not on this machine — scoring must happen elsewhere"
[ -f "tasks/$TASK/task.md" ] && ok "task.md present" || bad "tasks/$TASK/task.md missing"
[ -f "tasks/$TASK/brief.md" ] && ok "brief.md present (hard track input)" \
    || warn "tasks/$TASK/brief.md missing — hard track unavailable"

# 4. trainers
[ -d "learning_agent_workspace/toolbox/training_tool/axolotl" ] && [ -d "learning_agent_workspace/toolbox/training_tool/slime" ] \
    && ok "pinned training packages (axolotl + slime materialized)" \
    || bad "learning_agent_workspace/toolbox/training_tool/{axolotl,slime} missing — run toolbox/clone_repos.py"

# 5. .env + judge path
if [ ! -f .env ]; then
    bad ".env missing (cp .env.example .env, then edit — see launch/README.md step 0)"
else
    KEY="$(sed -n 's/^ANTHROPIC_API_KEY=//p' .env | tail -1)"
    if [ -z "$KEY" ]; then
        if command -v claude >/dev/null 2>&1; then
            warn "no ANTHROPIC_API_KEY — judge falls back to the claude CLI (canonical:false)"
        else
            bad "no ANTHROPIC_API_KEY and no claude CLI — no judge path at all"
        fi
    elif [ "${#KEY}" -lt 40 ]; then
        bad "ANTHROPIC_API_KEY looks like a placeholder (${#KEY} chars) — remove the line or use a real key; a fake value breaks judge auto-fallback"
    else
        ok "ANTHROPIC_API_KEY present (canonical judging)"
    fi
    MENV="$(sed -n 's/^MODAL_ENVIRONMENT=//p' .env | tail -1)"
    if [ -z "$MENV" ]; then
        bad "MODAL_ENVIRONMENT not in .env — weights and run records would scatter across envs"
    else
        ok "MODAL_ENVIRONMENT=$MENV"
    fi
fi

# 6. Modal reachable + secret present in the env
if command -v modal >/dev/null 2>&1; then
    # NB: modal's table truncates long names ("huggingface-secr…") — match a prefix
    if [ -n "${MENV:-}" ] && modal secret list --env "$MENV" 2>/dev/null | grep -q "huggingface-secr"; then
        ok "huggingface-secret exists in $MENV (training + eval serving)"
    elif [ -n "${MENV:-}" ]; then
        bad "huggingface-secret missing in env '$MENV' — bench.py train and bench.py score will crash (see dev/MODAL.md)"
    fi
else
    bad "modal CLI not installed/authenticated"
fi

# 7. agent CLI for claude* scaffolds
command -v claude >/dev/null 2>&1 && ok "claude CLI on PATH (claude* scaffolds)" \
    || warn "no claude CLI — only codex/gemini/opencode scaffolds usable"

# 8. modal_glm52 scaffold: opencode CLI + the team's Modal GLM endpoint
source agents/modal_glm52/config.env
command -v opencode >/dev/null 2>&1 \
    && ok "opencode CLI on PATH (modal_glm52 + opencode scaffolds)" \
    || warn "opencode CLI missing — modal_glm52/opencode scaffolds unusable"
MODELS_JSON="$(curl -fsS --max-time 15 "${MODAL_GLM52_BASE_URL}/v1/models" 2>/dev/null || true)"
if python3 -c 'import json,sys
expected, raw = sys.argv[1:]
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    raise SystemExit(1)
raise SystemExit(0 if any(x.get("id") == expected for x in data.get("data", [])) else 1)
' "$MODAL_GLM52_MODEL" "$MODELS_JSON"; then
    ok "Modal GLM-5.2 endpoint live ($MODAL_GLM52_MODEL)"
else
    warn "Modal GLM-5.2 endpoint unavailable or serving the wrong model — modal_glm52 runs would waste"
fi

echo
if [ "$FAIL" = 0 ]; then
    echo "PREFLIGHT PASSED — launch (detached) with:"
    echo "  bash launch/detach_run.sh --watch --track easy claude_reprompt $TASK 24"
else
    echo "PREFLIGHT FAILED — fix the FAIL lines above before spending a run."
fi
exit "$FAIL"
