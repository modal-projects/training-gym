#!/bin/bash
# codex_non_api at model_reasoning_effort=xhigh (mirrors PTB's codex_non_api_xhigh).
# Auth setup as in codex_non_api: put auth.json in this directory (gitignored).
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

export CODEX_API_KEY=""
export OPENAI_API_KEY=""

SCAFFOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTH_FILE="$SCAFFOLD_DIR/auth.json"
if [ ! -f "$AUTH_FILE" ]; then
    echo "ERROR: no auth.json at $AUTH_FILE (codex login --device-auth, then copy ~/.codex/auth.json)" >&2
    exit 1
fi

export CODEX_HOME="$(mktemp -d)"
cp "$AUTH_FILE" "$CODEX_HOME/auth.json"
printf 'forced_login_method = "chatgpt"\n' > "$CODEX_HOME/config.toml"

printf '%s' "$PROMPT" | codex --search exec --json \
    -c model_reasoning_effort=xhigh \
    -c model_reasoning_summary=detailed \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    --model "$AGENT_CONFIG"
