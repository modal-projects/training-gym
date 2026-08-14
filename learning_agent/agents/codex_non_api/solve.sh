#!/bin/bash
# Codex CLI contestant authenticated via a ChatGPT Pro subscription instead of an API
# key (mirrors PTB's codex_non_api). Setup:
#   codex login --device-auth                       # follow the browser prompt
#   cp ~/.codex/auth.json agents/codex_non_api/     # gitignored
# Uses an isolated CODEX_HOME so the operator's real ~/.codex config is never touched
# (PTB appends forced_login_method to the real config; Learning Agent runs in-place, so we don't).
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Clear API keys so the CLI uses the ChatGPT auth from auth.json.
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
    -c model_reasoning_summary=detailed \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    --model "$AGENT_CONFIG"
