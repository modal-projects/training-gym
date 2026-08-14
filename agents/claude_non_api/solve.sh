#!/bin/bash
# Claude Code contestant authenticated via a Claude subscription OAuth token instead of
# an API key (mirrors PTB's claude_non_api). Setup:
#   claude setup-token                     # follow the browser prompt
#   echo "sk-ant-..." > agents/claude_non_api/oauth_token   (gitignored)
unset GEMINI_API_KEY
unset CODEX_API_KEY

# Clear the API key so the CLI uses the OAuth token from the subscription.
export ANTHROPIC_API_KEY=""

TOKEN_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/oauth_token"
if [ -f "$TOKEN_FILE" ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$TOKEN_FILE")"
else
    echo "ERROR: no oauth_token file at $TOKEN_FILE (run: claude setup-token)" >&2
    exit 1
fi

export BASH_MAX_TIMEOUT_MS="36000000"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --effort high --thinking-display summarized \
    --dangerously-skip-permissions
