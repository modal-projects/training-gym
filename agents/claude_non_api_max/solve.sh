#!/bin/bash
# Claude Code contestant on a subscription OAuth token at effort level "max"
# (mirrors PTB's claude_non_api_max; Opus 4.6+ only — maximum reasoning). Setup as in
# claude_non_api: put the token in agents/claude_non_api_max/oauth_token (gitignored).
unset GEMINI_API_KEY
unset CODEX_API_KEY

export ANTHROPIC_API_KEY=""

TOKEN_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/oauth_token"
if [ -f "$TOKEN_FILE" ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$TOKEN_FILE")"
else
    echo "ERROR: no oauth_token file at $TOKEN_FILE (run: claude setup-token)" >&2
    exit 1
fi

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="max"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions
