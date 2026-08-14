#!/bin/bash
# GLM-5 contestant via Claude Code pointed at Z.AI's Anthropic-compatible API (mirrors
# PTB's glm5). Requires ZAI_API_KEY in the environment (a Z.AI "Coding Plan"; a bare
# API key does not work with the Anthropic endpoint).
# Reference: https://docs.z.ai/devpack/tool/claude
unset GEMINI_API_KEY
unset CODEX_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"
export API_TIMEOUT_MS="3000000"

export ANTHROPIC_API_KEY="${ZAI_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${ZAI_API_KEY}"
export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_SMALL_FAST_MODEL="${AGENT_CONFIG}"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --dangerously-skip-permissions
