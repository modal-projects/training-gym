#!/bin/bash
# Qwen3-Max contestant via Claude Code pointed at Qwen's Anthropic-compatible API
# (mirrors PTB's qwen3max). Requires DASHSCOPE_API_KEY in the environment
# (international DashScope endpoint).
# Reference: https://qwen.ai/blog?id=qwen3-max-thinking
unset GEMINI_API_KEY
unset CODEX_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"

export ANTHROPIC_API_KEY="${DASHSCOPE_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${DASHSCOPE_API_KEY}"
export ANTHROPIC_BASE_URL="https://dashscope-intl.aliyuncs.com/apps/anthropic"
export ANTHROPIC_MODEL="${AGENT_CONFIG}"
export ANTHROPIC_SMALL_FAST_MODEL="${AGENT_CONFIG}"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --dangerously-skip-permissions
