#!/bin/bash
# Codex CLI contestant at model_reasoning_effort=xhigh (mirrors PTB's codex_xhigh).
# Learning Agent passes the effort as a -c override instead of editing ~/.codex/config.toml,
# because Learning Agent runs in-place on the operator's machine.
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

printf '%s' "$PROMPT" | codex --search exec --json \
    -c model_reasoning_effort=xhigh \
    -c model_reasoning_summary=detailed \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    --model "$AGENT_CONFIG"
