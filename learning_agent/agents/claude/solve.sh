#!/bin/bash
# Launch Claude Code as a Learning Agent contestant. Reads $PROMPT and $AGENT_CONFIG (the model
# id/alias) from the environment, streams stream-json events to stdout so the run's
# trace can be captured and parsed. Run from the repo root by run.sh.
unset GEMINI_API_KEY
unset CODEX_API_KEY

# Learning Agent training/eval jobs are long; let the Bash tool block up to 10 h.
export BASH_MAX_TIMEOUT_MS="36000000"

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions
