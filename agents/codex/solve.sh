#!/bin/bash
# Launch Codex CLI as a Learning Agent contestant. Reads $PROMPT and $AGENT_CONFIG (the model id)
# from the environment, streams JSONL events to stdout for trace capture. Run from the
# repo root by run.sh; --skip-git-repo-check because the benchmark repo need not be a
# git checkout, and the bypass flag because the run is already externally scoped.
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

printf '%s' "$PROMPT" | codex --search exec --json \
    -c model_reasoning_summary=detailed \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    --model "$AGENT_CONFIG"
