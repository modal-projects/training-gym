#!/bin/bash
# Launch Gemini CLI as a Learning Agent contestant (mirrors PTB's gemini). Reads $PROMPT and
# $AGENT_CONFIG from the environment, streams stream-json events to stdout for trace
# capture. Sandbox off because the run is already externally scoped by run.sh.
unset ANTHROPIC_API_KEY
unset CODEX_API_KEY

export GEMINI_SANDBOX="false"

gemini --yolo --model "$AGENT_CONFIG" --output-format stream-json -p "$PROMPT"
