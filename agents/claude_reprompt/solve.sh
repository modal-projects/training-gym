#!/bin/bash
# Claude Code contestant with a re-prompt loop (mirrors PTB's claude_reprompt): if the
# agent finishes early, resume the same session with the time remaining until fewer
# than MIN_REMAINING_SECONDS are left. Reads $PROMPT and $AGENT_CONFIG from the
# environment; run from the repo root by run.sh (timer.sh lives there).
unset GEMINI_API_KEY
unset CODEX_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"

# 30-min floor: do not resume a finished session with less than this left. A fresh
# continuation cannot meaningfully improve the result before the hard deadline, and
# re-warming the session would consume most of the remaining window.
MIN_REMAINING_SECONDS=1800

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions

# Re-prompt loop: parse the machine-readable seconds_left= line (portable — no GNU grep).
while true; do
    LEFT=$(bash timer.sh 2>/dev/null | sed -n 's/^seconds_left=//p')
    [ -n "$LEFT" ] || break
    [ "$LEFT" -ge "$MIN_REMAINING_SECONDS" ] || break

    H=$(( LEFT / 3600 )); M=$(( (LEFT % 3600) / 60 ))
    CONTINUATION_PROMPT="You still have ${H}h ${M}m remaining. Please continue improving your result and maximize performance."

    printf '%s' "$CONTINUATION_PROMPT" | claude --print --verbose --continue --model "$AGENT_CONFIG" \
        --output-format stream-json --thinking-display summarized \
        --dangerously-skip-permissions
done
