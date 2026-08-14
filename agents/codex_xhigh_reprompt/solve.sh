#!/bin/bash
# Codex CLI contestant at effort=xhigh with a re-prompt loop (mirrors PTB's
# codex_xhigh_reprompt): if the agent finishes early, resume the last session with the
# time remaining until fewer than MIN_REMAINING_SECONDS are left. Run from the repo
# root by run.sh (timer.sh lives there).
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

CODEX_ARGS=(--search exec --json
    -c model_reasoning_effort=xhigh
    -c model_reasoning_summary=detailed
    --skip-git-repo-check
    --dangerously-bypass-approvals-and-sandbox
    --model "$AGENT_CONFIG")

MIN_REMAINING_SECONDS=1800

printf '%s' "$PROMPT" | codex "${CODEX_ARGS[@]}"

# Re-prompt loop: parse the machine-readable seconds_left= line (portable — no GNU grep).
while true; do
    LEFT=$(bash timer.sh 2>/dev/null | sed -n 's/^seconds_left=//p')
    [ -n "$LEFT" ] || break
    [ "$LEFT" -ge "$MIN_REMAINING_SECONDS" ] || break

    H=$(( LEFT / 3600 )); M=$(( (LEFT % 3600) / 60 ))
    CONTINUATION_PROMPT="You still have ${H}h ${M}m remaining. Please continue improving your result and maximize performance."

    printf '%s' "$CONTINUATION_PROMPT" | codex --search exec resume --last --json \
        -c model_reasoning_effort=xhigh \
        -c model_reasoning_summary=detailed \
        --skip-git-repo-check \
        --dangerously-bypass-approvals-and-sandbox \
        --model "$AGENT_CONFIG" -
done
