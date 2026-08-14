#!/bin/bash
# codex_non_api at effort=high with a re-prompt loop (mirrors PTB's
# codex_non_api_high_reprompt). Auth setup as in codex_non_api: put auth.json in this
# directory (gitignored). Run from the repo root by run.sh (timer.sh lives there).
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

export CODEX_API_KEY=""
export OPENAI_API_KEY=""

SCAFFOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTH_FILE="$SCAFFOLD_DIR/auth.json"
if [ ! -f "$AUTH_FILE" ]; then
    echo "ERROR: no auth.json at $AUTH_FILE (codex login --device-auth, then copy ~/.codex/auth.json)" >&2
    exit 1
fi

export CODEX_HOME="$(mktemp -d)"
cp "$AUTH_FILE" "$CODEX_HOME/auth.json"
printf 'forced_login_method = "chatgpt"\n' > "$CODEX_HOME/config.toml"

MIN_REMAINING_SECONDS=1800

printf '%s' "$PROMPT" | codex --search exec --json \
    -c model_reasoning_effort=high \
    -c model_reasoning_summary=detailed \
    --skip-git-repo-check \
    --dangerously-bypass-approvals-and-sandbox \
    --model "$AGENT_CONFIG"

while true; do
    LEFT=$(bash timer.sh 2>/dev/null | sed -n 's/^seconds_left=//p')
    [ -n "$LEFT" ] || break
    [ "$LEFT" -ge "$MIN_REMAINING_SECONDS" ] || break

    H=$(( LEFT / 3600 )); M=$(( (LEFT % 3600) / 60 ))
    CONTINUATION_PROMPT="You still have ${H}h ${M}m remaining. Please continue improving your result and maximize performance."

    printf '%s' "$CONTINUATION_PROMPT" | codex --search exec resume --last --json \
        -c model_reasoning_effort=high \
        -c model_reasoning_summary=detailed \
        --skip-git-repo-check \
        --dangerously-bypass-approvals-and-sandbox \
        --model "$AGENT_CONFIG" -
done
