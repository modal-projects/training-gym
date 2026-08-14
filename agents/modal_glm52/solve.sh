#!/bin/bash
# GLM-5.2 contestant via OpenCode pointed at the team's own Modal SGLang endpoint
# (lab-dev env, app ep-glm-5-2-fp8 — see dev/MODAL.md). The endpoint's
# Anthropic /v1/messages emulation is partial (rejects Claude Code's wire format),
# so this scaffold speaks the OpenAI-compatible API through OpenCode. The endpoint
# is public and unauthenticated — the apiKey below is a placeholder, not a secret.
#
# Re-prompt loop (mirrors claude_reprompt): if the session ends early, continue the
# same session with the time remaining until fewer than MIN_REMAINING_SECONDS are
# left. First live run quit cleanly at 7 min of a 24 h budget — GLM-5.2 stopping
# mid-debug is the dominant failure mode, and the loop is what counters it.
# No `set -e`: a nonzero opencode exit must fall through to the re-prompt loop.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/agents/lib/glm52_endpoint.env"

cat > "$ROOT/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "modal-glm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Modal GLM-5.2",
      "options": {
        "baseURL": "${MODAL_GLM52_BASE_URL}/v1",
        "apiKey": "public-endpoint"
      },
      "models": {
        "${MODAL_GLM52_MODEL}": {
          "name": "GLM-5.2 FP8",
          "limit": {
            "context": 1048576,
            "output": 131072
          }
        }
      }
    }
  }
}
EOF

# 30-min floor: do not resume a finished session with less than this left. A fresh
# continuation cannot meaningfully improve the result before the hard deadline, and
# re-warming the session would consume most of the remaining window.
MIN_REMAINING_SECONDS=1800

printf '%s' "$PROMPT" |
  opencode run --model "$AGENT_CONFIG" --format json

# Re-prompt loop: parse the machine-readable seconds_left= line (portable — no GNU grep).
while true; do
    LEFT=$(bash timer.sh 2>/dev/null | sed -n 's/^seconds_left=//p')
    [ -n "$LEFT" ] || break
    [ "$LEFT" -ge "$MIN_REMAINING_SECONDS" ] || break

    H=$(( LEFT / 3600 )); M=$(( (LEFT % 3600) / 60 ))
    CONTINUATION_PROMPT="You still have ${H}h ${M}m remaining. Please continue improving your result and maximize performance."

    printf '%s' "$CONTINUATION_PROMPT" |
      opencode run --continue --model "$AGENT_CONFIG" --format json
done
