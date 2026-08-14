#!/bin/bash
# Launch OpenCode as a Learning Agent contestant (mirrors PTB's opencode). Reads $PROMPT and
# $AGENT_CONFIG (provider/model, e.g. anthropic/claude-opus-4-6 or zai/glm-5) from the
# environment, streams JSON events to stdout for trace capture.
#
# OpenCode needs a config file for auto-approval and provider setup; we write it to the
# working directory (repo root — gitignored via opencode.json).

cat > opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "anthropic": {
      "options": {
        "apiKey": "{env:ANTHROPIC_API_KEY}"
      }
    },
    "openai": {
      "options": {
        "apiKey": "{env:OPENAI_API_KEY}"
      }
    },
    "opencode": {
      "options": {
        "apiKey": "{env:OPENCODE_API_KEY}"
      }
    },
    "zai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Z.AI",
      "options": {
        "baseURL": "https://api.z.ai/api/paas/v4",
        "apiKey": "{env:ZAI_API_KEY}"
      },
      "models": {
        "glm-5": {
          "name": "GLM-5"
        },
        "glm-4.7": {
          "name": "GLM-4.7"
        }
      }
    }
  }
}
EOF

printf '%s' "$PROMPT" | opencode run --model "$AGENT_CONFIG" --format json
