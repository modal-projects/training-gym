#!/bin/bash
# Kimi K3 contestant via Codex CLI — the default learning-agent scaffold
# (2026-08-12), driving the team endpoint (agents/lib/kimi_k3_endpoint.env —
# one pinned source of truth) through the in-repo Responses->Chat shim.
# Isolated CODEX_HOME (house pattern from codex_non_api*) so the operator's
# real ~/.codex is never touched; re-prompt loop mirrors codex_xhigh_reprompt.
# No --search: endpoint runs stay offline for parity with the modal_glm52
# (OpenCode) scaffold in harness A/B comparisons.
#
# Known endpoint caveat (applies to the OpenCode scaffold too): the server's
# reasoning parser strips literal <think>...</think> spans out of tool-call
# arguments, so file contents containing those tags arrive without them. The
# model can route around it (escaping/base64); smoke-verified 2026-07-31.
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY
unset OPENAI_API_KEY

# The scaffold was verified against codex-cli 0.145.0 (also the version pinned
# in the Modal/Docker runner images). A different version is not fatal, but
# name it in the log so a behavior change is attributable.
CODEX_VERSION="$(codex --version 2>/dev/null || echo 'MISSING')"
case "$CODEX_VERSION" in
    *0.145.*) ;;
    *) echo "[codex_kimi3] WARNING: codex version '$CODEX_VERSION' != verified 0.145.x" >&2 ;;
esac

SCAFFOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCAFFOLD_DIR/../lib/kimi_k3_endpoint.env"

# Model id: resolve from the endpoint when not pinned ("auto" is run.sh's
# resolve-at-launch sentinel). Fails loudly when the endpoint is not serving.
if [ -z "${MODAL_KIMI3_MODEL:-}" ]; then
    MODAL_KIMI3_MODEL="$(curl -s -m 30 "$MODAL_KIMI3_BASE_URL/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
fi
if [ -z "$MODAL_KIMI3_MODEL" ]; then
    echo "[codex_kimi3] FATAL: cannot resolve model id — $MODAL_KIMI3_BASE_URL/v1/models not serving" >&2
    exit 1
fi
if [ -z "${AGENT_CONFIG:-}" ] || [ "$AGENT_CONFIG" = "auto" ]; then
    AGENT_CONFIG="$MODAL_KIMI3_MODEL"
fi

# Codex is Responses-only (chat wire hard-removed Feb 2026), and SGLang's
# /v1/responses cannot serve custom function tools (sglang#13292, closed
# stale) — so codex talks to our own in-repo Responses->Chat translator
# (agents/lib/responses_shim.py) on localhost, which speaks the endpoint's
# fully-working chat wire. Self-owned, auditable, endpoint-agnostic.
SHIM_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
SHIM_LOG="${LEARNING_AGENT_RUN_DIR:-${TMPDIR:-/tmp}}/shim.log"
python3 "$SCAFFOLD_DIR/../lib/responses_shim.py" --port "$SHIM_PORT" \
    --upstream "${MODAL_KIMI3_BASE_URL}/v1" >"$SHIM_LOG" 2>&1 &
SHIM_PID=$!
trap 'kill "$SHIM_PID" 2>/dev/null' EXIT
for _ in $(seq 1 20); do
    curl -s -m 2 "http://127.0.0.1:${SHIM_PORT}/healthz" >/dev/null 2>&1 && break
    sleep 0.5
done
# A dead shim means every codex turn fails while codex still exits 0 — fail
# HERE, loudly, instead of recording a 30-second "successful" run.
if ! curl -s -m 2 "http://127.0.0.1:${SHIM_PORT}/healthz" >/dev/null 2>&1; then
    echo "[codex_kimi3] FATAL: responses shim never became healthy on :${SHIM_PORT}" >&2
    tail -20 "$SHIM_LOG" >&2 || true
    exit 1
fi

# One CODEX_HOME per run, alive across the re-prompt loop (resume --last needs
# the session files in it). In containers LEARNING_AGENT_LOGS_DIR is the session's logs
# dir on the volume, so codex's native session state survives the run; local
# dev runs fall back to a temp dir (never the operator's real ~/.codex). The
# endpoint is public/unauthenticated — env_key is a placeholder so the CLI
# has an auth var to read, not a secret.
export CODEX_HOME="${LEARNING_AGENT_LOGS_DIR:+$LEARNING_AGENT_LOGS_DIR/codex_home}"
export CODEX_HOME="${CODEX_HOME:-$(mktemp -d)}"
mkdir -p "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "${MODAL_KIMI3_MODEL}"
model_provider = "modal_kimi"
model_supports_reasoning_summaries = true

[model_providers.modal_kimi]
# Large-model prefill on a long context can exceed codex's default stream
# idle timeout: the server sends nothing until the first token, codex kills
# the stream and retries, and the retry re-queues the same slow prefill
# (the 08-12 "idle timeout waiting for SSE" stalls). 10 minutes of patience.
stream_idle_timeout_ms = 600000
request_max_retries = 4
name = "Modal Kimi K3 (via responses shim)"
base_url = "http://127.0.0.1:${SHIM_PORT}/v1"
wire_api = "responses"
env_key = "MODAL_KIMI_API_KEY"
EOF
export MODAL_KIMI_API_KEY="public-endpoint"

CODEX_ARGS=(exec --json
    --skip-git-repo-check
    --dangerously-bypass-approvals-and-sandbox
    --model "$AGENT_CONFIG")

MIN_REMAINING_SECONDS=1800

# Preemption resume: CODEX_HOME lives in the session's logs dir on the volume,
# so after a container restart the previous conversation's session files are
# still there — continue it instead of starting the mission from scratch.
if find "$CODEX_HOME/sessions" -type f -name '*.jsonl' 2>/dev/null | head -1 | grep -q .; then
    RESUME_PROMPT="Your container was restarted (platform preemption); your workspace and this conversation are intact. Re-check timer.sh for the real remaining budget, then continue from where you left off."
    printf '%s' "$RESUME_PROMPT" | codex exec resume --last --json \
        --skip-git-repo-check \
        --dangerously-bypass-approvals-and-sandbox \
        --model "$AGENT_CONFIG" -
else
    printf '%s' "$PROMPT" | codex "${CODEX_ARGS[@]}"
fi

# Re-prompt loop: parse the machine-readable seconds_left= line (portable — no GNU grep).
while true; do
    LEFT=$(bash timer.sh 2>/dev/null | sed -n 's/^seconds_left=//p')
    [ -n "$LEFT" ] || break
    [ "$LEFT" -ge "$MIN_REMAINING_SECONDS" ] || break

    H=$(( LEFT / 3600 )); M=$(( (LEFT % 3600) / 60 ))
    CONTINUATION_PROMPT="You still have ${H}h ${M}m remaining. Please continue improving your result and maximize performance."

    printf '%s' "$CONTINUATION_PROMPT" | codex exec resume --last --json \
        --skip-git-repo-check \
        --dangerously-bypass-approvals-and-sandbox \
        --model "$AGENT_CONFIG" -
done
