#!/usr/bin/env python3
"""Learning Agent judge service — the pinned rubric judge, served to agent workspaces.

The benchmark's one rule about measurement: every number that gets compared —
the agent's dev-set evals, its intermediate-checkpoint evals during training,
and the operator's official scoring — must come from the SAME judge
(bench/config.yaml judge.model, canonical backend), or the numbers are not
comparable. But an agent workspace must never hold a frontier API key
(learning_agent.md rule 6), so it cannot call api.openai.com itself.

This service resolves that the same way harness/user_sim.py does for the
customer simulator: the OPERATOR runs it, it holds the key, and it serves ONE
pinned judge model over an OpenAI-compatible endpoint. A workspace gets a URL
and a scoped, revocable token — never an OpenAI key. The toolbox judge
(toolbox/api_clients/judge_client.py, backend "openai") sends its
forced-structured-output judging calls here, so the agent's dev-time verdicts
are produced by the identical model + wire shape the official judge_cli uses.

    modal secret create lab-judge-secrets \
        OPENAI_API_KEY=<key> LEARNING_AGENT_JUDGE_TOKEN=<token>
    modal app stop lab-judge --yes         # see REVISION note below
    modal deploy harness/judge_service.py  # -> https://<...>-lab-judge-web.modal.run
    LEARNING_AGENT_JUDGE_URL=<that>/v1  LEARNING_AGENT_JUDGE_TOKEN=<token>

Redeploy gotcha: `modal deploy` alone leaves a warm container serving the OLD
code. `modal app stop lab-judge --yes` first, then check `/health` reports the
REVISION you just bumped.

What it enforces:
  - MODEL PINNING: whatever model the caller names is ignored; every request is
    served by JUDGE_MODEL (keep it equal to bench/config.yaml judge.model — the
    integrity pin on that file is what makes drift a deliberate act).
  - AUTH: a shared LEARNING_AGENT_JUDGE_TOKEN (rotate it to cut off a workspace).
  - BUDGET: per-session completion-token cap. Judge verdicts are small; the cap
    exists so the endpoint cannot be farmed as a general frontier-model API.
  - AUDIT: every call logs session, tokens, and a fingerprint of the request.

Residual risk, stated plainly: this is a chat-completions endpoint, so a
determined agent can spend its judge budget on something other than judging.
The budget bounds the blast radius, the log makes it visible, and the
deterministic trace audit (agents/lib/audit_trace.py) is the backstop — the
answer-path rule (no external LLM at answer time) is unchanged and audited.
"""
# NB: no `from __future__ import annotations` here — FastAPI resolves route
# annotations through module globals and fastapi only exists in the container
# image (same constraint as harness/user_sim.py).
import json
import os
import time
import urllib.error
import urllib.request

import modal

app = modal.App("lab-judge")

# The pinned judge. MUST match bench/config.yaml judge.model — changing either
# changes every score in the benchmark; re-measure floors and re-freeze
# deliberately.
JUDGE_MODEL = "gpt-5.6-luna"
UPSTREAM = "https://api.openai.com/v1/chat/completions"
REVISION = 1   # bump on deploy so /health proves which code is live

# Per-session completion-token budget. A full dev-set eval (50 questions x
# 3 votes, verdicts of a few hundred tokens each — reasoning models burn
# hidden-reasoning tokens too) uses low single-digit millions; this is a
# farming ceiling, not a working limit.
SESSION_TOKEN_BUDGET = int(os.environ.get("LEARNING_AGENT_JUDGE_BUDGET", "20000000"))

BUDGETS = modal.Dict.from_name("lab-judge-budgets", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


@app.function(image=image, timeout=15 * 60, min_containers=0,
              secrets=[modal.Secret.from_name("lab-judge-secrets")])
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Header, HTTPException, Request

    api = FastAPI(title="Learning Agent judge service")

    def _authorize(authorization: str | None) -> None:
        want = os.environ.get("LEARNING_AGENT_JUDGE_TOKEN", "")
        if not want:
            raise HTTPException(500, "service misconfigured: no LEARNING_AGENT_JUDGE_TOKEN")
        got = (authorization or "").removeprefix("Bearer ").strip()
        if got != want:
            raise HTTPException(401, "bad or missing LEARNING_AGENT_JUDGE_TOKEN")

    def _charge(session: str, tokens: int) -> None:
        spent = BUDGETS.get(session, 0) + tokens
        BUDGETS[session] = spent
        if spent > SESSION_TOKEN_BUDGET:
            raise HTTPException(
                429, f"session {session!r} exhausted its judge budget "
                     f"({spent} > {SESSION_TOKEN_BUDGET} completion tokens). "
                     "This endpoint grades rubric verdicts; it is not a general "
                     "text-generation service.")

    async def _complete(request: Request, authorization: str | None,
                        session: str | None):
        _authorize(authorization)
        body = await request.json()
        session = session or body.get("user") or "anonymous"
        # MODEL PINNING: the caller does not get to choose the judge.
        asked = body.get("model")
        body["model"] = JUDGE_MODEL
        body.pop("user", None)

        req = urllib.request.Request(
            UPSTREAM, data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            # Pass the upstream status through untouched: the client's
            # shape-adaptation loop (max_tokens vs max_completion_tokens,
            # response_format vs function tool) keys off 400 bodies.
            raise HTTPException(e.code, f"upstream {e.code}: {detail}") from e

        usage = payload.get("usage") or {}
        out_tokens = int(usage.get("completion_tokens", 0))
        _charge(session, out_tokens)
        # AUDIT: session, cost, and a fingerprint of what was asked (never the
        # full prompt — the run's own eval outputs hold the transcripts).
        msgs = body.get("messages") or []
        last = (msgs[-1].get("content") if msgs else "") or ""
        print(json.dumps({
            "session": session, "asked_model": asked, "served": JUDGE_MODEL,
            "completion_tokens": out_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "n_messages": len(msgs), "last_msg_chars": len(str(last)),
            "structured": bool(body.get("response_format") or body.get("tools")),
            "seconds": round(time.time() - started, 2),
            "session_spent": BUDGETS.get(session, 0),
        }), flush=True)
        return payload

    @api.post("/v1/chat/completions")
    async def v1_chat(request: Request,
                      authorization: str | None = Header(default=None),
                      x_lab_session: str | None = Header(default=None)):
        return await _complete(request, authorization, x_lab_session)

    @api.post("/chat/completions")
    async def chat(request: Request,
                   authorization: str | None = Header(default=None),
                   x_lab_session: str | None = Header(default=None)):
        return await _complete(request, authorization, x_lab_session)

    @api.get("/v1/models")
    async def models():
        """OpenAI clients probe this; advertise ONLY the pinned judge."""
        return {"object": "list",
                "data": [{"id": JUDGE_MODEL, "object": "model",
                          "owned_by": "lab-judge"}]}

    @api.get("/health")
    async def health():
        return {"ok": True, "model": JUDGE_MODEL,
                "session_token_budget": SESSION_TOKEN_BUDGET,
                "revision": REVISION}

    return api


@app.local_entrypoint()
def budgets(session: str = "", reset: bool = False):
    """Inspect (or clear) per-session judge spend."""
    if session and reset:
        BUDGETS.pop(session, None)
        print(f"[judge] cleared budget for {session!r}")
        return
    rows = [(k, BUDGETS[k]) for k in BUDGETS.keys()]
    if session:
        rows = [r for r in rows if r[0] == session]
    if not rows:
        print("[judge] no session spend recorded")
        return
    for k, v in sorted(rows, key=lambda r: -r[1]):
        print(f"  {v:>12,} completion tokens   {k}")
