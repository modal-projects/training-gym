#!/usr/bin/env python3
"""Learning Agent user-simulator service — the ENVIRONMENT's half of a conversational task.

In τ²-style tasks the customer is not the contestant's policy, it is part of
the environment: the same simulator must drive dev-time iteration and official
scoring, or the two numbers are not comparable. But an agent workspace must
never hold a frontier API key (learning_agent.md rule 6: no external LLM in the
answer path, and the student must be the base model or a fine-tune of it).

This service resolves that: the OPERATOR runs it, it holds the key, and it
serves ONE pinned simulator model over an OpenAI-compatible endpoint. A
workspace gets a URL and a scoped, revocable token — never an OpenAI key — so
`user_llm: self` (a weaker, non-comparable stand-in) is no longer needed.

    modal secret create lab-user-sim-secrets \
        OPENAI_API_KEY=<key> LEARNING_AGENT_USER_SIM_TOKEN=<token>
    modal app stop lab-user-sim --yes      # see REVISION note below
    modal deploy harness/user_sim.py       # -> https://<...>-user-sim.modal.run
    LEARNING_AGENT_USER_SIM_URL=<that>/v1  LEARNING_AGENT_USER_SIM_TOKEN=<token>

Redeploy gotcha: `modal deploy` alone leaves a warm container serving the OLD
code, so a code change can appear to have no effect. `modal app stop
lab-user-sim --yes` first, then check `/health` reports the REVISION you just
bumped before trusting a test result.

What it enforces:
  - MODEL PINNING: whatever model the caller names is ignored; every request is
    served by USER_SIM_MODEL. The simulator cannot be swapped per run.
  - AUTH: a shared LEARNING_AGENT_USER_SIM_TOKEN (rotate it to cut off a workspace).
  - BUDGET: per-session completion-token cap, so the endpoint cannot be farmed
    for large-scale distillation even though it speaks plain chat-completions.
  - AUDIT: every call logs session, tokens, and a prompt fingerprint.

Residual risk, stated plainly: this is a chat-completions endpoint, so a
determined agent can spend its session budget on something other than user
simulation. The budget bounds the blast radius and the log makes it visible;
the deterministic trace audit (agents/lib/audit_trace.py) is the backstop.
"""
# NB: no `from __future__ import annotations` here. FastAPI resolves route
# annotations through the MODULE globals, and fastapi is imported inside web()
# (it exists only in the container image) — stringified annotations would fail
# to resolve and every route would read `request` as a missing query param.
import json
import os
import time
import urllib.error
import urllib.request

import modal

app = modal.App("lab-user-sim")

# The pinned simulator. Changing this changes every conversational task's
# environment — re-measure the floors and re-freeze deliberately.
USER_SIM_MODEL = "gpt-5.6-luna"
UPSTREAM = "https://api.openai.com/v1/chat/completions"
REVISION = 2   # bump on deploy so /health proves which code is live

# Per-session completion-token budget. A full tau2_airline dev rollout
# (30 scenarios x 4 trials x <=200 steps) uses far less than this; it is a
# distillation ceiling, not a working limit.
SESSION_TOKEN_BUDGET = int(os.environ.get("LEARNING_AGENT_USER_SIM_BUDGET", "2000000"))

BUDGETS = modal.Dict.from_name("lab-user-sim-budgets", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


@app.function(image=image, timeout=15 * 60, min_containers=0,
              secrets=[modal.Secret.from_name("lab-user-sim-secrets")])
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Header, HTTPException, Request

    api = FastAPI(title="Learning Agent user simulator")

    def _authorize(authorization: str | None) -> None:
        want = os.environ.get("LEARNING_AGENT_USER_SIM_TOKEN", "")
        if not want:
            raise HTTPException(500, "service misconfigured: no LEARNING_AGENT_USER_SIM_TOKEN")
        got = (authorization or "").removeprefix("Bearer ").strip()
        if got != want:
            raise HTTPException(401, "bad or missing LEARNING_AGENT_USER_SIM_TOKEN")

    def _charge(session: str, tokens: int) -> None:
        spent = BUDGETS.get(session, 0) + tokens
        BUDGETS[session] = spent
        if spent > SESSION_TOKEN_BUDGET:
            raise HTTPException(
                429, f"session {session!r} exhausted its user-simulator budget "
                     f"({spent} > {SESSION_TOKEN_BUDGET} completion tokens). "
                     "This endpoint simulates the customer; it is not a general "
                     "text-generation service.")

    async def _complete(request: Request, authorization: str | None,
                        session: str | None):
        _authorize(authorization)
        body = await request.json()
        session = session or body.get("user") or "anonymous"
        # MODEL PINNING: the caller does not get to choose the simulator.
        asked = body.get("model")
        body["model"] = USER_SIM_MODEL
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
            raise HTTPException(e.code, f"upstream {e.code}: {detail}") from e

        usage = payload.get("usage") or {}
        out_tokens = int(usage.get("completion_tokens", 0))
        _charge(session, out_tokens)
        # AUDIT: session, cost, and a fingerprint of what was asked (never the
        # full prompt — transcripts already live in the run's episode files).
        msgs = body.get("messages") or []
        last = (msgs[-1].get("content") if msgs else "") or ""
        print(json.dumps({
            "session": session, "asked_model": asked, "served": USER_SIM_MODEL,
            "completion_tokens": out_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "n_messages": len(msgs), "last_msg_chars": len(str(last)),
            "tools_in_request": bool(body.get("tools")),
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
        """litellm/OpenAI clients probe this; advertise ONLY the pinned model."""
        return {"object": "list",
                "data": [{"id": USER_SIM_MODEL, "object": "model",
                          "owned_by": "lab-user-sim"}]}

    @api.get("/health")
    async def health():
        return {"ok": True, "model": USER_SIM_MODEL,
                "session_token_budget": SESSION_TOKEN_BUDGET,
                "revision": REVISION}

    return api


@app.local_entrypoint()
def budgets(session: str = "", reset: bool = False):
    """Inspect (or clear) per-session user-simulator spend."""
    if session and reset:
        BUDGETS.pop(session, None)
        print(f"[user-sim] cleared budget for {session!r}")
        return
    rows = [(k, BUDGETS[k]) for k in BUDGETS.keys()]
    if session:
        rows = [r for r in rows if r[0] == session]
    if not rows:
        print("[user-sim] no session spend recorded")
        return
    for k, v in sorted(rows, key=lambda r: -r[1]):
        print(f"  {v:>12,} completion tokens   {k}")
