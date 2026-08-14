#!/usr/bin/env python3
"""react_tool_agent.py — NATIVE OpenAI tool-calling policies.

This file owns the tool-call wire format; the other drivers own text protocols
(react_env_agent.py: `ACTION:` lines; mini_swe_agent.py: fenced bash). Two
entry points, one per shape of task:

    agent_turn(base_url, model, messages, tools, execute_tool) -> (text, messages)
        ONE assistant turn of a CONVERSATION (tau2-style): call the model,
        execute whatever tools it requests, return its user-facing reply. The
        conversation itself (user simulator, termination, reward) belongs to
        the environment's own runner, so this composes with tau2's native
        orchestration or any custom loop.

    run_tools(instruction, base_url, model, tools, execute_tool, ...) -> dict
        A full EPISODE in an interactive environment (ALFWorld-style): loop
        {model -> tool call -> env step -> observation} until the environment
        reports done or the step budget runs out. Same result dict as
        react_env_agent.run_react, so adapters can swap drivers by config.

Both talk raw HTTP to an OpenAI-compatible /chat/completions endpoint (stdlib
only — OAIClient doesn't forward `tools`, and the wire format is the whole
point). `execute_tool(name, args_dict)` is the ONLY bridge to the environment.
The SERVER must be launched with tool-call parsing enabled (e.g. vLLM
--enable-auto-tool-choice --tool-call-parser, sglang --tool-call-parser),
otherwise the model's calls come back as prose and never reach the env.

Offline smoke (no model, no network):
    python toolbox/agentic_toolbox/react_tool_agent.py --self-test
"""
from __future__ import annotations

import argparse
import json
import urllib.request

# A model that loops tool calls without ever addressing the user is a failure
# mode, not a strategy: cap tool iterations per assistant turn.
MAX_TOOL_ITERS = 8


def _post_chat(base_url: str, body: dict, timeout: int = 300,
               api_key: str = "EMPTY") -> dict:
    """POST one chat completion. The Authorization header is always sent:
    vLLM/SGLang ignore it when the server has no key configured, but a server
    started WITH `--api-key` 401s without it."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {api_key}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def agent_turn(base_url: str, model: str, messages: list[dict], tools: list[dict],
               execute_tool, temperature: float = 0.0, max_tokens: int = 2048,
               max_tool_iters: int = MAX_TOOL_ITERS, api_key: str = "EMPTY",
               _post=_post_chat) -> tuple[str, list[dict]]:
    """One agent turn: call the model; while it requests tools, execute them and
    feed results back; return (final assistant text, updated messages).
    `_post` is injectable for offline tests."""
    messages = list(messages)
    for _ in range(max_tool_iters):
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        msg = _post(base_url, body, api_key=api_key)["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            text = msg.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages
        messages.append({"role": "assistant", "content": msg.get("content"),
                         "tool_calls": calls})
        for c in calls:
            fn = (c.get("function") or {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = execute_tool(fn.get("name", ""), args)
            except Exception as e:  # noqa: BLE001 — env errors are observations
                result = f"[tool error] {type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                             "content": str(result)})
    # Out of tool iterations: force a user-facing reply from what it has.
    messages.append({"role": "assistant",
                     "content": "[agent] tool-iteration budget exhausted"})
    return "[agent] tool-iteration budget exhausted", messages


# --------------------- env episodes (ALFWorld-style) ----------------------- #

ENV_SYSTEM_TEMPLATE = """You are an agent acting in an interactive environment.

TASK:
{instruction}

Rules:
- Each turn, call EXACTLY ONE tool to act in the environment.
- The tool's output is what the environment reports back; read it before acting again.
- Only if you are certain no tool can make further progress, reply with plain
  text instead of calling a tool — that ends the episode.
- You have {max_steps} steps total. Act efficiently."""

# Per-step observation cap: environments are chatty; keep the context bounded.
OBS_TAIL_CHARS = 2000


def run_tools(instruction: str, base_url: str, model: str, tools: list[dict],
              execute_tool, max_steps: int = 40, temperature: float = 0.0,
              max_tokens: int = 2048, system_template: str = ENV_SYSTEM_TEMPLATE,
              extra_body: dict | None = None, api_key: str = "EMPTY",
              log=lambda s: None, _post=_post_chat) -> dict:
    """One EPISODE driven by native tool calls.

    `execute_tool(name, args_dict) -> (observation, done)` is the only bridge to
    the environment — note this returns a (obs, done) PAIR, matching
    react_env_agent's `execute`, because an environment (not the model) decides
    when an episode is over. Returns run_react's result shape:

        {"done": bool, "steps": int, "summary": str,
         "actions": [{"tool": str, "args": dict, "obs_tail": str}, ...]}

    A model reply with NO tool calls ends the episode (the model is stopping).
    If that happens on every turn, the served endpoint most likely lacks a
    tool-call parser — the summary says so rather than silently scoring 0.
    """
    messages = [{"role": "system",
                 "content": system_template.format(instruction=instruction.strip(),
                                                   max_steps=max_steps)},
                {"role": "user", "content": "Begin. Call your first tool."}]
    actions: list[dict] = []
    steps = 0
    calls_seen = 0
    for _turn in range(1, max_steps + 1):
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "tools": tools}
        if extra_body:
            body.update(extra_body)
        msg = _post(base_url, body, api_key=api_key)["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            text = (msg.get("content") or "").strip()
            note = ("" if calls_seen else
                    " (NO tool call was made all episode — is the endpoint "
                    "served with a tool-call parser?)")
            log(f"[tool-env] agent stopped after {steps} step(s){note}")
            return {"done": False, "steps": steps,
                    "summary": f"agent stopped: {text[:200]}{note}",
                    "actions": actions}
        messages.append({"role": "assistant", "content": msg.get("content"),
                         "tool_calls": calls})
        finished = False
        # The OpenAI protocol requires a tool message per tool_call_id, so every
        # call gets a reply even after the episode ends mid-batch.
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if finished or steps >= max_steps:
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                 "content": "[episode already ended]"})
                continue
            calls_seen += 1
            try:
                obs, done = execute_tool(name, args)
            except Exception as e:  # noqa: BLE001 — env errors are observations
                obs, done = f"[tool error] {type(e).__name__}: {e}", False
            steps += 1
            tail = str(obs)[-OBS_TAIL_CHARS:]
            actions.append({"tool": name, "args": args, "obs_tail": tail})
            log(f"[tool-env] step {steps}: {name}({args})  (done={bool(done)})")
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                             "content": tail})
            if done:
                finished = True
        if finished:
            return {"done": True, "steps": steps,
                    "summary": "environment reached a terminal state",
                    "actions": actions}
        if steps >= max_steps:
            break
    return {"done": False, "steps": steps,
            "summary": "step budget exhausted", "actions": actions}


# ----------------------------- self-test ----------------------------------- #

def _self_test() -> int:
    """Scripted server: first reply requests a tool, second returns prose."""
    db = {"EHGLP3": {"status": "confirmed"}}
    scripted = [
        {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "function": {"name": "get_reservation",
                                      "arguments": json.dumps({"id": "EHGLP3"})}}]}}]},
        {"choices": [{"message": {"content":
            "Your reservation EHGLP3 is confirmed."}}]},
    ]
    seen = []

    def fake_post(base_url, body):
        seen.append(body)
        return scripted.pop(0)

    def execute_tool(name, args):
        assert name == "get_reservation", name
        return json.dumps(db[args["id"]])

    text, msgs = agent_turn("http://mock", "task model",
                            [{"role": "user", "content": "Check EHGLP3?"}],
                            tools=[{"type": "function",
                                    "function": {"name": "get_reservation",
                                                 "parameters": {"type": "object"}}}],
                            execute_tool=execute_tool, _post=fake_post)
    ok = ("confirmed" in text
          and any(m.get("role") == "tool" for m in msgs)
          and len(seen) == 2
          and "tools" in seen[0])

    # run_tools: an env episode over the same wire format
    env = {"at": None}

    def env_tool(name, args):
        if name == "go_to":
            env["at"] = args.get("receptacle")
            return f"You arrive at {env['at']}.", False
        if name == "take" and env["at"] == "drawer 1":
            return "You pick up the key. Task complete.", True
        return "Nothing happens.", False

    ep_replies = [
        {"choices": [{"message": {"tool_calls": [
            {"id": "a", "function": {"name": "go_to",
                                     "arguments": json.dumps({"receptacle": "drawer 1"})}}]}}]},
        {"choices": [{"message": {"tool_calls": [
            {"id": "b", "function": {"name": "take",
                                     "arguments": json.dumps({"object": "key"})}}]}}]},
    ]
    res = run_tools("find the key", "http://mock", "task model",
                    tools=[{"type": "function", "function": {"name": "go_to"}}],
                    execute_tool=env_tool, max_steps=5,
                    _post=lambda u, b, **k: ep_replies.pop(0))
    ok &= (res["done"] and res["steps"] == 2
           and res["actions"][0]["tool"] == "go_to")

    # a server with no tool-call parser -> prose, never a silent 0
    res2 = run_tools("x", "http://mock", "task model", tools=[],
                     execute_tool=lambda n, a: ("", False), max_steps=3,
                     _post=lambda u, b, **k: {"choices": [{"message":
                                                           {"content": "I would go north."}}]})
    ok &= (not res2["done"] and res2["steps"] == 0
           and "tool-call parser" in res2["summary"])

    print(f"[self-test] {'PASS' if ok else 'FAIL'}: {text} | env episode "
          f"done={res['done']} steps={res['steps']}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        raise SystemExit(_self_test())
    ap.error("this module is a library; run --self-test or import agent_turn")
