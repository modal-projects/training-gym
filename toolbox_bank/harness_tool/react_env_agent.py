#!/usr/bin/env python3
"""react_env_agent.py — the DEFAULT policy driver for interactive text envs
(ALFWorld, WebShop, and anything with a {observation -> action} step loop).

Same ReAct line grammar as the QA search loop in harness/eval.py — one
convention repo-wide. Per turn the model thinks briefly, then emits EXACTLY one:

    ACTION: <one environment action>       e.g.  ACTION: go to shelf 1
                                                 ACTION: search[red running shoes]
    FINAL: <one-line reason>               give up / declare nothing left to do

The loop ends when the ENVIRONMENT reports done (success is the env's call,
never the model's), on FINAL, or when the step budget runs out.

Everything here is the agent's to modify — better prompts, action validation,
self-reflection turns. The contract that must survive:
`run_react(instruction, client, execute)` returns the result dict below and
`execute` is the ONLY way actions reach the environment (the operator's
post-eval injects an executor wired to the real env via the task's adapter;
swapping in your own env locally never touches the scored one).

    execute(action: str) -> (observation: str, done: bool)

Result dict:
    {"done": bool,            # env reached a terminal state (NOT model opinion)
     "steps": int,            # env actions actually executed
     "summary": str,
     "actions": [{"action": str, "obs_tail": str}, ...]}

Reward is deliberately absent: the task's verifier/adapter computes it from the
env after the episode — the policy never grades itself.

Offline smoke (no model, no network — scripted client + fake env):
    python toolbox/agentic_toolbox/react_env_agent.py --self-test
"""
from __future__ import annotations

import re

SYSTEM_TEMPLATE = """You are an agent acting in an interactive text environment.

TASK:
{instruction}

Rules:
- Each turn, think briefly, then emit EXACTLY ONE action line:
ACTION: <one environment action>
- You will see the environment's observation next turn. Valid actions are
  described by the task and the observations; keep actions short and literal.
- Only if you are certain no action can make further progress, reply instead:
FINAL: <one-line reason>
- You have {max_steps} steps total. Explore efficiently."""

_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(.*)$", re.MULTILINE)
_FINAL_RE = re.compile(r"^\s*FINAL\s*:\s*(.*)$", re.MULTILINE)

# Observation cap per turn: environments are chatty (room descriptions, search
# result pages); keep the task model's context from flooding.
OBS_TAIL_CHARS = 2000


def parse_react(reply: str) -> tuple[str | None, str | None]:
    """-> (action, final): at most one is non-None; (None, None) = malformed.
    Whichever directive appears FIRST wins — mirrors harness/eval.py's QA loop,
    so a model that thinks out loud about FINAL after acting still acts."""
    am = _ACTION_RE.search(reply or "")
    fm = _FINAL_RE.search(reply or "")
    if am and (not fm or am.start() < fm.start()):
        act = am.group(1).strip()
        return (act or None), None
    if fm:
        return None, fm.group(1).strip()
    return None, None


def run_react(instruction: str, client, execute, max_steps: int = 40,
              temperature: float = 0.0, max_tokens: int = 2048,
              system_template: str = SYSTEM_TEMPLATE, log=lambda s: None) -> dict:
    """The loop. `client` needs .chat(messages, temperature=, max_tokens=) -> str;
    `execute` is (action) -> (observation, done)."""
    messages = [{"role": "system",
                 "content": system_template.format(instruction=instruction.strip(),
                                                   max_steps=max_steps)},
                {"role": "user", "content": "Begin. Emit your first ACTION."}]
    actions: list[dict] = []
    steps = 0
    # budget counts MODEL CALLS (like mini_swe's turns), not executed actions —
    # a model that never emits a parseable directive must still terminate
    for turn in range(1, max_steps + 1):
        reply = client.chat(messages, temperature=temperature, max_tokens=max_tokens) or ""
        messages.append({"role": "assistant", "content": reply})
        action, final = parse_react(reply)
        if final is not None:
            log(f"[react-env] FINAL after {steps} step(s): {final}")
            return {"done": False, "steps": steps,
                    "summary": f"agent stopped: {final}", "actions": actions}
        if action is None:
            messages.append({"role": "user", "content":
                             "Malformed turn: emit ONE `ACTION: <action>` line, "
                             "or `FINAL: <reason>` to stop."})
            continue
        obs, done = execute(action)
        steps += 1
        tail = (obs or "")[-OBS_TAIL_CHARS:]
        actions.append({"action": action, "obs_tail": tail})
        log(f"[react-env] step {steps}: {action}  (done={bool(done)})")
        if done:
            return {"done": True, "steps": steps,
                    "summary": "environment reached a terminal state",
                    "actions": actions}
        messages.append({"role": "user", "content":
                         f"OBSERVATION:\n{tail}\n\n"
                         f"Steps left: {max_steps - turn}. Next ACTION (or FINAL:)."})
    return {"done": False, "steps": steps,
            "summary": "step budget exhausted", "actions": actions}


# ----------------------------- self-test / CLI ----------------------------- #

class _ScriptedClient:
    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, messages, **_):
        return self._replies.pop(0) if self._replies else "FINAL: out of script"


class _FakeEnv:
    """Two-room toy: the episode is done after 'take key' in room B."""

    def __init__(self):
        self.room, self.done = "A", False

    def __call__(self, action: str) -> tuple[str, bool]:
        if action == "go to room B":
            self.room = "B"
            return "Room B. A key lies on the table.", False
        if action == "take key" and self.room == "B":
            self.done = True
            return "You take the key. Task complete.", True
        return "Nothing happens.", False


def _self_test() -> int:
    env = _FakeEnv()
    client = _ScriptedClient([
        "I should explore.\nACTION: go to room B",
        "no directive this turn (malformed on purpose)",
        "Take it.\nACTION: take key",
    ])
    res = run_react("find the key", client, env, max_steps=5, log=print)
    ok = (res["done"] and res["steps"] == 2 and len(res["actions"]) == 2
          and res["actions"][0]["action"] == "go to room B" and env.done)
    # grammar edge: FINAL before ACTION in one reply -> FINAL wins (QA parity)
    ok &= parse_react("FINAL: stuck\nACTION: flail") == (None, "stuck")
    ok &= parse_react("ACTION: push\nFINAL: later") == ("push", None)
    print(f"[self-test] {'PASS' if ok else 'FAIL'}: {res['summary']}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ReAct driver for interactive text envs.")
    ap.add_argument("--self-test", action="store_true",
                    help="offline scripted smoke (no model, no network)")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    ap.error("only --self-test runs standalone; in real use the task's adapter "
             "wires run_react(instruction, client, execute) to a live env")
