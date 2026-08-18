"""tau2-bench env adapter — the NATIVE orchestrator, batch path (run_split).

tau2 episodes are conversations: the policy talks to a simulated USER while
calling domain tools; reward is tau2's own check (final DB state + required
actions + required communication). Faithfulness means running tau2's runner —
its user simulator, tool loop, and reward — not re-implementing a turn loop,
so this adapter exposes `run_split` (see harness/rollout.py): tau2 handles
trials/concurrency internally; Learning Agent's rollout keeps scoring + artifacts.

Scenario rows (tasks/tau2_<domain>/{dev,test}.json — copied verbatim from the
pinned repo's shipped data/tau2/domains/<domain>/split_tasks.json: dev = its
"train" ids, test = its "test" ids):

    {"id": "<tau2 task id>", "purpose": "<human-readable description>"}

Policy: tau2's llm_agent pointed (via litellm) at the served submission
endpoint — `agent.base_url` / `agent.model` from submission/agent.py build().
Sampling comes from the task.yaml `agent:` block (default = the Qwen3.5
thinking-mode recipe the tau2 leaderboard protocol uses).

User simulator (task.yaml `env.user_llm`): ONE pinned model for dev and for
scoring — the customer is part of the ENVIRONMENT, so a weaker dev stand-in
would make the two numbers incomparable. Agent workspaces reach it through the
operator's user-sim service (harness/user_sim.py) via LEARNING_AGENT_USER_SIM_URL, which
holds the API key and pins the model; no workspace ever carries a frontier
credential. See _user_simulator() below.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def load_split(split_path) -> list[dict]:
    return json.loads(Path(split_path).read_text())


def _termination_done(reason: str) -> bool:
    """A conversation that ENDED (either side stopped) is a completed episode;
    budget/error terminations are not."""
    return reason in ("user_stop", "agent_stop")


def _user_simulator(cfg: dict) -> tuple[str, dict]:
    """Resolve the customer simulator — ONE pinned model everywhere, so dev and
    scored rollouts are comparable (there is no weaker stand-in mode).

    Two ways to reach it, in order:
      1. LEARNING_AGENT_USER_SIM_URL (+ LEARNING_AGENT_USER_SIM_TOKEN) — the operator's user-sim
         service (harness/user_sim.py). It holds the API key and pins the
         model, so an agent workspace runs the REAL simulator while carrying
         no frontier credential of its own. This is the normal path.
      2. OPENAI_API_KEY — direct to the provider, for operator machines.
    """
    env = cfg.get("env") or cfg          # tolerate being handed the env block
    model = env.get("user_llm")
    if not model:
        raise RuntimeError("task.yaml must pin env.user_llm (the simulator model)")
    args = dict(env.get("user_llm_args") or {})
    proxy = os.environ.get("LEARNING_AGENT_USER_SIM_URL", "").rstrip("/")
    if proxy:
        args["api_base"] = proxy
        args["api_key"] = os.environ.get("LEARNING_AGENT_USER_SIM_TOKEN", "") or "unused"
        # Per-session budget accounting on the service side. EVERY run must
        # carry an id: without one they all pool under "anonymous" and would
        # eventually 429 an operator scoring run on an agent's spend.
        session = (cfg.get("_session") or os.environ.get("LEARNING_AGENT_SESSION")
                   or "unattributed")
        args.setdefault("extra_headers", {})["x-lab-session"] = session
        return f"openai/{model}", args
    if os.environ.get("OPENAI_API_KEY"):
        return f"openai/{model}", args
    raise RuntimeError(
        f"no way to reach the {model!r} user simulator: set LEARNING_AGENT_USER_SIM_URL "
        "(+ LEARNING_AGENT_USER_SIM_TOKEN) to the operator's user-sim service "
        "(`modal deploy harness/user_sim.py`), or OPENAI_API_KEY on an "
        "operator machine.")


BRIDGE_AGENT_NAME = "learning_agent_submission"


def _register_bridge(policy) -> str:
    """Register the SUBMISSION as a tau2 agent (task.yaml `agent.driver: bridge`).

    tau2's orchestrator keeps everything that defines the benchmark — user-sim
    turn-taking, tool EXECUTION, termination, and the transcript its evaluator
    reads (action + communication checks) — while each assistant turn is
    computed by the submission's policy_turn(messages, tools) instead of
    tau2's stock litellm call. The bridge translates tau2 messages to OpenAI
    format with tau2's OWN converter and the reply back into tau2's native
    AssistantMessage/ToolCall objects, so the evaluator sees a transcript
    byte-compatible with the stock loop's.

    Registered per-process (run_domain builds one bridge instance per
    simulation via the factory; `policy` is shared — it only POSTs to the
    submission endpoint, so concurrent trials are safe)."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
    from tau2.registry import registry
    from tau2.utils.llm_utils import to_litellm_messages

    class SubmissionBridge(LLMAgent):
        def _generate_next_message(self, message, state):
            # Same state bookkeeping as the stock LLMAgent...
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            oai_msgs = to_litellm_messages(state.system_messages + state.messages)
            oai_tools = [t.openai_schema for t in self.tools]
            # ...but the turn is the SUBMISSION's to compute.
            reply = policy.policy_turn(oai_msgs, oai_tools) or {}
            calls = [
                ToolCall(id=(tc.get("id") or f"call_{i}"),
                         name=tc["function"]["name"],
                         arguments=json.loads(tc["function"]["arguments"] or "{}"))
                for i, tc in enumerate(reply.get("tool_calls") or [])
            ] or None
            return AssistantMessage(role="assistant",
                                    content=reply.get("content"),
                                    tool_calls=calls)

    def factory(tools, domain_policy, llm=None, llm_args=None, **kwargs):
        return SubmissionBridge(tools=tools, domain_policy=domain_policy,
                                llm=llm or "submission", llm_args={})

    if registry.get_agent_factory(BRIDGE_AGENT_NAME) is None:
        registry.register_agent_factory(factory, BRIDGE_AGENT_NAME)
    return BRIDGE_AGENT_NAME


def run_split(agent, rows, cfg: dict) -> dict:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import run_domain

    env = cfg.get("env") or {}
    acfg = cfg.get("agent") or {}
    task_ids = [str(r["id"]) for r in rows]
    k = int(env.get("num_trials", 1))

    # Qwen3.5 thinking-mode recipe (leaderboard protocol defaults; model card):
    # presence_penalty is the documented anti-degeneration knob for thinking mode.
    llm_args_agent = {
        "api_base": agent.base_url,
        "api_key": "local",
        "temperature": float(acfg.get("temperature", 1.0)),
        "top_p": float(acfg.get("top_p", 0.95)),
        "max_tokens": int(acfg.get("max_tokens", 81920)),
        "presence_penalty": float(acfg.get("presence_penalty", 1.5)),
        "extra_body": {"top_k": int(acfg.get("top_k", 20)), "min_p": 0.0,
                       "chat_template_kwargs": {"enable_thinking": True}},
    }
    llm_user, llm_args_user = _user_simulator(cfg)

    # driver: native (default) = tau2's stock llm_agent computes each turn
    # (surface: weights only). driver: bridge = the submission's policy_turn
    # computes each turn inside tau2's orchestrator (surface: + per-turn
    # policy). Flipping this changes the benchmark protocol — re-measure the
    # base floor before comparing across drivers (provenance.driver records it).
    driver = acfg.get("driver", "native")
    if driver == "bridge":
        agent_name = _register_bridge(agent)
    elif driver == "native":
        agent_name = "llm_agent"
    else:
        raise ValueError(f"unknown tau2 driver {driver!r} (native | bridge)")

    save_to = tempfile.mkdtemp(prefix="tau2_rollout_")
    config = TextRunConfig(
        domain=env["domain"],
        task_ids=task_ids,
        agent=agent_name,
        llm_agent=f"openai/{agent.model}",
        llm_args_agent=llm_args_agent,
        user="user_simulator",
        llm_user=llm_user,
        llm_args_user=llm_args_user,
        num_trials=k,
        max_concurrency=int(env.get("max_concurrency", 8)),
        max_steps=int(env.get("max_steps", 200)),
        seed=int(env.get("seed", 300)),
        log_level="ERROR",
        save_to=save_to,
        auto_resume=False,
    )
    results = run_domain(config)

    out = {qid: {"rewards": [], "steps": [], "done": [], "episodes": []}
           for qid in task_ids}
    for sim in results.simulations:
        entry = out.get(str(sim.task_id))
        if entry is None:
            continue
        ri = getattr(sim, "reward_info", None)
        reward = float(getattr(ri, "reward", 0.0) or 0.0)
        reason = str(getattr(sim, "termination_reason", "") or "")
        reason = reason.split(".")[-1].lower()  # enum-or-string tolerant
        msgs = sim.messages or []
        entry["rewards"].append(round(reward, 4))
        entry["steps"].append(len(msgs))
        entry["done"].append(_termination_done(reason))
        entry["episodes"].append({
            "reward": round(reward, 4), "steps": len(msgs),
            "done": _termination_done(reason),
            "transcript": [m.model_dump(mode="json") for m in msgs],
            "info": {"termination_reason": reason,
                     "trial": getattr(sim, "trial", None),
                     "seed": getattr(sim, "seed", None),
                     "user_llm": llm_user},
        })
    return out
