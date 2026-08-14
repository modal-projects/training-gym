#!/usr/bin/env python3
"""Submission agent — build() the policy that post-eval drives.

This is the SECOND half of the submission contract. serve.py provides the raw
task model endpoint; this file wraps it in the submission's HARNESS — the minimal
agent that turns a query into a structured result. At post-eval time the
operator imports `build()` and calls the SAME functions the agent used
dev-time, so what is scored is exactly what was built:

    from submission.agent import build
    agent = build()                          # serves WEIGHTS if needed
    agent.answer("...")                      # QA    -> {"answer": str}
    agent.act(instruction, execute)          # env/terminal episode (driver below)
    agent.tool_turn(messages, tools, execute_tool)   # tau2 conversation turn

act() drives ONE episode through a pluggable policy driver — the task's
adapter picks it from the resolved config (task.yaml `agent: driver:`):
    tools     NATIVE OpenAI tool calls (ALFWorld); the adapter supplies the
              env's function schemas; execute(name, args) -> (observation, done).
              Needs an endpoint served WITH a tool-call parser.
    react     ReAct ACTION:/FINAL: text loop for interactive envs;
              execute(action) -> (observation, done). Works against any
              endpoint (and any text-only policy, e.g. a CLI reference).
    mini_swe  fenced-```bash``` terminal loop (Harbor-style tasks);
              execute(cmd) -> (output, exit_code)

Internals are THE AGENT'S TO MODIFY — the starting QA harness is already a
ReAct search loop over the task corpus (grep/glob/read_file, the toolbox's
react_loop tools); improve it, tune the mini-swe loop, replace the
tool-calling strategy. The contract
that must survive: `build(**overrides)` returns an object exposing the three
methods above with these signatures and dict/str returns. `execute` /
`execute_tool` are injected by the caller and are the ONLY bridge into the
scored environment.

The per-archetype minimal agents come from the mounted toolbox (a workspace
carries toolbox/eval_tool+toolbox/harness_tool for QA tasks OR agentic_toolbox for
agentic tasks — see workspace_setup/prepare_workspace.sh), so imports are guarded:
calling a method whose toolbox is absent raises a clear error instead of an
ImportError at module load.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOLBOX_ROOT = str(_ROOT / "toolbox")
if _TOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, _TOOLBOX_ROOT)
from api_clients.oai_client import OAIClient  # noqa: E402

# ---- agent-owned configuration ------------------------------------------- #
SYSTEM_PROMPT = "Answer the question directly and completely."
MAX_TURNS = 30          # default episode step budget (adapters pass the pinned one)
ANSWER_MAX_TOKENS = 2048
SEARCH_BUDGET = 15      # ReAct tool turns per question in the starting QA harness
DEFAULT_DRIVER = "react"  # see module docstring; task.yaml `agent: driver:` selects


def _find_corpus() -> "Path | None":
    """The workspace carries its one task at task/; the corpus (when the
    track ships one) is the starting harness's search space."""
    corpus = _ROOT / "task" / "corpus"
    return corpus if corpus.is_dir() else None


def _task_sys_prompt() -> str:
    sys_txt = _ROOT / "task" / "sys.txt"
    return sys_txt.read_text().strip() if sys_txt.is_file() else SYSTEM_PROMPT


class Agent:
    """The submission policy: one served task model, three ways to drive it."""

    def __init__(self, client: OAIClient, base_url: str = "", model: str = ""):
        self.client = client
        self.base_url = base_url        # needed for raw tool-calling (tau2)
        self.model = model

    # -- QA (openclaw / fav2 / maud) ----------------------------------------- #
    def answer(self, question: str) -> dict:
        """One question -> {"answer": str}.

        Starting harness: a ReAct search loop (grep / glob / read_file over the
        task corpus, SEARCH_BUDGET tool turns) when the workspace has a corpus;
        plain closed-book otherwise. This is a floor, not a ceiling — replace
        it with a better harness, and remember the weights are what carry."""
        corpus = _find_corpus()
        if corpus is not None:
            try:
                return self._react_answer(question, str(corpus))
            except Exception as e:  # noqa: BLE001 — degrade, never zero a question
                return {"answer": self._closed_book(question), "harness_error": str(e)}
        return {"answer": self._closed_book(question)}

    def _closed_book(self, question: str) -> str:
        reply = self.client.chat(
            [{"role": "system", "content": _task_sys_prompt()},
             {"role": "user", "content": question}],
            temperature=0.0, max_tokens=ANSWER_MAX_TOKENS)
        return (reply or "").strip()

    def _react_answer(self, question: str, corpus_root: str) -> dict:
        """ACTION/OBSERVATION/FINAL loop using the toolbox's corpus tools
        (toolbox/harness_tool/react_loop.py — same grammar as the reference
        harness)."""
        from harness_tool.react_loop import (build_react_sys, extract_json,
                                             run_tool, strip_think)
        msgs = [{"role": "system",
                 "content": build_react_sys(_task_sys_prompt())},
                {"role": "user", "content": question}]
        for _ in range(SEARCH_BUDGET):
            reply = strip_think(self.client.chat(
                msgs, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS) or "")
            if "FINAL:" in reply:
                return {"answer": reply.split("FINAL:", 1)[1].strip()}
            if "ACTION:" in reply:
                obs = run_tool(corpus_root,
                               extract_json(reply.split("ACTION:", 1)[1]))
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"OBSERVATION:\n{obs}"}]
                continue
            # Neither marker: treat the whole reply as the answer.
            return {"answer": reply.strip()}
        # Budget exhausted: force a closed-book style final from the transcript.
        msgs.append({"role": "user",
                     "content": "Search budget exhausted. Give your FINAL answer now."})
        reply = strip_think(self.client.chat(
            msgs, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS) or "")
        return {"answer": reply.split("FINAL:", 1)[-1].strip()}

    def answer_batch(self, questions: list[dict], max_workers: int = 4) -> dict[str, str]:
        """[{id, question}] -> {id: answer} (the submission/eval.py contract).

        Delegates to answer() so rewriting answer() is automatically what gets
        scored — the footgun this prevents: eval.py calls answer_batch, so a
        batch path that bypassed answer() would silently score the baseline
        harness instead of yours. Replace with true batching only if you keep
        the two in lockstep."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            replies = list(ex.map(lambda q: self.answer(q["question"]), questions))
        return {str(q["id"]): (r.get("answer") or "").strip()
                for q, r in zip(questions, replies)}

    # -- environment episodes (alfworld / webshop / terminal tasks) --------- #
    def act(self, instruction: str, execute, driver: str = DEFAULT_DRIVER,
            max_turns: int = MAX_TURNS, tools: list | None = None,
            temperature: float = 0.0, max_tokens: int = ANSWER_MAX_TOKENS,
            extra_body: dict | None = None, log=lambda s: None) -> dict:
        """Drive ONE episode with the selected policy driver (module docstring).

        tools:    execute(name, args) -> (observation, done); `tools` = the env's
                  OpenAI function schemas (supplied by the task's adapter)
        react:    execute(action) -> (observation, done)
        mini_swe: execute(cmd) -> (output, exit_code)
        All return {"done", "steps"|"turns", "summary", "actions": [...]}.
        `done` is the ENVIRONMENT's terminal signal, never the model's opinion;
        reward stays with the task's verifier/adapter."""
        try:
            if driver == "tools":
                from harness_tool.react_tool_agent import run_tools
                if not self.base_url:
                    raise RuntimeError(
                        "driver='tools' needs a real endpoint served with a "
                        "tool-call parser (build with weights/base_url, not "
                        "backend='mock'/'cli-claude')")
                return run_tools(instruction, self.base_url, self.model,
                                 tools or [], execute, max_steps=max_turns,
                                 temperature=temperature, max_tokens=max_tokens,
                                 extra_body=extra_body, log=log,
                                 api_key=getattr(self.client, "api_key", "EMPTY"))
            if driver == "react":
                from harness_tool.react_env_agent import run_react
                return run_react(instruction, self.client, execute,
                                 max_steps=max_turns, temperature=temperature,
                                 max_tokens=max_tokens, log=log)
            if driver == "mini_swe":
                from agentic_toolbox.mini_swe_agent import run_agent
                return run_agent(instruction, self.client, execute,
                                 max_turns=max_turns, temperature=temperature,
                                 max_tokens=max_tokens, log=log)
        except ImportError as e:
            raise RuntimeError(
                "act() needs toolbox/agentic_toolbox (mounted for agentic tasks "
                "only — this workspace looks QA-shaped)") from e
        raise ValueError(f"unknown act() driver {driver!r} "
                         "(tools | react | mini_swe)")

    # -- conversational tool use (tau2_*) ------------------------------------ #
    def tool_turn(self, messages: list[dict], tools: list[dict], execute_tool) -> tuple[str, list[dict]]:
        """One assistant turn in a τ²-style conversation; tool calls execute via
        `execute_tool(name, args) -> str`. -> (assistant_text, updated_messages)."""
        try:
            from agentic_toolbox.react_tool_agent import agent_turn
        except ImportError as e:
            raise RuntimeError(
                "tool_turn() needs toolbox/agentic_toolbox (mounted for agentic "
                "tasks only — this workspace looks QA-shaped)") from e
        if not self.base_url:
            raise RuntimeError("tool_turn() needs a real endpoint (build with "
                               "weights/base_url, not backend='mock')")
        return agent_turn(self.base_url, self.model, messages, tools, execute_tool)


def build(weights: str = "", base_url: str = "", model: str = "",
          backend: str = "", port: int = 8000) -> Agent:
    """Construct the submission agent. Post-eval calls this with no arguments
    (WEIGHTS in serve.py / an operator-passed endpoint); dev-time you can point
    it anywhere. backend='mock' gives the deterministic offline stub used by
    contract tests; backend='cli-claude' drives the logged-in claude CLI
    (operator reference baselines — never a scoreable task model)."""
    if backend == "mock":
        return Agent(OAIClient(backend="mock", model="mock"))
    if backend == "cli-claude":
        name = model or "opus"
        return Agent(OAIClient(backend="cli-claude", model=name), model=name)
    from serve import ensure_endpoint  # sibling module, path set below
    base, name = ensure_endpoint(weights=weights, base_url=base_url, port=port)
    name = model or name
    return Agent(OAIClient(backend="openai", model=name, base_url=base),
                 base_url=base, model=name)


# `from serve import ...` inside build(): make the sibling importable whether the
# caller did `import submission.agent` or ran a script from the repo root.
_SUB_DIR = str(Path(__file__).resolve().parent)
if _SUB_DIR not in sys.path:
    sys.path.insert(0, _SUB_DIR)
