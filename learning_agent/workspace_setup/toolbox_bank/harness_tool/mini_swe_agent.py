#!/usr/bin/env python3
"""mini_swe_agent.py — a one-file terminal agent for Harbor-style tasks (tb-lite).

The minimal POLICY for agentic terminal tasks: given a natural-language
instruction and a shell, loop {model -> ONE bash command -> observation} until
the model declares DONE or the turn budget runs out. This is the agentic
counterpart of harness_toolbox/react_loop.py (which searches a corpus to ANSWER;
this one acts on an environment to CHANGE it — scored by the task's verifier
re-running afterward, not by a judge reading prose).

Everything here is the agent's to modify — better prompts, richer stop
conditions, multi-command turns, self-verification before DONE. The contract
that must survive: `run_agent(instruction, client, execute)` returns the result
dict below, and `execute` is the ONLY way commands reach the environment (the
operator's post-eval injects an executor wired to the task container; swapping
in your own executor locally never touches the scored environment).

Protocol per turn — the model replies with its reasoning plus EXACTLY ONE of:
    ```bash
    <one shell command>
    ```
    DONE: <one-line summary of what was accomplished>

Result dict:
    {"done": bool, "turns": int, "summary": str,
     "actions": [{"cmd": str, "exit": int, "output_tail": str}, ...]}

Offline smoke (no model, no network — scripted client):
    python toolbox/harness_tool/mini_swe_agent.py --self-test
Live, against any OpenAI-compatible endpoint (e.g. submission/serve.py):
    python toolbox/harness_tool/mini_swe_agent.py --base-url http://127.0.0.1:8000/v1 \
        --model <name> --instruction "create hello.txt containing hi" --cwd /tmp/sandbox
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_TOOLBOX_ROOT = str(Path(__file__).resolve().parents[1])
if _TOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, _TOOLBOX_ROOT)
from api_clients.oai_client import OAIClient  # noqa: E402

SYSTEM_TEMPLATE = """You are an autonomous engineer working in a Linux shell to complete a task.

TASK:
{instruction}

Rules:
- Each turn, think briefly, then emit EXACTLY ONE fenced bash block with ONE command:
```bash
<command>
```
- You see the command's output (tail) and exit code next turn. No interactivity:
  never run editors/pagers/watch; write files with heredocs or python -c.
- Verify your work with a command before finishing.
- When the task is genuinely complete, reply with a single line instead of a command:
DONE: <one-line summary>
- You have {max_turns} turns total. Prefer small, checkable steps."""

_FENCE_RES: dict[str, re.Pattern] = {
    # default: bash/sh/shell (or untagged) fences — the terminal-task grammar
    "bash": re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL),
}
_DONE_RE = re.compile(r"^\s*DONE\s*:\s*(.*)$", re.MULTILINE)


def _fence_re(fence: str) -> re.Pattern:
    """Fenced-block regex for a given tag (cached). A non-bash fence (e.g.
    ```action) turns this loop into a generic single-action driver for text
    envs whose actions aren't shell commands."""
    if fence not in _FENCE_RES:
        _FENCE_RES[fence] = re.compile(
            rf"```(?:{re.escape(fence)})?\s*\n(.*?)```", re.DOTALL)
    return _FENCE_RES[fence]

# Observation cap per turn: enough to see errors and file listings, small enough
# that a noisy build log cannot blow the task model's context window.
OUTPUT_TAIL_CHARS = 4000


def parse_turn(reply: str, fence: str = "bash") -> tuple[str | None, str | None]:
    """-> (command, done_summary): exactly one is non-None; (None, None) = malformed.
    DONE wins only when no command block is present — a model that emits both is
    told to finish properly rather than silently having its command dropped."""
    m = _fence_re(fence).search(reply or "")
    if m:
        cmd = m.group(1).strip()
        return (cmd or None), None
    d = _DONE_RE.search(reply or "")
    if d:
        return None, d.group(1).strip()
    return None, None


class LocalExecutor:
    """Run commands in a local directory — the DEV-TIME executor (your own
    sandbox). Post-eval replaces this with one wired to the task's container."""

    def __init__(self, cwd: str, timeout: int = 120):
        self.cwd, self.timeout = cwd, timeout

    def __call__(self, cmd: str) -> tuple[str, int]:
        try:
            cp = subprocess.run(["bash", "-c", cmd], cwd=self.cwd, timeout=self.timeout,
                                capture_output=True, text=True)
            return (cp.stdout or "") + (cp.stderr or ""), cp.returncode
        except subprocess.TimeoutExpired:
            return f"[timeout after {self.timeout}s]", 124


def run_agent(instruction: str, client, execute, max_turns: int = 30,
              temperature: float = 0.0, max_tokens: int = 2048,
              fence: str = "bash", system_template: str = SYSTEM_TEMPLATE,
              log=lambda s: None) -> dict:
    """The loop. `client` needs .chat(messages, temperature=, max_tokens=) -> str;
    `execute` is (cmd) -> (output, exit_code). `fence`/`system_template`
    generalize the grammar beyond bash (pass a template that documents the
    matching fence tag)."""
    messages = [{"role": "system",
                 "content": system_template.format(instruction=instruction.strip(),
                                                   max_turns=max_turns)},
                {"role": "user", "content": "Begin. Emit your first command."}]
    actions: list[dict] = []
    for turn in range(1, max_turns + 1):
        reply = client.chat(messages, temperature=temperature, max_tokens=max_tokens) or ""
        messages.append({"role": "assistant", "content": reply})
        cmd, done = parse_turn(reply, fence=fence)
        if done is not None:
            log(f"[mini-swe] DONE after {turn - 1} command(s): {done}")
            return {"done": True, "turns": turn, "summary": done, "actions": actions}
        if cmd is None:
            messages.append({"role": "user", "content":
                             f"Malformed turn: emit ONE ```{fence}``` block with one "
                             "command, or a `DONE: <summary>` line."})
            continue
        out, rc = execute(cmd)
        tail = out[-OUTPUT_TAIL_CHARS:]
        actions.append({"cmd": cmd, "exit": rc, "output_tail": tail})
        log(f"[mini-swe] turn {turn}: $ {cmd}  (exit {rc})")
        messages.append({"role": "user", "content":
                         f"exit code: {rc}\noutput (tail):\n{tail}\n\n"
                         f"Turns left: {max_turns - turn}. Next command, or DONE:."})
    return {"done": False, "turns": max_turns,
            "summary": "turn budget exhausted before DONE", "actions": actions}


# ----------------------------- self-test / CLI ----------------------------- #

class _ScriptedClient:
    """Deterministic offline stand-in: replays canned turns (contract smoke)."""

    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, messages, **_):
        return self._replies.pop(0) if self._replies else "DONE: out of script"


def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        client = _ScriptedClient([
            "Creating the file.\n```bash\necho hi > hello.txt\n```",
            "Verifying.\n```bash\ncat hello.txt\n```",
            "All good.\nDONE: hello.txt created and verified",
        ])
        res = run_agent("create hello.txt containing hi", client,
                        LocalExecutor(td), max_turns=5, log=print)
        ok = (res["done"] and len(res["actions"]) == 2
              and res["actions"][1]["output_tail"].strip() == "hi"
              and (Path(td) / "hello.txt").read_text().strip() == "hi")
        print(f"[self-test] {'PASS' if ok else 'FAIL'}: {res['summary']}")
        return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="One-file terminal agent (Harbor-style tasks).")
    ap.add_argument("--instruction", help="the task instruction")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", default="", help="model name at the endpoint")
    ap.add_argument("--cwd", default=".", help="working dir for LocalExecutor")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=120, help="per-command seconds")
    ap.add_argument("--self-test", action="store_true",
                    help="offline scripted-client smoke (no model, no network)")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(_self_test())
    if not args.instruction or not args.base_url:
        raise SystemExit("need --instruction and --base-url (or --self-test)")
    client = OAIClient(backend="openai", model=args.model or "task model",
                       base_url=args.base_url)
    res = run_agent(args.instruction, client,
                    LocalExecutor(args.cwd, timeout=args.timeout),
                    max_turns=args.max_turns, log=print)
    print(f"[mini-swe] done={res['done']} turns={res['turns']}: {res['summary']}")


if __name__ == "__main__":
    main()
