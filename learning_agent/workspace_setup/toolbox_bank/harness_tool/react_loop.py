#!/usr/bin/env python3
"""react_loop — configurable ReAct rollout sandbox (NOT the scored instrument).

An EDITABLE mirror of the frozen measurement harness: serves/queries
a model, exposes `grep`/`glob`/`read_file` over a `--corpus` (path-sandboxed), runs the
ACTION/OBSERVATION/FINAL protocol under a `--budget`, and returns the answer + full tool
trace. Experiment freely with tools, budget, prompt, backend — tune here.

NOT THE SCORED INSTRUMENT. The official number comes only from the operator's frozen harness,
hash-pinned in `bench/pins.json`. Edits here change your private experiment and have ZERO
effect on the score; do not copy tweaks back into `harness/` and do not point a scoring run
at this file. To make an idea count it must survive in the task model's weights (axes 1-3) or be
opened as a transfer meta-eval (see transfer_eval.py).

Grammar (verbatim from the frozen harness) — each turn the model emits EXACTLY ONE of:
    ACTION: {"tool": "grep",      "args": {"pattern": "<regex>", "glob": "<glob>"}}
    ACTION: {"tool": "glob",      "args": {"pattern": "<glob>"}}
    ACTION: {"tool": "read_file", "args": {"path": "<path>", "start_line": 1, "end_line": 120}}
    FINAL: <answer>

Backends (heavy imports guarded, so --help always works): vllm, sglang (same engine as the
frozen harness), openai (any OpenAI-compatible endpoint), cli-claude (debug the LOOP on a
keyless host), mock (dependency-free scripted glob->grep->FINAL smoke test).

    python3 toolbox/harness_tool/react_loop.py --backend mock --corpus <corpus> --glob '**/*.txt' --question "..."

`build_engine`, `react_rollout`, and the corpus tools are importable so transfer_eval.py can
drive a rollout with a self-modified config.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ReAct grammar (kept verbatim from the frozen measurement harness so experiments transfer)
# ---------------------------------------------------------------------------

REACT_TEMPLATE = """

You may SEARCH the source + docs (rooted at the corpus) before answering. Each turn \
output EXACTLY ONE of:

  ACTION: {{"tool": "grep",      "args": {{"pattern": "<regex>", "glob": "{glob_ext}"}}}}
  ACTION: {{"tool": "glob",      "args": {{"pattern": "<glob pattern>"}}}}
  ACTION: {{"tool": "read_file", "args": {{"path": "<path under corpus root>", "start_line": 1, "end_line": 120}}}}
  FINAL: {final_hint}

Use searches to ground your answer in the real API and avoid hallucinating. When ready, \
emit FINAL. Be efficient — search only what you need."""

DEFAULT_FINAL_HINT = "<your full answer>"


def build_react_sys(sys_direct: str, glob_ext: str = "**/*",
                    final_hint: str = DEFAULT_FINAL_HINT) -> str:
    """Augment a task's direct system prompt with the ReAct tool grammar."""
    return sys_direct + REACT_TEMPLATE.format(glob_ext=glob_ext, final_hint=final_hint)


# ---------------------------------------------------------------------------
# Corpus tools — path-sandboxed to the corpus root (grep / glob / read_file).
# The sandbox here is intentionally STRICTER than the frozen harness (realpath +
# commonpath, so `..` and symlink escapes are rejected): this is your sandbox, be safe.
# ---------------------------------------------------------------------------


def safe_path(corpus_root: str, path: str) -> str | None:
    """Resolve `path` under `corpus_root`, rejecting traversal/symlink escapes.

    Returns an absolute path inside the corpus, or None if it escapes / is missing.
    """
    root = os.path.realpath(corpus_root)
    candidate = os.path.realpath(os.path.join(root, str(path).lstrip("/")))
    try:
        inside = os.path.commonpath([root, candidate]) == root
    except ValueError:  # different drives on Windows, etc.
        return None
    if not inside or not os.path.exists(candidate):
        return None
    return candidate


def grep_corpus(corpus_root: str, pattern: str, glob: str = "**/*",
                max_results: int = 40) -> str:
    """Regex-search files under the corpus matching `glob`; return `path:line: text`."""
    import glob as G
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"[grep error: bad regex: {e}]"
    root = os.path.realpath(corpus_root)
    hits: list[str] = []
    for fp in G.glob(os.path.join(root, glob), recursive=True):
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        hits.append(f"{os.path.relpath(fp, root)}:{i}: {line.rstrip()[:200]}")
                        if len(hits) >= max_results:
                            return "\n".join(hits) + f"\n[truncated at {max_results}]"
        except OSError:
            continue
    return "\n".join(hits) if hits else "[no matches]"


def glob_corpus(corpus_root: str, pattern: str, max_results: int = 80) -> str:
    """List files under the corpus matching a glob pattern (sorted, truncated)."""
    import glob as G
    root = os.path.realpath(corpus_root)
    fs = [os.path.relpath(p, root)
          for p in G.glob(os.path.join(root, pattern), recursive=True) if os.path.isfile(p)]
    return "\n".join(sorted(fs)[:max_results]) if fs else "[no files]"


def read_file_corpus(corpus_root: str, path: str, start_line: int = 1, end_line: int = 0,
                     max_lines: int = 200) -> str:
    """Read a line range of a file under the corpus, with 1-based line-number gutters."""
    p = safe_path(corpus_root, path)
    if not p or not os.path.isfile(p):
        return f"[not found: {path}]"
    lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
    s = max(1, int(start_line))
    e = int(end_line) if end_line else s + max_lines - 1
    e = min(e, len(lines), s + max_lines - 1)
    body = "\n".join(f"{i:>5}: {lines[i - 1]}" for i in range(s, e + 1))
    return body + (f"\n[file has {len(lines)} lines]" if e < len(lines) else "")


def extract_json(s: str) -> dict | None:
    """Find and decode the first balanced JSON object in `s` (tolerant of prose)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def run_tool(corpus_root: str, obj: dict | None, default_glob: str = "**/*",
             max_grep_results: int = 40, max_read_lines: int = 200) -> str:
    """Dispatch one parsed ACTION object to the corpus tools; never raises."""
    tool = (obj or {}).get("tool")
    args = (obj or {}).get("args", {}) or {}
    try:
        if tool == "grep":
            return grep_corpus(corpus_root, args.get("pattern", ""),
                               args.get("glob", default_glob), max_grep_results)
        if tool == "glob":
            return glob_corpus(corpus_root, args.get("pattern", "**/*"))
        if tool == "read_file":
            return read_file_corpus(corpus_root, args.get("path", ""),
                                    args.get("start_line", 1), args.get("end_line", 0),
                                    max_read_lines)
        return f"[unknown tool: {tool}]"
    except Exception as e:  # noqa: BLE001 — tool errors must not kill the rollout
        return f"[tool error: {e}]"


def strip_think(text: str) -> str:
    """If the model emits <think>...</think> reasoning blocks, drop them so FINAL/ACTION
    parsing and the graded answer see only post-reasoning content (mirrors the frozen
    harness). A no-op for models that don't produce <think> blocks."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


# ---------------------------------------------------------------------------
# Model backends. Each Engine exposes .chat(messages, max_new_tokens, temperature)
# -> raw text. Heavy imports live INSIDE __init__ so --help works without the dep.
# ---------------------------------------------------------------------------


class _HFChat:
    """Shared chat-template rendering for local HF-tokenizer backends."""

    def __init__(self, model: str):
        from transformers import AutoTokenizer  # guarded
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def render(self, messages: list[dict]) -> str:
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # tokenizer without the enable_thinking kwarg
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)


class VLLMEngine(_HFChat):
    """Local vLLM serving of the task model (needs vllm + transformers + GPU)."""

    def __init__(self, model: str, adapter: str = "", tp_size: int = 1,
                 max_len: int = 131072):
        try:
            from vllm import LLM  # guarded
        except ImportError as e:  # pragma: no cover
            raise SystemExit("backend 'vllm' needs vllm installed (pip install vllm)") from e
        super().__init__(model)
        kw = dict(model=model, tensor_parallel_size=tp_size, trust_remote_code=True,
                  max_model_len=max_len)
        if adapter:
            kw.update(enable_lora=True)
            self._lora = adapter
        else:
            self._lora = None
        self.llm = LLM(**kw)

    def chat(self, messages, max_new_tokens=4096, temperature=0.0) -> str:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        kw = {}
        if self._lora:
            from vllm.lora.request import LoRARequest
            kw["lora_request"] = LoRARequest("adapter", 1, self._lora)
        out = self.llm.generate([self.render(messages)], sp, **kw)
        return out[0].outputs[0].text


class SGLangEngine(_HFChat):
    """Local SGLang serving — the same engine the frozen harness uses."""

    def __init__(self, model: str, adapter: str = "", tp_size: int = 1,
                 max_len: int = 131072):
        try:
            import sglang as sgl  # guarded
        except ImportError as e:  # pragma: no cover
            raise SystemExit("backend 'sglang' needs sglang installed") from e
        super().__init__(model)
        kw = dict(model_path=model, tp_size=tp_size, mem_fraction_static=0.85,
                  context_length=max_len, trust_remote_code=True)
        if adapter:
            kw["lora_paths"] = [adapter]
        self.engine = sgl.Engine(**kw)

    def chat(self, messages, max_new_tokens=4096, temperature=0.0) -> str:
        out = self.engine.generate(
            self.render(messages),
            {"temperature": temperature, "max_new_tokens": max_new_tokens})
        return out["text"]


class OpenAIEngine:
    """Any OpenAI-compatible /chat/completions endpoint (stdlib urllib, no SDK).

    Point --base-url at a vLLM/SGLang server you launched (e.g. on Modal, tunneled).
    """

    def __init__(self, model: str, base_url: str, api_key: str = "",
                 disable_thinking: bool = True, timeout: int = 300):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or "EMPTY"
        self.disable_thinking = disable_thinking
        self.timeout = timeout

    def chat(self, messages, max_new_tokens=4096, temperature=0.0) -> str:
        import urllib.request
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_new_tokens}
        if self.disable_thinking:
            # vLLM/SGLang honor this to turn off a model's thinking mode; strict servers ignore it.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read().decode())
        return payload["choices"][0]["message"]["content"]


class CLIClaudeEngine:
    """`claude --print --output-format json --model ...` — for debugging the LOOP itself
    on a keyless host (this is NOT the task model; use it to sanity-check the protocol)."""

    def __init__(self, model: str = "claude-opus-4", timeout: int = 300):
        from shutil import which
        if which("claude") is None:
            raise SystemExit("backend 'cli-claude' needs the `claude` CLI on PATH")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, max_new_tokens=4096, temperature=0.0) -> str:
        parts = []
        for m in messages:
            parts.append(f"[{m['role'].upper()}]\n{m['content']}")
        prompt = "\n\n".join(parts)
        proc = subprocess.run(
            ["claude", "--print", "--output-format", "json", "--model", self.model, prompt],
            capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:400]}")
        try:
            env = json.loads(proc.stdout)
            return env.get("result", proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout


class MockEngine:
    """Dependency-free scripted engine: glob -> grep -> FINAL. Smoke-tests the sandbox
    and the ACTION/OBSERVATION/FINAL parser over a real corpus with no model at all."""

    def __init__(self, glob_ext: str = "**/*"):
        self.glob_ext = glob_ext
        self._turn = 0

    def chat(self, messages, max_new_tokens=4096, temperature=0.0) -> str:
        self._turn += 1
        if self._turn == 1:
            return 'ACTION: {"tool": "glob", "args": {"pattern": "**/*"}}'
        if self._turn == 2:
            return ('ACTION: {"tool": "grep", "args": {"pattern": "the", "glob": "'
                    + self.glob_ext + '"}}')
        return ("FINAL: [mock] loop + sandbox exercised over the corpus; "
                "last observation reviewed. Replace with a real backend to get real answers.")


def build_engine(args) -> object:
    """Construct the requested backend from parsed args (or an args-like namespace)."""
    backend = args.backend
    # No model is baked in. vllm/sglang/openai serve the task model, so they need one.
    if backend in ("vllm", "sglang", "openai") and not getattr(args, "model", None):
        raise SystemExit(
            "no model specified: pass --model <task-model> or set $TASK_MODEL "
            "(no model is baked in; the task model is pinned per task by the operator)")
    if backend == "vllm":
        return VLLMEngine(args.model, adapter=args.adapter, tp_size=args.tp,
                          max_len=args.max_len)
    if backend == "sglang":
        return SGLangEngine(args.model, adapter=args.adapter, tp_size=args.tp,
                            max_len=args.max_len)
    if backend == "openai":
        if not args.base_url:
            raise SystemExit("backend 'openai' needs --base-url (e.g. http://host:8000/v1)")
        return OpenAIEngine(args.model, args.base_url, api_key=args.api_key,
                            disable_thinking=args.disable_thinking)
    if backend == "cli-claude":
        # cli-claude debugs the LOOP, not the task model — default to a claude-* model.
        return CLIClaudeEngine(args.model or "claude-opus-4")
    if backend == "mock":
        return MockEngine(glob_ext=args.glob)
    raise SystemExit(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# The rollout — structurally mirrors harness/eval.py `_react_rollout`, but every
# knob is exposed so you can experiment. Returns answer + full tool-call trace.
# ---------------------------------------------------------------------------


def react_rollout(engine, corpus_root: str, question: str, budget: int,
                  sys_direct: str, sys_react: str, default_glob: str = "**/*",
                  note: str = "", temperature: float = 0.0, max_new_tokens: int = 4096,
                  max_grep_results: int = 40, max_read_lines: int = 200) -> dict:
    """Run one ReAct rollout. Returns a dict with:
        answer          final answer string
        trace           [{iter, action, observation}] tool-call log
        n_tool_calls    number of executed tool calls
        iterations      loop iterations consumed
        forced          True if the budget cap forced a FINAL
    `note` (if set) is an open-book cheatsheet prepended to the system prompt.
    """
    pre = f"Reference notes on the current API (study these):\n{note}\n\n" if note else ""

    if budget == 0:  # closed-book, no tools
        msgs = [{"role": "system", "content": pre + sys_direct},
                {"role": "user", "content": question}]
        text = strip_think(engine.chat(msgs, max_new_tokens, temperature))
        return {"answer": text, "trace": [], "n_tool_calls": 0,
                "iterations": 0, "forced": False}

    msgs = [{"role": "system", "content": pre + sys_react},
            {"role": "user", "content": question}]
    trace: list[dict] = []
    for it in range(budget + 1):
        force = it == budget
        if force:
            msgs.append({"role": "user",
                         "content": "Search budget exhausted. Output FINAL with your answer now."})
        text = strip_think(engine.chat(msgs, max_new_tokens, temperature))
        fm = re.search(r"FINAL\s*:?", text)
        am = re.search(r"ACTION\s*:?", text)
        if force or (fm and (not am or fm.start() < am.start())):
            answer = text[fm.end():].strip() if fm else text.strip()
            return {"answer": answer, "trace": trace, "n_tool_calls": len(trace),
                    "iterations": it, "forced": bool(force and not fm)}
        if am:
            action = extract_json(text[am.end():])
            obs = run_tool(corpus_root, action, default_glob,
                           max_grep_results, max_read_lines)
            trace.append({"iter": it, "action": action, "observation": obs})
            msgs.append({"role": "assistant", "content": text.strip()})
            msgs.append({"role": "user", "content": f"OBSERVATION:\n{obs}"})
            continue
        # no parseable directive -> treat the whole text as the answer
        return {"answer": text.strip(), "trace": trace, "n_tool_calls": len(trace),
                "iterations": it, "forced": False}
    return {"answer": "", "trace": trace, "n_tool_calls": len(trace),
            "iterations": budget, "forced": True}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_questions(args) -> list[dict]:
    """Return [{id, question}] from --question or --questions-file (JSON list or JSONL)."""
    if args.question:
        return [{"id": "q0", "question": args.question}]
    if not args.questions_file:
        raise SystemExit("provide --question or --questions-file")
    p = Path(args.questions_file)
    raw = p.read_text()
    rows: list[dict]
    if p.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    else:
        rows = json.loads(raw)
    out = []
    for i, r in enumerate(rows):
        out.append({"id": r.get("id", f"q{i}"), "question": r["question"]})
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Configurable ReAct rollout sandbox (NOT the scored instrument — "
                    "that is the operator's hash-pinned harness).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus root the tools search over")
    ap.add_argument("--backend", default="mock",
                    choices=["vllm", "sglang", "openai", "cli-claude", "mock"])
    ap.add_argument("--model", default=os.environ.get("TASK_MODEL"),
                    help="task model id / path (defaults to $TASK_MODEL; "
                         "use a claude-* id for cli-claude; ignored by mock)")
    ap.add_argument("--adapter", default="", help="LoRA adapter path (vllm/sglang)")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint (openai backend)")
    ap.add_argument("--api-key", default="", help="bearer token for the openai backend")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (vllm/sglang)")
    ap.add_argument("--max-len", type=int, default=131072,
                    help="model context length; 131072 = the task model's full 128K window")
    ap.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="ask an openai server to disable the model's thinking mode "
                         "(--no-disable-thinking to leave it on)")
    ap.add_argument("--budget", type=int, default=20,
                    help="max tool iterations before FINAL is forced (0 = closed-book); 20 "
                         "matches the harness's largest search budget (harness/eval.py 0,5,20)")
    ap.add_argument("--glob", default="**/*",
                    help="default grep/glob extension advertised in the prompt "
                         "(fav2=**/*.txt, dspy=**/*.py, openclaw=**/*.ts)")
    ap.add_argument("--final-hint", default=DEFAULT_FINAL_HINT,
                    help="FINAL-line hint injected into the ReAct system prompt")
    ap.add_argument("--sys", default="", help="inline system prompt (overrides --sys-file)")
    ap.add_argument("--sys-file", default="", help="path to a task sys.txt (verbatim)")
    ap.add_argument("--note-file", default="", help="open-book cheatsheet prepended to system")
    ap.add_argument("--question", default="", help="single question to answer")
    ap.add_argument("--questions-file", default="",
                    help="JSON list or .jsonl of {id, question}")
    # temperature=0.0 -> greedy decoding so a re-run reproduces the answer; max-new-tokens=4096
    # caps one turn. Tool-output caps (grep 40 hits / read 200 lines; glob 80 elsewhere) bound
    # each OBSERVATION so a long file cannot blow the context window.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-grep-results", type=int, default=40)
    ap.add_argument("--max-read-lines", type=int, default=200)
    ap.add_argument("--out", default="", help="write full results JSON here")
    ap.add_argument("--quiet", action="store_true", help="suppress per-question stderr log")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.isdir(args.corpus):
        raise SystemExit(f"--corpus is not a directory: {args.corpus}")

    if args.sys:
        sys_direct = args.sys
    elif args.sys_file:
        sys_direct = Path(args.sys_file).read_text()
    else:
        sys_direct = ("You are an expert assistant. Answer the question grounded in the "
                      "corpus you can search.")
    note = Path(args.note_file).read_text() if args.note_file else ""
    sys_react = build_react_sys(sys_direct, args.glob, args.final_hint)

    questions = _load_questions(args)
    engine = build_engine(args)

    answers, traces, meta = {}, {}, {}
    for q in questions:
        # fresh scripted state per question for the mock engine
        if isinstance(engine, MockEngine):
            engine._turn = 0
        res = react_rollout(
            engine, args.corpus, q["question"], args.budget, sys_direct, sys_react,
            default_glob=args.glob, note=note, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, max_grep_results=args.max_grep_results,
            max_read_lines=args.max_read_lines)
        answers[q["id"]] = res["answer"]
        traces[q["id"]] = res["trace"]
        meta[q["id"]] = {"tool_calls": res["n_tool_calls"], "iterations": res["iterations"],
                         "forced": res["forced"]}
        if not args.quiet:
            print(f"[{q['id']}] tool_calls={res['n_tool_calls']} "
                  f"iters={res['iterations']} forced={res['forced']}", file=sys.stderr)

    result = {"backend": args.backend, "model": args.model, "corpus": args.corpus,
              "budget": args.budget, "glob": args.glob,
              "answers": answers, "traces": traces, "meta": meta,
              "note": "SANDBOX RESULT — not an official score (see module docstring)."}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        if not args.quiet:
            print(f"wrote {args.out}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
