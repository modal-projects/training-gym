"""Thin OpenAI-compatible chat client used by the data-toolbox generators.

`OAIClient` talks to one of three backends (stdlib only, no `openai` package):
  * "openai"     -> OpenAI-compatible HTTP endpoint (vLLM/SGLang); POSTs to
                    {base_url}/chat/completions.
  * "cli-claude" -> Claude CLI fallback for hosts with no ANTHROPIC_API_KEY
                    (`claude --print --output-format json`).
  * "mock"       -> DETERMINISTIC offline stub: no model, no network, no deps. Returns
                    hash-seeded, clearly-labelled "[MOCK ...]" text (or a valid JSON object
                    when the prompt asks for strict JSON) so every generator + rubric_eval
                    can run end-to-end in tests / reproducible smoke. NOT real output.
  * "auto"       -> "openai" when --base-url is set, else "cli-claude".

Model-agnostic: no model id is baked in; resolve_model() fills --model from $TASK_MODEL
and errors if neither is set (see ../README.md).

Helpers for the generators: add_client_args()/client_from_args()/resolve_model() wire up
consistent flags; chat()/batch_chat() do single and concurrent completions
(optional top_p/top_k forwarded to vLLM/SGLang); parse_json_object()/extract_first_json()
pull JSON out of a fenced/prose-wrapped reply.

  python toolbox/data_toolbox/serving/oai_client.py --base-url http://localhost:8000/v1 --prompt 'reply: ok'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

Message = dict[str, str]
Messages = list[Message]

# the operator defaults ---------------------------------------------------------------
# No model is baked in — it comes from $TASK_MODEL (the operator pins the base model per
# task in bench/config.yaml). May be None if unset; OAIClient then raises a clear
# error unless an explicit model is passed, and resolve_model() surfaces it as an
# argparse error at the CLI boundary.
TASK_MODEL = os.environ.get("TASK_MODEL")  # served/generator model; None if unset


# --------------------------------------------------------------------------- #
# JSON helpers (shared by every generator + the judge)
# --------------------------------------------------------------------------- #
def extract_first_json(text: str) -> Optional[Any]:
    """Return the first balanced JSON value (object or array) found in `text`,
    tolerating code fences and surrounding prose. None if nothing parses."""
    if not text:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def parse_json_object(text: str) -> Optional[dict]:
    """Like extract_first_json but insists the result is a dict."""
    obj = extract_first_json(text)
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
class OAIClient:
    """Backend-agnostic chat helper.

    Args:
      base_url:    OpenAI-compatible root, e.g. "http://localhost:8000/v1".
                   Trailing "/chat/completions" is appended automatically.
      model:       model id / served name (or the CLI model for cli-claude). Falls
                   back to $TASK_MODEL; a clear error is raised if both are unset
                   (except backend "mock", which defaults the model to "mock").
      backend:     "auto" | "openai" | "cli-claude" | "mock".
      api_key:     bearer token for the HTTP backend (vLLM/SGLang ignore it;
                   default "EMPTY").
      temperature, max_tokens, timeout: request defaults, overridable per call.
        temperature=0.7 is a general-purpose sampling default (generators pass 0.0 for
        deterministic data, higher for exploration); max_tokens=1024 caps a single reply;
        timeout=600s (10 min) tolerates slow first-token latency on a cold/large served model.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        backend: str = "auto",
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 600,
    ) -> None:
        if backend == "auto":
            backend = "openai" if base_url else "cli-claude"
        if backend not in ("openai", "cli-claude", "mock"):
            raise ValueError(f"unknown backend {backend!r} (openai|cli-claude|mock|auto)")
        if backend == "openai" and not base_url:
            raise ValueError("backend 'openai' needs --base-url")
        self.base_url = (base_url or "").rstrip("/")
        self.model = model if model is not None else TASK_MODEL
        if not self.model:
            if backend == "mock":
                self.model = "mock"  # provenance sentinel — no model is contacted
            else:
                raise ValueError(
                    "no model set: pass model=<task-model> or set $TASK_MODEL "
                    "(no model is baked in; the task model is pinned per task by the operator)")
        self.backend = backend
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # -- public API --------------------------------------------------------- #
    def chat(
        self,
        messages: Messages,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> str:
        """Return the assistant text for one prompt (list of role/content msgs).

        `top_p` / `top_k` are forwarded to the vLLM/SGLang request body when set
        (the report samples at T=0.7, top_p=0.8, top_k=20); the CLI backend ignores
        them. Unknown kwargs are swallowed so callers can pass extras uniformly."""
        temp = self.temperature if temperature is None else temperature
        mx = self.max_tokens if max_tokens is None else max_tokens
        if self.backend == "openai":
            return self._chat_http(messages, temp, mx, stop, top_p, top_k)
        if self.backend == "mock":
            return self._chat_mock(messages)
        return self._chat_cli(messages, temp, mx)

    def batch_chat(
        self,
        batch: list[Messages],
        max_workers: int = 8,
        **kw: Any,
    ) -> list[str]:
        """Run many prompts concurrently; results align with `batch` order.
        A failing item yields "" rather than aborting the batch; failures are
        counted and the first few are printed to stderr so a dead endpoint is
        never silent (callers then see 0 usable replies AND the reason)."""
        if not batch:
            return []
        # max_workers=8: a throughput/politeness default that keeps one served endpoint
        # busy without tripping provider rate limits; clamped to the batch size below.
        workers = max(1, min(max_workers, len(batch)))
        errors: list[Exception] = []

        def _one(msgs: Messages) -> str:
            try:
                return self.chat(msgs, **kw)
            except Exception as e:  # noqa: BLE001 - one bad prompt must not kill the batch
                errors.append(e)
                return ""

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_one, batch))
        if errors:
            print(f"[oai_client] batch_chat: {len(errors)}/{len(batch)} requests FAILED; "
                  f"first {min(3, len(errors))} error(s):", file=sys.stderr)
            for e in errors[:3]:
                print(f"  - {type(e).__name__}: {e}", file=sys.stderr)
        return results

    # -- HTTP (vLLM / SGLang, OpenAI-compatible) ---------------------------- #
    def _chat_http(self, messages, temperature, max_tokens, stop,
                   top_p=None, top_k=None) -> str:
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = stop
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k  # vLLM/SGLang extension (ignored by strict OpenAI)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - network dependent
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:  # pragma: no cover - network dependent
            raise RuntimeError(f"cannot reach {url}: {e.reason} "
                               f"(is the server up? see serving/vllm_serve.py)") from e
        return payload["choices"][0]["message"]["content"]

    # -- CLI fallback (no ANTHROPIC_API_KEY on host) ------------------------ #
    def _chat_cli(self, messages, temperature, max_tokens) -> str:
        if shutil.which("claude") is None:
            raise RuntimeError(
                "backend 'cli-claude' needs the `claude` CLI on PATH. Either install "
                "it, or point --base-url at a vLLM/SGLang server (backend 'openai').")
        system, prompt = _flatten_messages(messages)
        cmd = ["claude", "--print", "--output-format", "json", "--model", self.model]
        if system:
            cmd += ["--append-system-prompt", system]
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired as e:  # pragma: no cover
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s") from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:500]}")
        out = proc.stdout.strip()
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            return out  # some CLI builds print bare text even with --output-format json
        # `claude --print --output-format json` -> {"type":"result","result":"..."}
        if isinstance(obj, dict):
            for key in ("result", "text", "content"):
                if isinstance(obj.get(key), str):
                    return obj[key]
        return out

    # -- deterministic offline stub (tests / reproducible smoke; no model/net) -- #
    def _chat_mock(self, messages: Messages) -> str:
        """Return a DETERMINISTIC, clearly-labelled reply with no model/network/deps.

        The text is hash-seeded from the flattened prompt, so it is stable across runs
        but varies by input. When the prompt asks for a strict JSON object (e.g. the
        eval-item authoring prompt), a minimal VALID JSON object is returned instead so
        downstream JSON parsing still succeeds. Every payload is stamped "[MOCK ...]"
        (or has MOCK-labelled fields) so it can never be mistaken for a real generation."""
        system, prompt = _flatten_messages(messages)
        digest = hashlib.sha256(f"{system}\x00{prompt}".encode("utf-8")).hexdigest()
        if _prompt_wants_json_object(prompt):
            return _mock_json_object(digest)
        return (f"[MOCK generation — deterministic offline stub, NOT a real model output] "
                f"backend=mock model={self.model} prompt_sha256={digest[:12]}")


def _flatten_messages(messages: Messages) -> tuple[str, str]:
    """Collapse role/content messages into (system_text, user_prompt) for the CLI.
    System turns are joined into the system slot; user/assistant turns become a
    labelled transcript so multi-turn context survives the single-prompt CLI."""
    sys_parts, turns = [], []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "system":
            sys_parts.append(content)
        elif role == "assistant":
            turns.append(f"Assistant: {content}")
        else:
            turns.append(f"User: {content}")
    prompt = "\n\n".join(turns) if turns else ""
    return "\n\n".join(sys_parts), prompt


def _prompt_wants_json_object(prompt: str) -> bool:
    """Heuristic used by the mock backend: did the caller demand a single strict JSON
    object? True for gen_eval's eval-item authoring prompt (embeds a JSON skeleton with
    "rubric"/"gold_answer"); False for the free-text generators."""
    p = (prompt or "").lower()
    return '"rubric"' in p and '"gold_answer"' in p


def _mock_json_object(digest: str) -> str:
    """A minimal, VALID, clearly-mock JSON object for structured-output prompts (gen_eval
    eval item). Weights already sum to 100; every field is MOCK-labelled so it is never
    mistaken for real generated data."""
    obj = {
        "topic": "mock",
        "question": f"[MOCK] deterministic stub question (sha {digest[:8]}); not a real generation.",
        "gold_answer": f"[MOCK] deterministic stub answer (sha {digest[:8]}); not a real generation.",
        "rubric": [
            {"claim_type": "fact", "weight": 60,
             "statement": "[MOCK] load-bearing stub claim; not a real rubric."},
            {"claim_type": "decoy", "weight": 40,
             "statement": "[MOCK] decoy stub claim a correct answer must avoid."},
        ],
    }
    return json.dumps(obj)


# --------------------------------------------------------------------------- #
# argparse plumbing so every generator exposes consistent flags
# --------------------------------------------------------------------------- #
def add_client_args(
    parser: argparse.ArgumentParser,
    default_model: Optional[str] = TASK_MODEL,
    default_backend: str = "auto",
    default_temperature: float = 0.7,
    default_max_tokens: int = 1024,
    group_title: Optional[str] = None,
) -> None:
    """Register --base-url/--model/--backend/--api-key/--temperature/--max-tokens."""
    g = parser.add_argument_group(group_title or "model")
    g.add_argument("--base-url", dest="base_url", default=None,
                   help="OpenAI-compatible base_url (e.g. http://localhost:8000/v1); "
                        "omit to use the claude CLI fallback")
    g.add_argument("--model", dest="model", default=default_model,
                   help=f"model id / served name (default {default_model or '$TASK_MODEL'})")
    g.add_argument("--backend", dest="backend", default=default_backend,
                   choices=["auto", "openai", "cli-claude", "mock"],
                   help="auto = openai if base-url set, else cli-claude; "
                        "mock = deterministic offline stub (no model/network, for tests)")
    g.add_argument("--api-key", dest="api_key", default="EMPTY",
                   help="bearer token for the HTTP backend (vLLM/SGLang ignore it)")
    g.add_argument("--temperature", dest="temperature", type=float,
                   default=default_temperature)
    g.add_argument("--max-tokens", dest="max_tokens", type=int,
                   default=default_max_tokens)


def client_from_args(args: argparse.Namespace) -> OAIClient:
    """Build an OAIClient from the flags added by add_client_args."""
    return OAIClient(
        base_url=args.base_url,
        model=args.model,
        backend=args.backend,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def resolve_model(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> str:
    """Return the resolved model id, or exit with a clean argparse error.

    MODEL-AGNOSTIC contract: no model id is baked in. `add_client_args` defaults
    `--model` to `$TASK_MODEL`; if neither the flag nor the env var is set,
    `getattr(...)` is None and we raise `parser.error` (exit code 2) rather than
    letting OAIClient raise a bare ValueError deep in the call stack. Call this once,
    right after `parse_args()`, before building the client (skip it on offline paths
    like --dry-run that never contact a model).

    Exception: backend "mock" needs no real model — it returns the "mock" sentinel for
    provenance so a fully offline run needs neither --model nor $TASK_MODEL.
    """
    model = getattr(args, "model", None)
    if not model:
        if getattr(args, "backend", None) == "mock":
            return "mock"
        parser.error(
            "no model set: pass --model <task-model> or set $TASK_MODEL "
            "(no model is baked in; the task model is pinned per task by the operator)")
    return model


# --------------------------------------------------------------------------- #
# CLI: smoke-test a backend without writing any generator
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Smoke-test an OpenAI-compatible endpoint or the claude CLI fallback.")
    add_client_args(ap)
    ap.add_argument("--prompt", default="Reply with the single word: ok.",
                    help="user prompt to send")
    ap.add_argument("--system", default="", help="optional system prompt")
    args = ap.parse_args()
    client = client_from_args(args)
    msgs: Messages = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    msgs.append({"role": "user", "content": args.prompt})
    print(f"[backend={client.backend} model={client.model} "
          f"base_url={client.base_url or '-'}]")
    print(client.chat(msgs))


if __name__ == "__main__":
    main()
