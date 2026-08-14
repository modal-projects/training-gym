"""LLM-as-judge verifier: score a candidate answer against a weighted-claim rubric.

question + candidate + weighted rubric -> forced STRUCTURED output (one verdict per claim,
score in {0, 0.5, 1}) -> N-vote majority per claim (median on ties) -> weighted sum / total
weight -> claim score in [0, 1]. This is the agent's OWN verifier and can double as an RL
reward. The official pinned judge and hidden test stay frozen in the operator `harness/` — not
vendored here.

Backends, in auto-selection order:
  - `openai` (CANONICAL — the same wire shape as the official harness/judge_cli.py): OpenAI
    Chat Completions with forced structured output. Used when $LEARNING_AGENT_JUDGE_URL is set (the
    operator-run judge service — harness/judge_service.py — which holds the key and PINS the
    judge model to bench/config.yaml judge.model) or when $OPENAI_API_KEY is set (direct).
    This is how every number stays comparable: dev-set evals, intermediate-checkpoint evals
    during training, and official scoring all go through the identical judge.
  - `api` (canonical alternative): Anthropic Messages API when ANTHROPIC_API_KEY is set
    (forced tool use).
  - `cli` (NON-canonical, stamped canonical=false): the `claude --print` CLI fallback.
  - `mock`: a DETERMINISTIC offline stub (no key/network/deps) that emits hash-seeded
    verdicts snapped to {0,0.5,1} for tests + reproducible smoke — NON-CANONICAL, never
    auto-selected; you must ask for it explicitly.
Model resolution: --model, else $JUDGE_MODEL, else the PINNED judge from bench/config.yaml
(judge.model) when that file is readable — so by default you are judged by the same model
that scores your submission. (--backend mock defaults to the "mock" sentinel.) The
`anthropic` SDK is imported lazily — a stdlib urllib path is used when it is absent but a key
is set — so --help works with nothing installed.

Rubric claim shape: {"claim_id","weight","statement","claim_type"?} (weight defaults to 1).

    python toolbox/eval_toolbox/judge_client.py --input row.json --n-votes 3
    # {question, candidate|answer, rubric, gold_answer?}; or --question/--rubric inline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# No judge model is baked in (model-agnostic). It comes from --model or $JUDGE_MODEL, with
# the benchmark's PINNED judge (bench/config.yaml judge.model) as the default of last resort;
# resolve_judge_model() surfaces a clean error only if none of the three exists.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

VALID_SCORES = (0.0, 0.5, 1.0)

_ENV_PLACEHOLDERS = {"", "put-your-anthropic-api-key-here", "sk-...", "changeme", "your-key-here"}

# Only the judge-service pointers are auto-loaded from .env — NEVER provider
# keys (those are the operator's identity; bench.py loads them explicitly via
# harness/envfile for official scoring). Sealed workspaces carry exactly these
# three in .env, so the judge instruments find the canonical judge service
# without the agent having to export anything.
_ENV_AUTOLOAD_KEYS = ("LEARNING_AGENT_JUDGE_URL", "LEARNING_AGENT_JUDGE_TOKEN", "LEARNING_AGENT_SESSION")


def _load_repo_env() -> None:
    """Merge the LEARNING_AGENT_JUDGE_* / LEARNING_AGENT_SESSION lines of the nearest .env (CWD first,
    then the repo root this file lives in) into os.environ, non-overriding —
    same placeholder rules as harness/envfile.py."""
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        path = root / ".env"
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if (key not in _ENV_AUTOLOAD_KEYS or key in os.environ
                    or value.lower() in _ENV_PLACEHOLDERS):
                continue
            os.environ[key] = value
        return  # first .env found wins (CWD = the workspace the tool runs in)


_load_repo_env()


def pinned_judge_model() -> str | None:
    """The judge model pinned in bench/config.yaml (judge: model: <id>), or None when the
    file isn't readable. Line-based on purpose — no yaml dep, and the config is pinned by
    sha256 so its shape doesn't drift under us. This is what makes the agent's own dev-time
    verdicts come from the SAME judge that scores submissions, by default."""
    cfg = Path(__file__).resolve().parents[2] / "bench" / "config.yaml"
    try:
        in_judge = False
        for raw in cfg.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.strip() == "judge:":
                in_judge = True
                continue
            if in_judge:
                if not line.startswith((" ", "\t")) or ":" not in line:
                    in_judge = False
                    continue
                key, _, value = line.strip().partition(":")
                if key.strip() == "model" and value.strip():
                    return value.strip()
    except OSError:
        return None
    return None


def resolve_judge_model(model: str | None = None, backend: str | None = None) -> str:
    """Return the judge model id or exit cleanly. Precedence: explicit --model, then
    $JUDGE_MODEL, then the pinned judge from bench/config.yaml. Backend "mock" needs no
    real model, so it resolves to the "mock" provenance sentinel — a fully offline run
    needs neither --model nor $JUDGE_MODEL."""
    if backend == "mock":
        return model or JUDGE_MODEL or "mock"
    m = model or JUDGE_MODEL or pinned_judge_model()
    if not m:
        raise SystemExit("no judge model: pass --model (or --judge-model), set $JUDGE_MODEL, "
                         "or run from a checkout with bench/config.yaml (the pinned judge)")
    return m


# ---- grader personas (strict, weighted-claim; task selects the persona) ----

JUDGE_SYS_CODE = """You are a strict grader for coding answers. You are given a question, a
reference (gold) answer, the candidate answer to grade, and a weighted rubric of claims.

For EACH claim, decide whether the CANDIDATE answer satisfies it:
  - score 1   : the candidate clearly satisfies the claim
  - score 0   : it does not (including if it falls for a stated "decoy" / "does NOT satisfy" case)
  - score 0.5 : genuinely partial — only when the claim has separable parts and some are met

Judge ONLY what the candidate answer actually does. The gold answer is a reference for what
"correct" looks like; do not credit the candidate for things only the gold answer does. Many
claims name a specific correct approach AND an incorrect "decoy" — credit only the correct one."""

JUDGE_SYS_FIN = """You are a strict grader for expert financial-analysis answers grounded in SEC
filings. You are given a question, a reference (gold) answer, the candidate answer to grade,
and a weighted rubric of claims.

For EACH claim, decide whether the CANDIDATE answer satisfies it:
  - score 1   : the candidate clearly and explicitly satisfies the claim
  - score 0   : it does not (vague references and hedged non-answers earn no credit)
  - score 0.5 : genuinely partial — only when the claim has separable parts and some are met

Numeric claims: credit requires the candidate's figure to match within roughly 1% relative
tolerance (or the tolerance the claim itself states). Judge ONLY what the candidate answer
actually says. The gold answer is a reference for what "correct" looks like; do not credit
the candidate for things only the gold answer does."""

# task name -> persona; unlisted tasks use the code persona.
JUDGE_SYS_BY_TASK = {"fav2": JUDGE_SYS_FIN}


# Forced structured output: the judge MUST answer through this tool, so per-claim
# verdicts always parse (score constrained to {0, 0.5, 1}).
VERDICT_TOOL = {
    "name": "submit_verdicts",
    "description": ("Submit your per-claim grading verdicts for the rubric. "
                    "One entry per claim_id, in rubric order."),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "score": {"type": "number", "enum": [0, 0.5, 1]},
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_id", "score", "reason"],
                },
            }
        },
        "required": ["verdicts"],
    },
}


class JudgeParseError(RuntimeError):
    """The judge response did not contain a usable verdicts payload."""


class JudgeRetryableError(RuntimeError):
    """Transient failure (429/5xx/connection/CLI) — worth a backoff retry."""


# ---------------------------------------------------------------- prompt build

def _sys_for_task(task: str | None) -> str:
    return JUDGE_SYS_BY_TASK.get(task or "", JUDGE_SYS_CODE)


def build_judge_prompt(question: str, candidate: str, rubric: list[dict],
                       gold_answer: str | None = None, task: str | None = None) -> str:
    """(question, candidate, weighted rubric) -> a single judge prompt string."""
    claims = "\n".join(
        f"- {c.get('claim_id', f'c{i}')} "
        f"(weight {c.get('weight', 1)}, {c.get('claim_type', 'claim')}): {c['statement']}"
        for i, c in enumerate(rubric)
    )
    gold = gold_answer if (gold_answer and gold_answer.strip()) else "[no reference provided]"
    cand = candidate if (candidate and candidate.strip()) else "[EMPTY ANSWER]"
    return f"""{_sys_for_task(task)}

## QUESTION
{question}

## GOLD (reference) ANSWER
{gold}

## CANDIDATE ANSWER (grade this)
{cand}

## RUBRIC CLAIMS
{claims}

Return your verdicts through the submit_verdicts tool, one entry per claim_id, in rubric
order. If you cannot use the tool, output ONLY strict JSON of the form
{{"verdicts":[{{"claim_id":"c1","score":1,"reason":"..."}}]}} and nothing else."""


def _claim_ids(rubric: list[dict]) -> list[str]:
    return [c.get("claim_id", f"c{i}") for i, c in enumerate(rubric)]


# --------------------------------------------------------- verdict validation

def _verdicts_from_entries(entries, rubric: list[dict]) -> dict[str, float]:
    """[{claim_id, score, ...}] -> {claim_id: score in {0,0.5,1}} (missing -> 0)."""
    ids = _claim_ids(rubric)
    scores: dict[str, float] = {}
    if not isinstance(entries, list) or not entries:
        raise JudgeParseError("verdicts array missing/empty")
    for v in entries:
        if not isinstance(v, dict):
            continue
        cid = v.get("claim_id")
        if cid not in ids:
            continue
        try:
            s = float(v.get("score"))
        except (TypeError, ValueError):
            continue
        scores[cid] = min(VALID_SCORES, key=lambda x: abs(x - s))  # snap to scale
    if not scores:
        raise JudgeParseError("no verdict matched any rubric claim_id")
    return {cid: scores.get(cid, 0.0) for cid in ids}


def _parse_free_json(text: str, rubric: list[dict]) -> dict[str, float]:
    """Tolerant free-text -> verdicts (for the CLI / non-tool path)."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise JudgeParseError("no JSON object in judge output")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"bad JSON in judge output: {e}") from e
    return _verdicts_from_entries(obj.get("verdicts"), rubric)


# ------------------------------------------------ canonical: Anthropic API path

def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise JudgeRetryableError("ANTHROPIC_API_KEY unset")
    return key


def _api_body(prompt: str, model: str, temperature: float, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": [VERDICT_TOOL],
        "tool_choice": {"type": "tool", "name": VERDICT_TOOL["name"]},
        "messages": [{"role": "user", "content": prompt}],
    }


def _extract_tool_input(content_blocks) -> dict:
    for block in content_blocks or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if name == VERDICT_TOOL["name"]:
                inp = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                if isinstance(inp, dict):
                    return inp
    raise JudgeParseError("no submit_verdicts tool_use block in judge response")


_SDK_CLIENT = None


def _api_vote(prompt: str, rubric: list[dict], model: str,
              temperature: float, max_tokens: int) -> dict[str, float]:
    """One forced-tool vote via the Anthropic Messages API (SDK if importable,
    else stdlib urllib). Raises Judge*Error on transient/parse failure."""
    global _SDK_CLIENT
    try:
        import anthropic  # noqa: PLC0415  (lazy, optional)
        have_sdk = True
    except ImportError:
        have_sdk = False

    if have_sdk:
        if _SDK_CLIENT is None:
            # max_retries=0: this module runs its own backoff loop (config judge.max_retries).
            # Leaving the SDK's retries on would nest two loops and multiply latency on 429s.
            _SDK_CLIENT = anthropic.Anthropic(api_key=_api_key(), max_retries=0)
        try:
            resp = _SDK_CLIENT.messages.create(**_api_body(prompt, model, temperature, max_tokens))
        except anthropic.RateLimitError as e:
            raise JudgeRetryableError(f"429: {e}") from e
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) >= 500:
                raise JudgeRetryableError(f"{e.status_code}: {e}") from e
            raise
        except anthropic.APIConnectionError as e:
            raise JudgeRetryableError(f"connection: {e}") from e
        blocks = [b.model_dump() if hasattr(b, "model_dump") else b for b in resp.content]
        return _verdicts_from_entries(_extract_tool_input(blocks).get("verdicts"), rubric)

    # stdlib fallback: raw HTTPS
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(_api_body(prompt, model, temperature, max_tokens)).encode(),
        headers={"content-type": "application/json", "x-api-key": _api_key(),
                 "anthropic-version": API_VERSION},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:  # noqa: BLE001
            pass
        if e.code == 429 or e.code >= 500:
            raise JudgeRetryableError(f"HTTP {e.code}: {body}") from e
        raise RuntimeError(f"HTTP {e.code} (non-retryable): {body}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise JudgeRetryableError(f"connection: {e}") from e
    return _verdicts_from_entries(_extract_tool_input(payload.get("content")).get("verdicts"), rubric)


# ---------------- canonical: OpenAI Chat Completions (direct or judge service)
#
# Mirrors the official judge's OpenAI backend (harness/judge_cli.py) so the
# agent's dev-time verdicts come off the identical wire shape: Structured
# Outputs response_format first (gpt-5.x reasoning models reject forced
# function tools), function-tool fallback, and bounded self-adaptation over the
# parameter axes that differ across model families. Endpoint resolution:
# $LEARNING_AGENT_JUDGE_URL (the operator-run judge service, harness/judge_service.py,
# which holds the provider key and PINS the judge model server-side — this is
# what a sealed workspace gets) else api.openai.com with $OPENAI_API_KEY.

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _strict_schema(schema):
    """Deep-copy a JSON schema into OpenAI Structured-Outputs strict form:
    additionalProperties:false and required=<all properties> at every object
    level (same tightening the official judge applies)."""
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(schema, list):
        return [_strict_schema(v) for v in schema]
    return schema


OPENAI_VERDICT_SCHEMA = _strict_schema(VERDICT_TOOL["input_schema"])
OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": VERDICT_TOOL["name"], "strict": True,
                    "schema": OPENAI_VERDICT_SCHEMA},
}
OPENAI_VERDICT_TOOL = {
    "type": "function",
    "function": {"name": VERDICT_TOOL["name"], "description": VERDICT_TOOL["description"],
                 "parameters": OPENAI_VERDICT_SCHEMA},
}

# Reasoning models spend the token budget on hidden reasoning BEFORE the answer,
# so a small cap starves the verdict JSON (empty content, finish_reason=length).
OPENAI_MIN_COMPLETION_TOKENS = 16384

# Per-model memo of the request shape that worked, so the adaptation
# round-trips are paid once, not on every one of the n_votes*n_questions calls.
_OPENAI_SHAPE: dict = {}


def _openai_endpoint() -> tuple[str, dict]:
    """(chat-completions URL, auth headers). Prefers the judge service."""
    service = os.environ.get("LEARNING_AGENT_JUDGE_URL", "").rstrip("/")
    if service:
        url = service + ("/chat/completions" if service.endswith("/v1")
                         else "/v1/chat/completions")
        headers = {"content-type": "application/json"}
        token = os.environ.get("LEARNING_AGENT_JUDGE_TOKEN", "")
        if token:
            headers["authorization"] = f"Bearer {token}"
        session = os.environ.get("LEARNING_AGENT_SESSION", "")
        if session:
            headers["x-lab-session"] = session
        return url, headers
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise JudgeRetryableError("neither LEARNING_AGENT_JUDGE_URL nor OPENAI_API_KEY set")
    return OPENAI_API_URL, {"content-type": "application/json",
                            "authorization": f"Bearer {key}"}


def _openai_request_body(prompt: str, model: str, temperature: float, max_tokens: int,
                         shape: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            shape["token_field"]: max(max_tokens, OPENAI_MIN_COMPLETION_TOKENS)}
    if shape["with_temperature"]:
        body["temperature"] = temperature
    if shape["mode"] == "json_schema":
        body["response_format"] = OPENAI_RESPONSE_FORMAT
    else:  # function-tool fallback
        body["tools"] = [OPENAI_VERDICT_TOOL]
        body["tool_choice"] = {"type": "function", "function": {"name": VERDICT_TOOL["name"]}}
    return body


def _openai_extract(payload: dict, mode: str) -> dict:
    """Chat Completions response -> the {'verdicts': [...]} dict."""
    try:
        msg = payload["choices"][0]["message"]
        finish = payload["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError) as e:
        raise JudgeParseError(f"malformed chat-completions response: {e}") from e
    if msg.get("refusal"):
        raise JudgeParseError(f"model refused: {str(msg['refusal'])[:200]}")
    if mode == "json_schema":
        content = msg.get("content") or ""
        if not content and finish == "length":
            raise JudgeRetryableError("empty content (finish_reason=length)")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise JudgeParseError(f"structured-output content not JSON: {e}") from e
    for c in msg.get("tool_calls") or []:
        fn = c.get("function", {}) if isinstance(c, dict) else {}
        if fn.get("name") == VERDICT_TOOL["name"]:
            try:
                return json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError as e:
                raise JudgeParseError(f"tool_call arguments not JSON: {e}") from e
    raise JudgeParseError("no submit_verdicts tool_call in judge response")


def _openai_vote(prompt: str, rubric: list[dict], model: str,
                 temperature: float, max_tokens: int) -> dict[str, float]:
    """One forced-structured-output vote over the OpenAI chat wire. Stdlib
    urllib. Self-adapts to model-family differences (max_completion_tokens vs
    max_tokens, reasoning models that reject `temperature`, models without
    Structured Outputs) and memoizes the shape that worked."""
    shape = dict(_OPENAI_SHAPE.get(
        model, {"mode": "json_schema", "token_field": "max_completion_tokens",
                "with_temperature": True}))
    for _ in range(6):  # bounded adaptation over the param/mode axes
        url, headers = _openai_endpoint()
        body = _openai_request_body(prompt, model, temperature, max_tokens, shape)
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                result = _openai_extract(json.loads(r.read().decode()), shape["mode"])
            _OPENAI_SHAPE[model] = shape  # remember what worked
            return _verdicts_from_entries(result.get("verdicts"), rubric)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode()[:500]
            except Exception:  # noqa: BLE001
                detail = ""
            low = detail.lower()
            if e.code == 400 and "max_completion_tokens" in low and shape["token_field"] == "max_completion_tokens":
                shape["token_field"] = "max_tokens"; continue
            if e.code == 400 and "'max_tokens'" in low and shape["token_field"] == "max_tokens":
                shape["token_field"] = "max_completion_tokens"; continue
            if e.code == 400 and "temperature" in low and shape["with_temperature"]:
                shape["with_temperature"] = False; continue
            if e.code == 400 and ("response_format" in low or "json_schema" in low
                                  or "structured output" in low) and shape["mode"] == "json_schema":
                shape["mode"] = "function"; continue
            if e.code == 400 and "function" in low and shape["mode"] == "function":
                shape["mode"] = "json_schema"; continue
            if e.code == 429 or e.code >= 500:
                raise JudgeRetryableError(f"HTTP {e.code}: {detail}") from e
            raise RuntimeError(f"HTTP {e.code} (non-retryable): {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise JudgeRetryableError(f"connection: {e}") from e
    raise RuntimeError("openai judge: exhausted parameter-adaptation retries")


# ------------------------------------------- non-canonical: claude-CLI fallback

def _cli_vote(prompt: str, rubric: list[dict], model: str, timeout: int = 300) -> dict[str, float]:
    """One vote via `claude --print --output-format json --model <model> <prompt>`.

    The CLI emits a JSON envelope; the assistant text is in `.result`, from which
    we parse the inner verdicts JSON. NON-CANONICAL — for hosts without a key."""
    if not shutil.which("claude"):
        raise JudgeRetryableError("claude CLI not on PATH")
    try:
        proc = subprocess.run(
            ["claude", "--print", "--output-format", "json", "--model", model, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise JudgeRetryableError(f"claude CLI timeout: {e}") from e
    if proc.returncode != 0:
        raise JudgeRetryableError(f"claude CLI rc={proc.returncode}: {proc.stderr[:300]}")
    raw = proc.stdout
    # Peel the CLI envelope (--output-format json) -> the model's text in .result.
    try:
        env = json.loads(raw)
        text = env.get("result", raw) if isinstance(env, dict) else raw
        if isinstance(env, dict) and env.get("is_error"):
            raise JudgeRetryableError(f"claude CLI error result: {str(text)[:200]}")
    except json.JSONDecodeError:
        text = raw  # some builds print bare text; parse it directly
    return _parse_free_json(text, rubric)


# ---------------------------------- non-canonical: deterministic offline mock

def _mock_vote(prompt: str, rubric: list[dict], model: str) -> dict[str, float]:
    """One DETERMINISTIC offline verdict — NO key/network/deps. For tests + reproducible
    smoke ONLY; NON-CANONICAL. Each claim's score is hash-derived from (model, claim_id,
    prompt) and snapped to {0, 0.5, 1}, so it is stable per input but is clearly a stub,
    never a real judgement."""
    scores: dict[str, float] = {}
    for cid in _claim_ids(rubric):
        h = hashlib.sha256(f"{model}|{cid}|{prompt}".encode("utf-8")).hexdigest()
        scores[cid] = VALID_SCORES[int(h[:8], 16) % len(VALID_SCORES)]
    return scores


# ------------------------------------------------------------ backend routing

def resolve_backend(requested: str) -> tuple[str, bool]:
    """(backend, canonical). 'auto' prefers the canonical OpenAI wire (the judge service /
    a key) — the same instrument official scoring uses — then the Anthropic API, then the
    non-canonical claude CLI. 'mock' is the deterministic offline stub (never auto-selected
    — must be requested explicitly)."""
    if requested == "auto":
        if os.environ.get("LEARNING_AGENT_JUDGE_URL") or os.environ.get("OPENAI_API_KEY"):
            return "openai", True
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "api", True
        if shutil.which("claude"):
            return "cli", False
        raise SystemExit("no judge backend: set LEARNING_AGENT_JUDGE_URL (judge service) or an API key "
                         "(OPENAI_API_KEY / ANTHROPIC_API_KEY), or install the `claude` CLI "
                         "(or use --backend mock for a deterministic offline stub)")
    if requested == "openai":
        if not (os.environ.get("LEARNING_AGENT_JUDGE_URL") or os.environ.get("OPENAI_API_KEY")):
            raise SystemExit("backend 'openai' needs LEARNING_AGENT_JUDGE_URL (the operator judge "
                             "service) or OPENAI_API_KEY (or use --backend auto)")
        return "openai", True
    if requested == "api":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("backend 'api' needs ANTHROPIC_API_KEY (or use --backend cli/auto)")
        return "api", True
    if requested == "cli":
        if not shutil.which("claude"):
            raise SystemExit("backend 'cli' needs the `claude` CLI on PATH")
        return "cli", False
    if requested == "mock":
        return "mock", False
    raise SystemExit(f"unknown backend {requested!r}")


def _vote_once(backend: str, prompt: str, rubric: list[dict], model: str,
               temperature: float, max_tokens: int) -> dict[str, float]:
    if backend == "openai":
        return _openai_vote(prompt, rubric, model, temperature, max_tokens)
    if backend == "api":
        return _api_vote(prompt, rubric, model, temperature, max_tokens)
    if backend == "cli":
        return _cli_vote(prompt, rubric, model)
    if backend == "mock":
        return _mock_vote(prompt, rubric, model)
    raise SystemExit(f"unknown backend {backend!r}")


def _vote_with_retries(backend: str, prompt: str, rubric: list[dict], model: str,
                       temperature: float, max_tokens: int, max_retries: int) -> dict[str, float]:
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _vote_once(backend, prompt, rubric, model, temperature, max_tokens)
        except (JudgeRetryableError, JudgeParseError) as e:
            last = e
            if attempt < max_retries:
                delay = min(2.0 * (2 ** attempt), 30.0)
                time.sleep(delay)
    raise RuntimeError(f"judge vote failed after {max_retries + 1} attempts: {last}") from last


# --------------------------------------------------- self-consistency + weights

def combine_votes(votes: list[float]) -> float:
    """Majority vote over {0, 0.5, 1}; median breaks ties deterministically."""
    if not votes:
        return 0.0
    counts = Counter(votes).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return float(counts[0][0])
    return float(statistics.median(sorted(votes)))


def weighted_score(rubric: list[dict], verdicts: dict[str, float]) -> float:
    """Weighted sum / total weight -> claim score in [0, 1]."""
    total_w = sum(c.get("weight", 1) for c in rubric) or 1
    ids = _claim_ids(rubric)
    return sum(verdicts.get(cid, 0.0) * c.get("weight", 1)
               for cid, c in zip(ids, rubric)) / total_w


# ------------------------------------------------------------- public entry

def judge_claims(question: str, candidate: str, rubric: list[dict],
                 gold_answer: str | None = None, task: str | None = None,
                 n_votes: int = 3, model: str | None = None,
                 backend: str = "auto", temperature: float = 0.0,
                 max_tokens: int = 4096, max_retries: int = 3) -> dict:
    """Judge one candidate against a weighted-claim rubric with n-vote majority.

    Returns:
        {
          "claim_score": float in [0,1],   # weighted, majority-combined
          "per_claim": {cid: {"weight", "final", "votes":[...]}},
          "n_votes", "backend", "canonical", "model",
        }
    """
    if not rubric:
        raise ValueError("rubric must be a non-empty list of claims")
    model = resolve_judge_model(model, backend)
    backend, canonical = resolve_backend(backend)
    prompt = build_judge_prompt(question, candidate, rubric, gold_answer, task)
    ids = _claim_ids(rubric)
    votes: dict[str, list[float]] = {cid: [] for cid in ids}
    for _ in range(max(1, n_votes)):
        one = _vote_with_retries(backend, prompt, rubric, model, temperature,
                                 max_tokens, max_retries)
        for cid in ids:
            votes[cid].append(one.get(cid, 0.0))
    final = {cid: combine_votes(vs) for cid, vs in votes.items()}
    per_claim = {
        cid: {"weight": c.get("weight", 1), "final": final[cid], "votes": votes[cid]}
        for cid, c in zip(ids, rubric)
    }
    return {
        "claim_score": round(weighted_score(rubric, final), 4),
        "per_claim": per_claim,
        "n_votes": n_votes,
        "backend": backend,
        "canonical": canonical,
        "model": model,
    }


# ------------------------------------------------------------------- CLI

def _load_row(args) -> tuple[str, str, list[dict], str | None]:
    if args.input:
        if not os.path.isfile(args.input):
            raise SystemExit(f"[judge] --input file not found: {args.input}")
        with open(args.input, encoding="utf-8") as f:
            row = json.load(f)
        q = row["question"]
        cand = row.get("candidate", row.get("answer", ""))
        rubric = row["rubric"]
        gold = row.get("gold_answer")
        return q, cand, rubric, gold
    if not (args.question and args.rubric is not None):
        raise SystemExit("provide --input FILE, or both --question and --rubric")
    rubric = json.loads(args.rubric)
    return args.question, (args.candidate or ""), rubric, args.gold_answer


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge verifier over a weighted-claim rubric.")
    ap.add_argument("--input", help="JSON file: {question, candidate|answer, rubric, gold_answer?}")
    ap.add_argument("--question")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--rubric", help='JSON list: [{"claim_id","weight","statement"}...]')
    ap.add_argument("--gold-answer", default=None)
    ap.add_argument("--task", default=None, help="fav2 uses the finance persona; else code")
    # n-votes / temperature / max-tokens / max-retries mirror bench/config.yaml judge.* —
    # the canonical rationale for each value lives there (this CLI is a single-question probe).
    ap.add_argument("--n-votes", type=int, default=3,
                    help="self-consistency votes per question, majority-voted (odd count breaks ties)")
    ap.add_argument("--model", default=None,
                    help="judge model id (or set $JUDGE_MODEL); none baked in "
                         "(--backend mock defaults it to the 'mock' sentinel)")
    ap.add_argument("--backend", default="auto", choices=["auto", "openai", "api", "cli", "mock"],
                    help="openai = canonical wire (judge service via $LEARNING_AGENT_JUDGE_URL, or "
                         "$OPENAI_API_KEY direct); mock = deterministic offline stub "
                         "(no key/network, for tests)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    q, cand, rubric, gold = _load_row(args)
    result = judge_claims(q, cand, rubric, gold_answer=gold, task=args.task,
                          n_votes=args.n_votes, model=args.model, backend=args.backend,
                          temperature=args.temperature, max_tokens=args.max_tokens,
                          max_retries=args.max_retries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
