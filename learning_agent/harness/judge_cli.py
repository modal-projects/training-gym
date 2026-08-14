"""Learning Agent canonical, reproducible JUDGE.

This is the ONE judge for the benchmark. It turns a run's candidate answers into a
score by, per question:
  0. verifying the INTEGRITY LOCK (bench/pins.json) — a drifted harness/spec/gold
     surface REFUSES to score (--allow-dirty stamps integrity:"DIRTY" instead),
  1. building the PINNED weighted-claim judge prompt (harness/judge.build_judge_prompt),
  2. getting per-claim 1 / 0 / 0.5 verdicts from the CANONICAL backend — a direct
     Anthropic Messages API call with the pinned snapshot model, temperature 0, and
     FORCED structured output (tool_choice + input_schema, so verdicts always parse;
     no free-text JSON hoping). N_VOTES independent votes are taken per question
     (self-consistency); the per-claim final score is the majority vote, median on
     ties. All votes are persisted for audit. 429/5xx/parse failures retry with
     backoff; a question that still fails is failed=true — never a silent 0.
  3. DETERMINISTICALLY aggregating (weighted sum / total weight) into a claim_score,
  4. computing the SECONDARY deterministic metric (python: code compiles + no
     hallucinated dspy.* APIs via grader.py/api_surface; typescript: fraction of
     referenced symbols that are real OpenClaw symbols).

NON-CANONICAL fallback backends `cli-claude` (`claude -p`) and `cli-codex`
(`codex exec`) are kept for machines without an API key; results they produce are
stamped canonical:false in provenance and are not comparable headline numbers.

It does NOT fold the deterministic check into the headline score (grade_mode:
lenient) — the rubric is the score; the secondary is reported alongside.

Outputs:
  runs/<tag>/budget_<N>/results_<split>.json   per-question + mean + bootstrap_ci95
                                               + failed + PROVENANCE block
  runs/<tag>/budget_<N>/verdicts_<split>/      per-question final verdicts + ALL votes
  runs/LEADERBOARD.jsonl                       one appended row (skipped by --no-record;
                                               REFUSED for all-failed runs — mean is
                                               null, never 0.0 — and --limit runs,
                                               which imply --no-record)

A --limit smoke run writes NEITHER of the two artifacts above; it writes
  runs/<tag>/budget_<N>/smoke_results_<split>_limit<N>.json
  runs/<tag>/budget_<N>/smoke_verdicts_<split>_limit<N>/
so a truncated run can never clobber a completed full run's results/verdicts, and
the `smoke_` prefix keeps it out of the `results_*.json` glob the observatory
collector uses (a suffixed name would still surface an n=2 smoke as a scored row).

Usage:
  python harness/judge_cli.py --task dspy --tag <run_tag> --split dev
  python harness/judge_cli.py --task openclaw --tag <run_tag> --split test [--budget N]
  python harness/judge_cli.py ... --no-record            # smoke test, no leaderboard row
  python harness/judge_cli.py ... --backend cli-claude   # NON-canonical fallback

The judge model / backend / votes / retries / seed all come from bench/config.yaml
unless overridden on the CLI. test.json is HIDDEN gold — read here, never shown to
the student. The bootstrap CI uses a FIXED seed from config so it reproduces
bit-identically.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import judge as J        # noqa: E402  (build_judge_prompt, parse_verdict)
import grader as G       # noqa: E402  (deterministic_report, load_api_surface)
import integrity as I    # noqa: E402  (verify_pins, judge_prompt_sha)
import envfile           # noqa: E402  (.env -> ANTHROPIC_API_KEY etc., non-overriding)

envfile.load_env(ROOT)

try:
    import config as labcfg  # noqa: E402  (harness/config.py — the one config loader)
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyyaml required: pip install pyyaml") from e

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# Forced structured output: the judge MUST answer through this tool, so per-claim
# verdicts always parse (score constrained to the {0, 0.5, 1} rubric scale).
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

VALID_SCORES = (0.0, 0.5, 1.0)


def load_config() -> dict:
    """Combined view {"global": …, "tasks": {…}} from bench/config.yaml +
    every task_configs/<T>.yaml (see harness/config.py)."""
    return labcfg.load_config(ROOT)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JudgeParseError(RuntimeError):
    """The judge response did not contain a usable verdicts payload."""


class JudgeRetryableError(RuntimeError):
    """Transient failure (429/5xx/connection) — worth a backoff retry."""


# ---------- canonical backend: direct Anthropic Messages API ----------

def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise SystemExit(
            "backend 'api' (canonical) needs ANTHROPIC_API_KEY in the environment.\n"
            "Export it, or run NON-canonically with --backend cli-claude / cli-codex\n"
            "(results will be stamped canonical:false).")
    return key


def _api_request_body(prompt: str, model: str, temperature: float, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "tools": [VERDICT_TOOL],
        "tool_choice": {"type": "tool", "name": VERDICT_TOOL["name"]},
        "messages": [{"role": "user", "content": prompt}],
    }


def _extract_tool_input(content_blocks) -> dict:
    """Pull the forced tool_use input out of a Messages API response content list."""
    for block in content_blocks or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if name == VERDICT_TOOL["name"]:
                inp = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                if isinstance(inp, dict):
                    return inp
    raise JudgeParseError("no submit_verdicts tool_use block in judge response")


# One SDK client, created lazily (SDK retries disabled — WE own retry policy so the
# configured max_retries governs both the SDK and the urllib path identically).
_SDK_CLIENT = None


def _api_call_once(prompt: str, model: str, temperature: float, max_tokens: int) -> dict:
    """One Messages API call -> the tool_use input dict. Uses the anthropic SDK when
    importable, else stdlib urllib HTTPS — no new hard dependency either way."""
    global _SDK_CLIENT
    try:
        import anthropic
        have_sdk = True
    except ImportError:
        have_sdk = False

    if have_sdk:
        if _SDK_CLIENT is None:
            _SDK_CLIENT = anthropic.Anthropic(api_key=_api_key(), max_retries=0)
        try:
            resp = _SDK_CLIENT.messages.create(**_api_request_body(prompt, model, temperature, max_tokens))
        except anthropic.RateLimitError as e:
            raise JudgeRetryableError(f"429 rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise JudgeRetryableError(f"{e.status_code} server error: {e}") from e
            raise
        except anthropic.APIConnectionError as e:
            raise JudgeRetryableError(f"connection error: {e}") from e
        blocks = [b.model_dump() if hasattr(b, "model_dump") else b for b in resp.content]
        return _extract_tool_input(blocks)

    # stdlib fallback: raw HTTPS
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(_api_request_body(prompt, model, temperature, max_tokens)).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": _api_key(),
            "anthropic-version": API_VERSION,
        },
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
        raise JudgeRetryableError(f"connection error: {e}") from e
    return _extract_tool_input(payload.get("content"))


def _verdicts_from_tool_input(tool_input: dict, row: dict) -> dict[str, float]:
    """Validated {claim_id: score in {0,0.5,1}}. Missing claims default to 0
    (unsatisfied) — same semantics as judge.parse_verdict."""
    claim_ids = [c["claim_id"] for c in row["rubric"]]
    scores: dict[str, float] = {}
    entries = tool_input.get("verdicts")
    if not isinstance(entries, list) or not entries:
        raise JudgeParseError("verdicts array missing/empty in tool input")
    for v in entries:
        if not isinstance(v, dict):
            continue
        cid = v.get("claim_id")
        if cid not in claim_ids:
            continue
        try:
            s = float(v.get("score"))
        except (TypeError, ValueError):
            continue
        # forced schema constrains score to {0, 0.5, 1}; snap defensively anyway
        scores[cid] = min(VALID_SCORES, key=lambda x: abs(x - s))
    if not scores:
        raise JudgeParseError("no verdict matched any rubric claim_id")
    return {cid: scores.get(cid, 0.0) for cid in claim_ids}


# ---------- alternate canonical backend: OpenAI Chat Completions ----------

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _strict_schema(schema):
    """Deep-copy a JSON schema into OpenAI Structured-Outputs strict form:
    additionalProperties:false and required=<all properties> at every object
    level. The verdict schema already lists the same required set, so this only
    tightens it — the grading contract is unchanged."""
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(schema, list):
        return [_strict_schema(v) for v in schema]
    return schema


# Same verdict contract as VERDICT_TOOL, expressed for OpenAI two ways: the
# modern Structured-Outputs response_format (primary — works for gpt-5.x
# reasoning models, which reject forced function tools) and a function tool
# (fallback for older models without Structured Outputs).
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

# Per-model memo of the request shape that worked, so we don't re-pay the
# adaptation round-trips (temperature/token-field/mode) on every one of the
# ~n_votes*n_questions calls.
_OPENAI_SHAPE: dict = {}


def _openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit(
            "backend 'openai' (canonical) needs OPENAI_API_KEY in the environment.\n"
            "Put it in .env (see .env.example), or run NON-canonically with\n"
            "--backend cli-claude / cli-codex (results stamped canonical:false).")
    return key


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
    """Parse a Chat Completions response into the {'verdicts': [...]} dict the
    Anthropic path also yields, so _verdicts_from_tool_input handles both."""
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
            raise JudgeRetryableError("empty content (finish_reason=length): raise max_completion_tokens")
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


def _openai_call_once(prompt: str, model: str, temperature: float, max_tokens: int) -> dict:
    """One OpenAI Chat Completions call -> the verdicts dict. Stdlib urllib (no
    hard dep). Self-adapts to model-family differences — max_completion_tokens vs
    max_tokens, reasoning models that reject `temperature`, and models without
    Structured Outputs (falls back to a function tool) — then memoizes the shape
    that worked in _OPENAI_SHAPE[model]."""
    shape = dict(_OPENAI_SHAPE.get(
        model, {"mode": "json_schema", "token_field": "max_completion_tokens",
                "with_temperature": True}))
    for _ in range(6):  # bounded adaptation over the param/mode axes
        body = _openai_request_body(prompt, model, temperature, max_tokens, shape)
        req = urllib.request.Request(
            OPENAI_API_URL, data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {_openai_api_key()}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                result = _openai_extract(json.loads(r.read().decode()), shape["mode"])
            _OPENAI_SHAPE[model] = shape  # remember what worked
            return result
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
            raise JudgeRetryableError(f"connection error: {e}") from e
    raise RuntimeError("openai judge: exhausted parameter-adaptation retries")


# ---------- NON-canonical fallback backends (headless CLI sessions) ----------

def _run_claude(prompt: str, model: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["claude", "-p", "--model", model, prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise JudgeRetryableError(f"claude -p failed (rc={proc.returncode}): {proc.stderr[:500]}")
    return proc.stdout


def _run_codex(prompt: str, model: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["codex", "exec", "--model", model, prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise JudgeRetryableError(f"codex exec failed (rc={proc.returncode}): {proc.stderr[:500]}")
    return proc.stdout


def _have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None


def _cli_vote(runner, prompt: str, model: str, row: dict) -> dict[str, float]:
    raw = runner(prompt, model)
    verdicts = J.parse_verdict(raw, row)
    # parse_verdict returns all-zeros when nothing parsed; treat a response that
    # matched NO claim as a parse failure so it retries instead of silently zeroing.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m is None:
        raise JudgeParseError("no JSON object in CLI judge output")
    return verdicts


def resolve_backend(requested: str) -> str:
    """Map legacy aliases and validate availability. Never silently degrades the
    canonical 'api' backend to a CLI — that must be an explicit operator choice."""
    aliases = {"claude": "cli-claude", "codex": "cli-codex"}
    backend = aliases.get(requested, requested)
    if backend == "api":
        _api_key()  # hard requirement, clear error otherwise
        return backend
    if backend == "openai":
        _openai_api_key()  # hard requirement, clear error otherwise
        return backend
    if backend == "cli-claude" and _have("claude"):
        return backend
    if backend == "cli-codex" and _have("codex"):
        return backend
    # requested CLI missing -> try the other CLI (both are equally non-canonical)
    if backend in ("cli-claude", "cli-codex"):
        if _have("claude"):
            return "cli-claude"
        if _have("codex"):
            return "cli-codex"
        raise SystemExit("no CLI judge backend available (need `claude` or `codex` on PATH)")
    raise SystemExit(f"unknown backend {requested!r}")


def judge_once(backend: str, prompt: str, row: dict, model: str,
               temperature: float, max_tokens: int) -> dict[str, float]:
    """One judge vote -> {claim_id: score}. Raises JudgeRetryableError/JudgeParseError."""
    if backend == "api":
        return _verdicts_from_tool_input(
            _api_call_once(prompt, model, temperature, max_tokens), row)
    if backend == "openai":
        return _verdicts_from_tool_input(
            _openai_call_once(prompt, model, temperature, max_tokens), row)
    if backend == "cli-claude":
        return _cli_vote(_run_claude, prompt, model, row)
    if backend == "cli-codex":
        return _cli_vote(_run_codex, prompt, model, row)
    raise SystemExit(f"unknown backend {backend!r}")


def judge_with_retries(backend: str, prompt: str, row: dict, model: str,
                       temperature: float, max_tokens: int, max_retries: int) -> dict[str, float]:
    """One vote with backoff on 429/5xx/connection/parse failures."""
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return judge_once(backend, prompt, row, model, temperature, max_tokens)
        except (JudgeRetryableError, JudgeParseError, subprocess.TimeoutExpired) as e:
            last = e
            if attempt < max_retries:
                delay = min(2.0 * (2 ** attempt), 30.0)
                print(f"    retry {attempt + 1}/{max_retries} in {delay:.0f}s ({str(e)[:100]})")
                time.sleep(delay)
    raise RuntimeError(f"judge failed after {max_retries + 1} attempts: {last}") from last


# ---------- self-consistency ----------

def combine_votes(votes: list[float]) -> float:
    """Majority vote over {0, 0.5, 1}; median breaks ties deterministically."""
    counts = Counter(votes).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return float(counts[0][0])
    return float(statistics.median(sorted(votes)))


# ---------- aggregation ----------

def aggregate_claims(row: dict, verdicts: dict[str, float]) -> float:
    """Weighted sum / total weight. verdicts = {claim_id: score in [0,1]}."""
    total_w = sum(c["weight"] for c in row["rubric"]) or 1
    return sum(verdicts.get(c["claim_id"], 0.0) * c["weight"] for c in row["rubric"]) / total_w


# ---------- secondary deterministic metric ----------

_TS_FENCE = re.compile(r"```(?:typescript|ts|tsx|js|javascript)?\s*\n(.*?)```",
                       re.DOTALL | re.IGNORECASE)
# A `foo.bar(`-style member access or a bare PascalCase/identifier token reference.
_TS_IDENT = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\b")
_TS_RESERVED = {
    "const", "let", "var", "function", "return", "if", "else", "for", "while", "new",
    "await", "async", "import", "export", "from", "default", "class", "extends", "this",
    "super", "true", "false", "null", "undefined", "void", "typeof", "instanceof", "in",
    "of", "try", "catch", "finally", "throw", "switch", "case", "break", "continue",
    "type", "interface", "enum", "namespace", "as", "is", "keyof", "readonly", "public",
    "private", "protected", "static", "get", "set", "yield", "do", "delete", "string",
    "number", "boolean", "any", "unknown", "never", "object", "Promise", "Array",
    "console", "log", "error", "warn", "Math", "JSON", "Object", "then", "catch",
    "map", "filter", "forEach", "push", "length", "constructor",
}


def python_secondary(answer: str, api: dict) -> dict:
    """DSPy: code compiles (AST) + no hallucinated dspy.* APIs. Reused from grader."""
    det = G.deterministic_report(answer, api)
    # secondary score: 1.0 if has code that compiles & no halluc; else fractional/None
    if not det["has_code"]:
        sec = None
    else:
        sec = 1.0
        if det["compiles"] is False:
            sec = 0.0
        elif det["hallucinated_apis"]:
            sec = 0.5
    return {"kind": "python_compiles", "score": sec, "detail": det}


def ts_secondary(answer: str, symbols: set[str]) -> dict:
    """OpenClaw: fraction of referenced (non-reserved) symbols that are real."""
    blocks = _TS_FENCE.findall(answer or "")
    if not blocks:
        return {"kind": "ts_symbol_grounding", "score": None,
                "detail": {"has_code": False, "n_refs": 0, "n_real": 0, "unknown": []}}
    refs: set[str] = set()
    for b in blocks:
        for m in _TS_IDENT.findall(b):
            if m in _TS_RESERVED or len(m) < 3 or m.islower():
                # skip reserved words, short tokens, and all-lowercase locals/keywords;
                # OpenClaw real symbols of interest are PascalCase / camelCase API names.
                continue
            refs.add(m)
    if not refs:
        return {"kind": "ts_symbol_grounding", "score": None,
                "detail": {"has_code": True, "n_refs": 0, "n_real": 0, "unknown": []}}
    real = {r for r in refs if r in symbols}
    unknown = sorted(refs - real)
    return {"kind": "ts_symbol_grounding",
            "score": round(len(real) / len(refs), 4),
            "detail": {"has_code": True, "n_refs": len(refs), "n_real": len(real),
                       "unknown": unknown[:40]}}


_FIN_ACCESSION = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
# Corpus-relative filing path: TICKER/<filing_date>_<FORM>_<accession>.txt
_FIN_PATH = re.compile(r"\b([A-Z][A-Z0-9.]{0,6}/\d{4}-\d{2}-\d{2}_[A-Za-z0-9-]+_\d{10}-\d{2}-\d{6}\.txt)\b")
# Exchange-prefixed tickers (unprefixed uppercase words are too noisy: AI, CEO, GAAP...)
_FIN_TICKER = re.compile(r"\b(?:NYSE|NASDAQ|Nasdaq|AMEX)\s*:\s*([A-Z]{1,5}(?:\.[A-Z])?)\b")


def fin_secondary(answer: str, refs: dict) -> dict:
    """fav2: fraction of explicit EDGAR references in the answer (accession numbers,
    corpus filing paths, exchange-prefixed tickers) that resolve against the pinned
    pack manifest. An answer with no explicit references scores None (reported, not
    penalized) — mirroring the has_code=False case of the code secondaries."""
    text = answer or ""
    paths = set(_FIN_PATH.findall(text))
    # don't double-count an accession already cited inside a path (the path format is
    # <date>_<FORM>_<accession>.txt, so slice it out structurally — \b can't match after '_')
    in_paths = {p.rsplit("_", 1)[-1].removesuffix(".txt") for p in paths}
    accs = set(_FIN_ACCESSION.findall(text)) - in_paths
    ticks = set(_FIN_TICKER.findall(text))
    checks = ([("path", p, p in refs["paths"]) for p in sorted(paths)]
              + [("accession", a, a in refs["accessions"]) for a in sorted(accs)]
              + [("ticker", t, t in refs["tickers"]) for t in sorted(ticks)])
    if not checks:
        return {"kind": "edgar_ref_grounding", "score": None,
                "detail": {"has_refs": False, "n_refs": 0, "n_real": 0, "unknown": []}}
    n_real = sum(1 for _, _, ok in checks if ok)
    unknown = [f"{kind}:{val}" for kind, val, ok in checks if not ok]
    return {"kind": "edgar_ref_grounding",
            "score": round(n_real / len(checks), 4),
            "detail": {"has_refs": True, "n_refs": len(checks), "n_real": n_real,
                       "unknown": unknown[:40]}}


def _is_degenerate(ans: str) -> bool:
    return not ans or not ans.strip() or len(ans.strip()) < 3


# ---------- bootstrap CI (SEEDED: reproduces bit-identically for a fixed seed) ----------

def bootstrap_ci95(values: list[float], resamples: int = 10000, seed: int = 0) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [round(values[0], 4), round(values[0], 4)]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples)]
    return [round(lo, 4), round(hi, 4)]


# ---------- integrity self-check ----------

def _verify_verifier(root: Path, pins_file: Path) -> list[str]:
    """Verify the integrity VERIFIER itself: hash harness/integrity.py directly
    against bench/pins.json WITHOUT calling integrity.verify_pins — a neutered
    verifier must never get to vouch for itself. Returns [] when clean."""
    rel = "harness/integrity.py"
    if not pins_file.exists():
        return []  # verify_pins reports the missing pins file itself
    try:
        pinned = json.loads(pins_file.read_text()).get("files", {})
    except (OSError, json.JSONDecodeError) as e:
        return [f"unreadable pins file {pins_file}: {e}"]
    want = pinned.get(rel)
    p = root / rel
    if want is None:
        return [f"{rel}: not pinned in {pins_file.name} — re-freeze deliberately"]
    if not p.exists():
        return [f"{rel}: pinned but missing on disk"]
    if _sha256_file(p) != want:
        return [f"{rel}: sha256 mismatch — the integrity verifier itself was modified"]
    return []


# ---------- provenance ----------

def build_provenance(args, backend: str, canonical: bool, tcfg: dict, jcfg: dict,
                     seed: int, integrity_status: str) -> dict:
    gold_path = ROOT / tcfg[args.split]
    return {
        "judge_model": args.judge_model,
        "pinned_judge_model": jcfg["model"],
        "judge_backend": backend,
        "canonical": canonical,
        "n_votes": args.n_votes,
        "judge_prompt_sha": I.judge_prompt_sha(ROOT),
        "harness_sha": _sha256_file(ROOT / "harness" / "eval.py"),
        "sys_sha": _sha256_file(ROOT / tcfg["sys"]),
        "gold_sha": _sha256_file(gold_path),
        "config_sha": labcfg.config_sha(ROOT, args.task),
        "corpus_pin": tcfg.get("corpus_version") or tcfg.get("corpus_commit") or "",
        "budget": args.budget,
        "limit": args.limit,
        "eval_temperature": 0.0,
        "seed": seed,
        "integrity": integrity_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------- main ----------

def main():
    cfg = load_config()
    g = cfg["global"]
    jcfg = g["judge"]
    ap = argparse.ArgumentParser(description="Learning Agent canonical reproducible judge.")
    ap.add_argument("--task", required=True, choices=list(cfg["tasks"]))
    ap.add_argument("--tag", required=True, help="run tag under runs/<tag>/")
    ap.add_argument("--split", required=True, choices=g["splits"])
    ap.add_argument("--budget", type=int, default=g["eval_budget"])
    ap.add_argument("--judge-model", default=jcfg["model"])
    ap.add_argument("--backend", default=jcfg.get("backend", "api"),
                    choices=["api", "openai", "cli-claude", "cli-codex", "claude", "codex"],
                    help="api (Anthropic) / openai = CANONICAL; cli-* = NON-canonical "
                         "fallbacks (claude/codex are legacy aliases for cli-*)")
    ap.add_argument("--n-votes", type=int, default=int(jcfg.get("n_votes", 3)),
                    help="self-consistency votes per question (majority/median)")
    ap.add_argument("--limit", type=int, default=0,
                    help="judge only first N questions (smoke test): implies --no-record "
                         "and writes smoke_* artifacts, never the full-run results")
    ap.add_argument("--no-record", action="store_true",
                    help="skip the LEADERBOARD.jsonl append (smoke tests)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="score despite integrity-pin mismatches; stamps integrity:DIRTY "
                         "into provenance and the leaderboard row")
    args = ap.parse_args()

    # --limit is a smoke test by definition: a truncated run must NEVER become a
    # leaderboard row (cherry-picking vector), so it implies --no-record. Its
    # results/verdicts also go to smoke_* paths (see below) so they cannot overwrite
    # a completed full run's artifacts.
    if args.limit and not args.no_record:
        print(f"[judge] --limit {args.limit} is a smoke test: truncated runs are "
              "never recorded (implying --no-record)")
        args.no_record = True

    # ---- INTEGRITY GATE: refuse to score a drifted benchmark surface ----
    # First verify the VERIFIER by direct hash (independent of integrity.verify_pins,
    # which a tampered integrity.py could neuter), then run the full pin check.
    pins_file = ROOT / g.get("pins", "bench/pins.json")
    mismatches = _verify_verifier(ROOT, pins_file) + I.verify_pins(ROOT)
    if mismatches and not args.allow_dirty:
        msg = "\n".join(f"  - {m}" for m in mismatches)
        raise SystemExit(
            "[integrity] REFUSING to judge: benchmark surface does not match bench/pins.json:\n"
            f"{msg}\n"
            "If the change is deliberate, re-freeze (`python bench.py freeze`).\n"
            "To score anyway (stamped integrity:DIRTY), pass --allow-dirty.")
    integrity_status = "OK" if not mismatches else "DIRTY"
    if mismatches:
        print(f"[integrity] WARNING: scoring with {len(mismatches)} pin mismatch(es); "
              "results stamped DIRTY")

    backend = resolve_backend(args.backend)
    # canonical requires BOTH the api backend AND the pinned judge snapshot from
    # bench/config.yaml — an arbitrary --judge-model is not a comparable number.
    pinned_judge_model = jcfg["model"]
    canonical = backend in ("api", "openai") and args.judge_model == pinned_judge_model
    if backend not in ("api", "openai"):
        print(f"[judge] WARNING: backend {backend} is NON-CANONICAL "
              "(results stamped canonical:false)")
    elif not canonical:
        print(f"[judge] WARNING: judge model {args.judge_model!r} != pinned snapshot "
              f"{pinned_judge_model!r} (bench/config.yaml); results stamped canonical:false")

    temperature = float(jcfg.get("temperature", 0.0))
    max_tokens = int(jcfg.get("max_tokens", 4096))
    max_retries = int(jcfg.get("max_retries", 3))
    seed = int(jcfg.get("bootstrap_seed", 0))

    tcfg = cfg["tasks"][args.task]
    bdir = ROOT / g["runs_dir"] / args.tag / f"budget_{args.budget}"
    cand_path = bdir / "candidates.json"
    if not cand_path.exists():
        raise SystemExit(f"no candidates at {cand_path}")
    candidates = json.loads(cand_path.read_text())

    gold_rows = json.loads((ROOT / tcfg[args.split]).read_text())
    if args.limit:
        gold_rows = gold_rows[: args.limit]

    # A truncated run writes to its own paths: it must never clobber the results or
    # per-question verdicts of a completed full run at the same tag/budget/split.
    # The `smoke_` PREFIX is deliberate (not a `_limit<N>` suffix): the observatory
    # collector globs `results_*.json` and derives the split from the filename, so a
    # suffixed name would still surface an n=2 smoke as a scored result row.
    if args.limit:
        results_name = f"smoke_results_{args.split}_limit{args.limit}.json"
        verdicts_name = f"smoke_verdicts_{args.split}_limit{args.limit}"
    else:
        results_name = f"results_{args.split}.json"
        verdicts_name = f"verdicts_{args.split}"

    # secondary-metric resources
    api = symbols = refs = None
    if tcfg["secondary"] == "python_compiles":
        api = G.load_api_surface(ROOT / tcfg["api_surface"])
    elif tcfg["secondary"] == "ts_symbol_grounding":
        symbols = set(json.loads((ROOT / tcfg["symbols"]).read_text())["symbols"])
    elif tcfg["secondary"] == "edgar_ref_grounding":
        refs = {k: set(v) for k, v in json.loads((ROOT / tcfg["refs"]).read_text()).items()}
    elif tcfg["secondary"] in (None, "none", ""):
        pass  # task has no deterministic secondary metric (rubric is the only score)
    else:
        raise SystemExit(f"unknown secondary metric {tcfg['secondary']!r} for task {args.task}")

    print(f"[judge] task={args.task} tag={args.tag} split={args.split} budget={args.budget} "
          f"model={args.judge_model} backend={backend} canonical={canonical} "
          f"n_votes={args.n_votes} n={len(gold_rows)} integrity={integrity_status}")

    vdir = bdir / verdicts_name
    vdir.mkdir(parents=True, exist_ok=True)

    per_question = {}
    claim_scores = []
    failed = []
    for r in gold_rows:
        qid = r["id"]
        ans = candidates.get(qid, "")
        # secondary first (cheap, deterministic)
        if tcfg["secondary"] == "python_compiles":
            sec = python_secondary(ans, api)
        elif tcfg["secondary"] == "edgar_ref_grounding":
            sec = fin_secondary(ans, refs)
        elif tcfg["secondary"] == "ts_symbol_grounding":
            sec = ts_secondary(ans, symbols)
        else:
            sec = {"kind": "none", "score": None, "detail": {}}  # no secondary for this task

        if _is_degenerate(ans):
            per_question[qid] = {"claim_score": None, "secondary": sec,
                                 "verdicts": {}, "votes": {}, "failed": True}
            failed.append(qid)
            print(f"  {qid[:22]:22} FAILED (empty/degenerate candidate)", flush=True)
            continue

        prompt = J.build_judge_prompt(r, ans, J.JUDGE_SYS_BY_TASK.get(args.task))
        claim_ids = [c["claim_id"] for c in r["rubric"]]
        votes: dict[str, list[float]] = {cid: [] for cid in claim_ids}
        try:
            for _k in range(args.n_votes):
                one = judge_with_retries(backend, prompt, r, args.judge_model,
                                         temperature, max_tokens, max_retries)
                for cid in claim_ids:
                    votes[cid].append(one.get(cid, 0.0))
            verdicts = {cid: combine_votes(vs) for cid, vs in votes.items()}
            # persist final verdicts + ALL raw votes for audit
            (vdir / f"{qid}.json").write_text(json.dumps(
                {"final": verdicts, "votes": votes, "n_votes": args.n_votes,
                 "backend": backend, "canonical": canonical}, indent=2))
        except Exception as e:  # noqa: BLE001  retries exhausted -> failed, never silent 0
            per_question[qid] = {"claim_score": None, "secondary": sec,
                                 "verdicts": {}, "votes": votes,
                                 "failed": True, "error": str(e)[:300]}
            failed.append(qid)
            print(f"  {qid[:22]:22} FAILED (judge error: {str(e)[:80]})", flush=True)
            continue

        cs = round(aggregate_claims(r, verdicts), 4)
        per_question[qid] = {"claim_score": cs, "secondary": sec,
                             "verdicts": verdicts, "votes": votes, "failed": False}
        claim_scores.append(cs)
        sstr = "-" if sec["score"] is None else f"{sec['score']:.2f}"
        print(f"  {qid[:22]:22} claim={cs:.4f}  secondary={sstr}", flush=True)

    n = len(claim_scores)
    # n == 0 (every question failed) must NEVER masquerade as a real score:
    # mean/ci are null and the leaderboard append below is refused outright.
    mean = round(sum(claim_scores) / n, 4) if n else None
    ci = bootstrap_ci95(claim_scores, resamples=jcfg["bootstrap_resamples"],
                        seed=seed) if n else None
    all_failed = len(failed) == len(gold_rows)

    # secondary mean over non-None scored questions
    sec_vals = [pq["secondary"]["score"] for pq in per_question.values()
                if pq["secondary"]["score"] is not None]
    sec_mean = round(sum(sec_vals) / len(sec_vals), 4) if sec_vals else None

    provenance = build_provenance(args, backend, canonical, tcfg, jcfg, seed,
                                  integrity_status)

    results = {
        "task": args.task, "tag": args.tag, "split": args.split, "budget": args.budget,
        "judge_model": args.judge_model, "backend": backend, "canonical": canonical,
        "grade_mode": jcfg["grade_mode"], "secondary_metric": tcfg["secondary"],
        "mean": mean, "n": n, "bootstrap_ci95": ci,
        "secondary_mean": sec_mean,
        "failed": failed, "n_failed": len(failed), "all_failed": all_failed,
        "provenance": provenance,
        "per_question": per_question,
    }
    if args.limit:
        results["smoke"] = True
        results["limit"] = args.limit
    out_path = bdir / results_name
    out_path.write_text(json.dumps(results, indent=2))
    if args.limit:
        print(f"[judge] --limit {args.limit}: smoke artifacts only "
              f"({out_path.name}, {vdir.name}/); "
              f"results_{args.split}.json / verdicts_{args.split}/ left untouched")

    # leaderboard row (skipped by --no-record for smoke tests)
    if args.no_record:
        print("[judge] --no-record: leaderboard append SKIPPED")
    elif n == 0:
        # An all-failed run has NO score — a 0.0 here would be a lie. Refuse.
        print(f"[judge] REFUSING leaderboard append: 0 questions scored "
              f"({len(failed)}/{len(gold_rows)} failed) — an all-failed run has "
              f"no score to record (see {out_path})")
    else:
        lb = ROOT / g["leaderboard"]
        row = {"task": args.task, "tag": args.tag, "split": args.split,
               "score": mean, "ci": ci, "n": n, "failed": all_failed,
               "n_failed": len(failed), "secondary_mean": sec_mean,
               "judge_model": args.judge_model,
               "backend": backend, "canonical": canonical,
               "integrity": integrity_status,
               "provenance": provenance}
        with lb.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[judge] appended leaderboard row -> {lb}")

    score_str = f"{mean:.4f}" if mean is not None else "n/a (all failed)"
    print(f"\n[judge] {args.tag}/{args.split}: score={score_str}  ci95={ci}  "
          f"n={n}  failed={len(failed)}  secondary={sec_mean}  "
          f"backend={backend}  canonical={canonical}  integrity={integrity_status}")
    print(f"[judge] wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
