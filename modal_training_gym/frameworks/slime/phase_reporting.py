from __future__ import annotations

import base64
import io
import json
import importlib
import os
import threading
import time
from queue import Queue
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from modal_training_gym.common.status import SlimeStatus

PHASE_REPORT_URL_ENV = "SLIME_PHASE_REPORT_URL"
PHASE_REPORT_TOKEN_ENV = "SLIME_PHASE_REPORT_TOKEN"
# Import path of the run model's response parser (a (str) -> ParsedResponse
# callable). The launcher exports it; the recorder resolves and applies it.
RESPONSE_PARSER_PATH_ENV = "TRAINING_GYM_RESPONSE_PARSER_PATH"
# Per-sample execution-trace capture (off by default — traces inflate rollout
# payloads). When on, only the first TRACE_SAMPLE_LIMIT_ENV samples of each
# rollout carry a trace, and only timing/scalar attributes are kept.
CAPTURE_TRACE_ENV = "TRAINING_GYM_CAPTURE_TRACE"
TRACE_SAMPLE_LIMIT_ENV = "TRAINING_GYM_TRACE_SAMPLE_LIMIT"
_TRACE_SAMPLE_LIMIT_DEFAULT = 16
# Backstops so a pathological trace can't blow up the payload we already keep
# small by sampling + dropping payload-bearing attributes.
_TRACE_MAX_SPANS = 256
_TRACE_ATTR_STR_MAX = 200
# Per-sample input-image capture (image-modality runs). Screenshots are large, so
# only the first IMAGE_SAMPLE_LIMIT_ENV samples of each rollout carry one, and each
# is thumbnailed + size-capped before it goes on the payload.
IMAGE_SAMPLE_LIMIT_ENV = "TRAINING_GYM_IMAGE_SAMPLE_LIMIT"
_IMAGE_SAMPLE_LIMIT_DEFAULT = 16
_IMAGE_MAX_DIM = 512
_IMAGE_MAX_BYTES = 256 * 1024
# Per-sample multi-turn trajectory capture. The full ``trajectory_messages`` blob
# (agent transcripts, many turns) is large, so only the first
# TRAJECTORY_SAMPLE_LIMIT_ENV samples of each rollout carry it and each message's
# content is length-capped — enough for the dashboard's ConversationView to render
# the conversation without bloating the rollout payload. Without it, multi-turn
# rollouts (e.g. toolathlon) collapse to a single flat block on the dashboard.
TRAJECTORY_SAMPLE_LIMIT_ENV = "TRAINING_GYM_TRAJECTORY_SAMPLE_LIMIT"
_TRAJECTORY_SAMPLE_LIMIT_DEFAULT = 16
_TRAJECTORY_MSG_CHARS_MAX = 8000
_TRAJECTORY_MAX_MESSAGES = 128
CUSTOM_ROLLOUT_LOG_FUNCTION_PATH_KEY = "training_gym_custom_rollout_log_function_path"
CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH_KEY = (
    "training_gym_custom_eval_rollout_log_function_path"
)
CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY = (
    "training_gym_custom_megatron_before_log_prob_hook_path"
)
CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH_KEY = (
    "training_gym_custom_megatron_before_train_step_hook_path"
)


# Internal queue entry: each item is {"_url": str, "_timeout": float, **payload}.
# Status reports use the framework-status URL with a short 1s timeout;
# rollout-data reports derive a /api/training-rollouts URL from the same base
# with a longer timeout because payloads can be 100KB+.
_REPORT_QUEUE: Queue[dict[str, Any] | None] = Queue(maxsize=512)
_REPORTER_STARTED = False
_REPORTER_LOCK = threading.Lock()
_PHASE_PATH = "/api/framework-status"
_ROLLOUT_PATH = "/api/training-rollouts"
_ADVANTAGE_PATH = "/api/advantage-distributions"
_PHASE_TIMEOUT_SECONDS = 1.0
_STEP_EVENT_TIMEOUT_SECONDS = 5.0
_ROLLOUT_TIMEOUT_SECONDS = 10.0


def _arg_value(args: Any, key: str) -> Any:
    value = getattr(args, key, None)
    if value not in (None, ""):
        return value

    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(key)
            if value not in (None, ""):
                return value
    return None


def _run_context(args: Any) -> dict[str, Any]:
    return {
        "training_run_id": _arg_value(args, "training_run_id")
        or _arg_value(args, "training_gym_training_run_id")
        or os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", "")
        or "",
        "app_name": _arg_value(args, "app_name")
        or _arg_value(args, "training_gym_app_name")
        or os.environ.get("TRAINING_GYM_APP_NAME", "")
        or "",
        "modal_app_id": os.environ.get("MODAL_APP_ID", ""),
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _total_steps(args: Any) -> int | None:
    for key in ("num_rollout", "training_gym_total_steps"):
        total = _positive_int(_arg_value(args, key))
        if total is not None:
            return total
    return _positive_int(os.environ.get("TRAINING_GYM_TOTAL_STEPS"))


def _step_progress(args: Any, rollout_id: int | None = None) -> dict[str, Any]:
    total = _total_steps(args)
    if rollout_id is None:
        current = 0
    else:
        current = max(0, int(rollout_id) + 1)
    if total is not None:
        current = min(current, total)
    return {
        "progress_current": current,
        "progress_total": total,
        "progress_unit": "step",
    }


def _phase_url() -> str:
    return (
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "")
        or os.environ.get(PHASE_REPORT_URL_ENV, "")
    ).strip()


def _derive_url(path: str) -> str:
    base = _phase_url()
    if not base:
        return ""
    if base.endswith(_PHASE_PATH):
        return base[: -len(_PHASE_PATH)] + path
    return base.rstrip("/") + path


def _rollout_url() -> str:
    return _derive_url(_ROLLOUT_PATH)


def _advantage_url() -> str:
    return _derive_url(_ADVANTAGE_PATH)


def _ensure_worker() -> None:
    global _REPORTER_STARTED
    if _REPORTER_STARTED:
        return
    with _REPORTER_LOCK:
        if _REPORTER_STARTED:
            return
        thread = threading.Thread(
            target=_worker,
            name="slime-phase-reporter",
            daemon=True,
        )
        thread.start()
        _REPORTER_STARTED = True


def _enqueue(payload: dict[str, Any]) -> None:
    """Enqueue a framework-status payload (small, 1s timeout)."""
    url = _phase_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _PHASE_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _enqueue_rollout(payload: dict[str, Any]) -> None:
    """Enqueue a rollout-data payload (large, longer timeout)."""
    url = _rollout_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _post_framework_status(payload: dict[str, Any], timeout: float) -> None:
    url = _phase_url()
    if not url:
        return
    _post({"_url": url, "_timeout": timeout, **payload})


def _enqueue_advantage(payload: dict[str, Any]) -> None:
    """Enqueue an advantage-distribution payload (longer timeout like rollouts)."""
    url = _advantage_url()
    if not url:
        return
    _ensure_worker()
    item = {"_url": url, "_timeout": _ROLLOUT_TIMEOUT_SECONDS, **payload}
    try:
        _REPORT_QUEUE.put_nowait(item)
    except Exception:
        pass


def _worker() -> None:
    while True:
        try:
            payload = _REPORT_QUEUE.get()
        except Exception:
            continue
        if payload is None:
            return
        try:
            _post(payload)
        finally:
            _REPORT_QUEUE.task_done()


def _post(item: dict[str, Any]) -> None:
    url = item.pop("_url", "")
    timeout = float(
        item.pop("_timeout", _PHASE_TIMEOUT_SECONDS) or _PHASE_TIMEOUT_SECONDS
    )
    if not url:
        return

    body = json.dumps(item, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = (
        os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", "")
        or os.environ.get(PHASE_REPORT_TOKEN_ENV, "")
    ).strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
    except (OSError, URLError):
        return


def _resolve_hook(path: str | None) -> Any:
    if not path:
        return None
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        return None
    module = importlib.import_module(module_name)
    return getattr(module, attr, None)


def _hook_path_from_args(args: Any, path_key: str) -> str | None:
    direct = getattr(args, path_key, None)
    if isinstance(direct, str) and direct.strip():
        return direct

    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(path_key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def report_phase(
    status: SlimeStatus,
    args: Any = None,
    **extra: Any,
) -> None:
    _enqueue({**_run_context(args), "phase": status.value, **extra})


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


_RESPONSE_PARSER: Any = None
_RESPONSE_PARSER_LOADED = False


def _response_parser() -> Any:
    """Resolve the run model's response parser — a ``(str) -> ParsedResponse``
    callable — from the import path the launcher exports in
    ``RESPONSE_PARSER_PATH_ENV``. Cached per process; ``None`` if unset."""
    global _RESPONSE_PARSER, _RESPONSE_PARSER_LOADED
    if not _RESPONSE_PARSER_LOADED:
        _RESPONSE_PARSER_LOADED = True
        _RESPONSE_PARSER = _resolve_hook(
            os.environ.get(RESPONSE_PARSER_PATH_ENV, "").strip()
        )
    return _RESPONSE_PARSER


def _parsed_response_dict(text: str, parser: Any) -> dict[str, Any] | None:
    """Run ``text`` through ``parser`` and return a JSON-able dict, or ``None``
    when there's no text or no parser."""
    if parser is None or not text:
        return None
    try:
        parsed = parser(text)
    except Exception:
        return None
    return {
        "content": getattr(parsed, "content", "") or "",
        "thinking": getattr(parsed, "thinking", None),
        "tool_calls": [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in (getattr(parsed, "tool_calls", None) or [])
        ],
    }


def _trace_enabled() -> bool:
    return os.environ.get(CAPTURE_TRACE_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _trace_sample_limit() -> int:
    try:
        n = int(os.environ.get(TRACE_SAMPLE_LIMIT_ENV, "").strip())
    except (TypeError, ValueError):
        return _TRACE_SAMPLE_LIMIT_DEFAULT
    return max(0, n)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_scalar(value: Any) -> Any:
    """Keep only small scalar attributes (timings/counts/flags). Drop lists,
    dicts, and long strings so trace payloads can't carry response/tool data
    that already lives on the Sample."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _TRACE_ATTR_STR_MAX else None
    return None


def _trace_attrs(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            scalar = _trace_scalar(value)
            if scalar is not None:
                out[str(key)] = scalar
    return out


def _normalize_span(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    get = raw.get
    name = get("name") or get("event") or get("span") or ""
    start = _coerce_float(get("start", get("start_time", get("ts", get("timestamp")))))
    end = _coerce_float(get("end", get("end_time")))
    if start is None:
        # An instant event may only carry an end/ts; fall back to it.
        start = end
    if start is None:
        return None
    parent = get("parent") or get("parent_name")
    return {
        "name": str(name),
        "start": start,
        "end": end if (end is not None and end != start) else None,
        "attributes": _trace_attrs(get("attributes") or get("attrs") or {}),
        "parent": str(parent) if parent else None,
    }


def _normalize_trace(raw: Any) -> list[dict[str, Any]] | None:
    """Coerce slime's per-sample trace (shape varies by version) into a list of
    ``{name, start, end, attributes, parent}`` spans rebased to start at 0.
    Returns ``None`` when there's nothing usable."""
    spans: list[Any] | None = None
    if isinstance(raw, list):
        spans = raw
    elif isinstance(raw, dict):
        for key in ("spans", "events", "timeline"):
            value = raw.get(key)
            if isinstance(value, list):
                spans = (spans or []) + value
        if spans is None:
            # A dict keyed by span name -> attrs/timing.
            spans = [{"name": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    if not spans:
        return None
    out: list[dict[str, Any]] = []
    for entry in spans[:_TRACE_MAX_SPANS]:
        norm = _normalize_span(entry)
        if norm is not None:
            out.append(norm)
    if not out:
        return None
    base = min(s["start"] for s in out)
    for s in out:
        s["start"] = round(s["start"] - base, 6)
        if s["end"] is not None:
            s["end"] = round(s["end"] - base, 6)
    return out


def _extract_trace(sample: Any) -> Any:
    """Pull slime's trace carrier off a sample — it lives either as a ``trace``
    attribute/key or nested under ``metadata['trace']``, depending on version."""
    if isinstance(sample, dict):
        raw = sample.get("trace")
        if raw is None and isinstance(sample.get("metadata"), dict):
            raw = sample["metadata"].get("trace")
        return raw
    raw = getattr(sample, "trace", None)
    if raw is None:
        meta = getattr(sample, "metadata", None)
        if isinstance(meta, dict):
            raw = meta.get("trace")
    return raw


def _extract_audio_from_prompt(prompt: Any) -> str | None:
    """Pull a browser-playable audio data-URI out of a conversation-list prompt.

    Multimodal datasets keep audio as ``{"type": "audio", "audio": "<data-uri>"}``
    inside message content lists.  Returns ``None`` for plain-text prompts.
    """
    if not isinstance(prompt, list):
        return None
    for msg in prompt:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and (
                item.get("type") == "audio" or "audio" in item
            ):
                ref = item.get("audio") or item.get("audio_url")
                if isinstance(ref, str) and ref:
                    return ref
    return None


def _image_sample_limit() -> int:
    try:
        n = int(os.environ.get(IMAGE_SAMPLE_LIMIT_ENV, "").strip())
    except (TypeError, ValueError):
        return _IMAGE_SAMPLE_LIMIT_DEFAULT
    return max(0, n)


def _trajectory_sample_limit() -> int:
    try:
        n = int(os.environ.get(TRAJECTORY_SAMPLE_LIMIT_ENV, "").strip())
    except (TypeError, ValueError):
        return _TRAJECTORY_SAMPLE_LIMIT_DEFAULT
    return max(0, n)


def _compact_trajectory_messages(raw: Any) -> list[dict[str, Any]] | None:
    """Coerce a Sample's ``trajectory_messages`` into a compact, size-bounded
    list of ``{role, content}`` turns the dashboard can render. Caps the number
    of messages and truncates each long content (head+tail with an elision
    notice) so a single verbose episode can't blow up the rollout payload."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    for msg in raw[:_TRAJECTORY_MAX_MESSAGES]:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            if content is None:
                content = ""
            else:
                try:
                    content = json.dumps(content)
                except TypeError:
                    content = str(content)
        if len(content) > _TRAJECTORY_MSG_CHARS_MAX:
            edge = _TRAJECTORY_MSG_CHARS_MAX // 2
            elided = len(content) - 2 * edge
            content = (
                f"{content[:edge]}\n\n… {elided} chars elided …\n\n{content[-edge:]}"
            )
        entry: dict[str, Any] = {
            "role": str(msg.get("role") or "unknown"),
            "content": content,
        }
        if msg.get("exit_status") is not None:
            entry["exit_status"] = msg["exit_status"]
        out.append(entry)
    return out or None


def _image_to_data_uri(value: Any) -> str | None:
    """Coerce an image reference into a browser-renderable, thumbnailed data-URI.

    Handles PIL Images (how slime's ``process_vision_info`` stores them on the
    Sample), raw bytes, and ``data:``/``http(s)`` strings. Downscales to a
    thumbnail and skips anything that's still too big, so a full-res screenshot
    can't blow up the rollout payload. Remote URLs pass through untouched (the
    browser fetches them — no payload cost). Returns ``None`` on any failure.
    """
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    try:
        from PIL import Image

        img: Any = None
        if isinstance(value, str):
            if value.startswith("data:"):
                _, _, b64 = value.partition(",")
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
            else:
                img = Image.open(value)  # filesystem path
        elif isinstance(value, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(value)))
        elif hasattr(value, "save"):  # already a PIL Image
            img = value
        if img is None:
            return None

        img = img.convert("RGB")
        img.thumbnail((_IMAGE_MAX_DIM, _IMAGE_MAX_DIM))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        mime = "image/png"
        if len(data) > _IMAGE_MAX_BYTES:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            data = buf.getvalue()
            mime = "image/jpeg"
        if len(data) > _IMAGE_MAX_BYTES:
            return None
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        # PIL missing or decode failed: pass through a small inline data-URI as-is.
        if (
            isinstance(value, str)
            and value.startswith("data:")
            and len(value) <= _IMAGE_MAX_BYTES * 2
        ):
            return value
        return None


def _extract_image_from_sample(sample: Any) -> str | None:
    """Pull a browser-renderable input image off a slime Sample.

    For image-modality runs slime's ``process_vision_info`` lifts the screenshot
    into ``sample.multimodal_inputs['images']`` (even when ``apply_chat_template``
    collapses the prompt to a string). Falls back to image items in a
    conversation-list prompt. Returns ``None`` for non-image samples.
    """
    if isinstance(sample, dict):
        get = sample.get
    else:

        def get(key: str, default: Any = None) -> Any:
            return getattr(sample, key, default)

    candidates: list[Any] = []
    mm = get("multimodal_inputs", None)
    if isinstance(mm, dict):
        imgs = mm.get("images") or mm.get("image")
        if imgs is not None:
            candidates.extend(imgs if isinstance(imgs, list) else [imgs])

    prompt = get("prompt", None)
    if isinstance(prompt, list):
        for msg in prompt:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("type") == "image"
                    or "image" in item
                    or "image_url" in item
                ):
                    ref = item.get("image") or item.get("image_url")
                    if isinstance(ref, dict):
                        ref = ref.get("url")
                    if ref is not None:
                        candidates.append(ref)

    for candidate in candidates:
        uri = _image_to_data_uri(candidate)
        if uri:
            return uri
    return None


def _sample_to_dict(
    sample: Any,
    parser: Any = None,
    *,
    include_trace: bool = False,
    include_image: bool = False,
    include_trajectory: bool = False,
) -> dict[str, Any]:
    """Best-effort extraction of (prompt, response, reward, metadata) from a
    slime Sample-like object. Duck-typed so we don't import slime here."""
    if isinstance(sample, dict):
        attrs = sample
        get = sample.get
    else:
        attrs = None

        def get(key: str, default: Any = None) -> Any:
            return getattr(sample, key, default)

    prompt = get("prompt") if attrs is not None else get("prompt", "")
    response = get("response") if attrs is not None else get("response", "")
    reward = get("reward") if attrs is not None else get("reward", 0.0)

    metadata: dict[str, Any] = {}
    for key in ("response_length", "prompt_length", "rollout_id", "rollout_idx"):
        value = get(key) if attrs is not None else get(key, None)
        if value is not None:
            metadata[key] = value

    # Pull display-relevant fields from the sample's own metadata dict so the
    # dashboard can render exit status, eval checks, etc. without needing the
    # (potentially huge) full trajectory_messages blob.
    sample_meta = get("metadata") if attrs is not None else get("metadata", None)
    if isinstance(sample_meta, dict):
        for key in (
            "exit_status",
            "eval_detail",
            "training_response_source",
            "training_assistant_turns",
        ):
            if key in sample_meta and sample_meta[key] is not None:
                metadata[key] = sample_meta[key]
        # Multi-turn trajectory (capped to the first N samples): lets the
        # dashboard's ConversationView render the full agent conversation.
        # Without it, multi-turn rollouts collapse to a single flat block.
        if include_trajectory:
            trajectory = _compact_trajectory_messages(
                sample_meta.get("trajectory_messages")
            )
            if trajectory:
                metadata["trajectory_messages"] = trajectory
        # Store eval_report but only the compact summary/checks, not huge blobs
        eval_report = sample_meta.get("eval_report")
        if isinstance(eval_report, dict):
            compact: dict[str, Any] = {}
            if "check_summary" in eval_report:
                compact["check_summary"] = eval_report["check_summary"]
            if "checks" in eval_report and isinstance(eval_report["checks"], dict):
                compact["checks"] = {
                    name: {
                        "passed": bool(detail.get("passed")),
                        "status": str(detail.get("status", "")),
                        "score": detail.get("score"),
                        "errors": (detail.get("errors") or [])[:3],
                    }
                    for name, detail in eval_report["checks"].items()
                    if isinstance(detail, dict)
                }
            if compact:
                metadata["eval_report"] = compact

    if audio_uri := _extract_audio_from_prompt(prompt):
        metadata["_metadata_type"] = "audio"
        metadata["audio"] = audio_uri
    elif include_image and (image_uri := _extract_image_from_sample(sample)):
        metadata["_metadata_type"] = "image"
        metadata["image"] = image_uri

    response_text = _coerce_text(response)
    out: dict[str, Any] = {
        "score": _coerce_score(reward),
        "prompt": _coerce_text(prompt),
        "response": response_text,
        "metadata": metadata,
    }
    # Store raw + parsed (mirrors eval's EvalRowResult) so the dashboard can show
    # cleaned content without re-parsing. Parsing happens here, in the recorder.
    parsed = _parsed_response_dict(response_text, parser)
    if parsed is not None:
        out["parsed_response"] = parsed
    if include_trace:
        trace = _normalize_trace(_extract_trace(sample))
        if trace:
            out["trace"] = trace
    return out


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        return {str(k): v for k, v in metrics.items()}
    return {}


def report_rollout_samples(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> None:
    """Post one TrainingRolloutResult-shaped payload to the dashboard."""
    if samples is None:
        return
    parser = _response_parser()
    # Trace/image only the first N samples (traces also gated by an enable flag) so
    # the payload stays small — the caps keep volume growth well under 1%.
    trace_limit = _trace_sample_limit() if _trace_enabled() else 0
    image_limit = _image_sample_limit()
    trajectory_limit = _trajectory_sample_limit()
    try:
        sample_dicts = [
            _sample_to_dict(
                s,
                parser,
                include_trace=(i < trace_limit),
                include_image=(i < image_limit),
                include_trajectory=(i < trajectory_limit),
            )
            for i, s in enumerate(samples)
        ]
    except TypeError:
        return
    payload = {
        **_run_context(args),
        "rollout_id": int(rollout_id),
        "created_at": int(time.time()),
        "samples": sample_dicts,
        "metrics": _metrics_to_dict(rollout_extra_metrics),
    }
    if rollout_time is not None:
        try:
            payload["rollout_time"] = float(rollout_time)
        except (TypeError, ValueError):
            pass
    _enqueue_rollout(payload)


def _advantage_samples_payload(
    sample_sums: list[float],
    sample_counts: list[float],
    sample_indices: list[int],
    raw_rewards: list[Any],
    n_samples_per_prompt: int,
) -> list[dict[str, Any]]:
    """Build the per-sample advantage rows from masked ``(sum, count)`` pairs.

    Pure (no torch / mpu) so the group-index and divide-by-count logic is
    unit-testable. ``advantage = sum / count`` is the mask-weighted mean over
    the sample's response tokens; ``group_index`` is the GRPO prompt group the
    sample belongs to (``sample_index // n_samples_per_prompt``).
    """
    n_per = max(1, int(n_samples_per_prompt or 1))
    out: list[dict[str, Any]] = []
    for i in range(len(sample_sums)):
        count = sample_counts[i] if i < len(sample_counts) else 0.0
        advantage = (sample_sums[i] / count) if count else 0.0
        if i < len(sample_indices) and sample_indices[i] is not None:
            idx = int(sample_indices[i])
        else:
            idx = i
        raw = raw_rewards[i] if i < len(raw_rewards) else None
        out.append(
            {
                "sample_index": idx,
                "group_index": idx // n_per,
                "advantage": float(advantage),
                "raw_reward": _coerce_float(raw) if raw is not None else None,
            }
        )
    return out


def report_advantage_distribution(
    rollout_id: int,
    args: Any,
    rollout_data: Any,
) -> None:
    """Emit per-sample advantages (tagged with their GRPO group) for one step.

    Injected into slime's ``log_rollout_data`` so it fires right after
    ``compute_advantages_and_returns``. slime itself only logs the *mean*
    advantage per step; this captures the full per-sample distribution.

    Runs on every actor rank but only the TP-rank-0 / last-PP-stage ranks hold
    the reduced advantages, and within those only CP-rank-0 posts (after a CP
    all-reduce makes each sample's mean cover its full response). Each surviving
    rank covers its own data-parallel shard of the step's samples; the dashboard
    merges shards into per-group distributions.
    """
    if not isinstance(rollout_data, dict):
        return
    try:
        import torch
        import torch.distributed as dist
        from megatron.core import mpu
    except Exception:
        return

    try:
        if not (
            mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()
        ):
            return
    except Exception:
        return

    advantages = rollout_data.get("advantages")
    loss_masks = rollout_data.get("loss_masks")
    response_lengths = rollout_data.get("response_lengths")
    total_lengths = rollout_data.get("total_lengths")
    if advantages is None or loss_masks is None:
        return
    if response_lengths is None or total_lengths is None:
        return

    n = len(advantages)
    try:
        device = advantages[0].device
        sums = torch.zeros(n, dtype=torch.float64, device=device)
        counts = torch.zeros(n, dtype=torch.float64, device=device)
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        if cp_size == 1:
            for i in range(n):
                adv = advantages[i].to(torch.float64)
                mask = loss_masks[i].to(torch.float64)
                m = min(adv.numel(), mask.numel())
                sums[i] = (adv[:m] * mask[:m]).sum()
                counts[i] = mask[:m].sum()
        else:
            from slime.backends.megatron_utils.cp_utils import (
                get_logits_and_tokens_offset_with_cp,
            )

            for i in range(n):
                total_len = int(total_lengths[i])
                resp_len = int(response_lengths[i])
                prompt_len = total_len - resp_len
                _, _, _, toff = get_logits_and_tokens_offset_with_cp(
                    total_len, resp_len
                )
                mask = loss_masks[i]
                m0 = mask[toff[0][0] - prompt_len : toff[0][1] - prompt_len]
                m1 = mask[toff[1][0] - prompt_len : toff[1][1] - prompt_len]
                chunked = torch.cat([m0, m1]).to(torch.float64)
                adv = advantages[i].to(torch.float64)
                m = min(adv.numel(), chunked.numel())
                sums[i] = (adv[:m] * chunked[:m]).sum()
                counts[i] = chunked[:m].sum()
            # Every CP rank holds a token-shard of the same samples; reduce so
            # each sample's mean is taken over its full response.
            cp_group = mpu.get_context_parallel_group()
            dist.all_reduce(sums, group=cp_group)
            dist.all_reduce(counts, group=cp_group)
    except Exception:
        return

    if cp_rank != 0:
        return

    raw_rewards = list(rollout_data.get("raw_reward") or [])
    sample_indices = list(rollout_data.get("sample_indices") or range(n))
    n_per = _positive_int(_arg_value(args, "n_samples_per_prompt")) or 1

    samples = _advantage_samples_payload(
        sums.tolist(),
        counts.tolist(),
        sample_indices,
        raw_rewards,
        n_per,
    )
    if not samples:
        return

    try:
        dp_rank = int(mpu.get_data_parallel_rank(with_context_parallel=False))
    except Exception:
        dp_rank = 0

    _enqueue_advantage(
        {
            **_run_context(args),
            "rollout_id": int(rollout_id),
            "created_at": int(time.time()),
            "dp_rank": dp_rank,
            "n_samples_per_prompt": int(n_per),
            "samples": samples,
        }
    )


def _call_hook(path_key: str, args: Any, *hook_args: Any, **hook_kwargs: Any) -> Any:
    hook = _resolve_hook(_hook_path_from_args(args, path_key))
    if hook is None:
        return None
    return hook(*hook_args, **hook_kwargs)


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: Any,
    rollout_time: Any,
) -> bool:
    progress = _step_progress(args, rollout_id)
    report_phase(
        SlimeStatus.ROLLOUT_LOGGING,
        args,
        **progress,
        rollout_id=rollout_id,
        sample_count=len(samples) if hasattr(samples, "__len__") else None,
        metrics=rollout_extra_metrics,
        rollout_time=rollout_time,
    )
    report_rollout_samples(
        rollout_id, args, samples, rollout_extra_metrics, rollout_time
    )
    result = _call_hook(
        CUSTOM_ROLLOUT_LOG_FUNCTION_PATH_KEY,
        args,
        rollout_id,
        args,
        samples,
        rollout_extra_metrics,
        rollout_time,
    )
    if result is None:
        return False
    return bool(result)


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: Any,
    extra_metrics: Any,
) -> bool:
    report_phase(
        SlimeStatus.EVAL_ROLLOUT_LOGGING,
        args,
        **_step_progress(args, rollout_id),
        rollout_id=rollout_id,
        sample_count=len(data) if hasattr(data, "__len__") else None,
        metrics=extra_metrics,
    )
    result = _call_hook(
        CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH_KEY,
        args,
        rollout_id,
        args,
        data,
        extra_metrics,
    )
    if result is None:
        return False
    return bool(result)


def before_log_prob_hook(args: Any, model: Any, store_prefix: str) -> None:
    report_phase(
        SlimeStatus.COMPUTE_LOG_PROBS,
        args,
        store_prefix=store_prefix,
    )
    _call_hook(
        CUSTOM_BEFORE_LOG_PROB_HOOK_PATH_KEY,
        args,
        args,
        model,
        store_prefix,
    )


def before_train_step_hook(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    report_phase(
        SlimeStatus.OPTIMIZER_STEP,
        args,
        **_step_progress(args, rollout_id),
        rollout_id=rollout_id,
        step_id=step_id,
    )
    _call_hook(
        CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH_KEY,
        args,
        args,
        rollout_id,
        step_id,
        model,
        optimizer,
        opt_param_scheduler,
    )


def report_rollout_initializing(args: Any) -> None:
    report_phase(
        SlimeStatus.ROLLOUT_INITIALIZING,
        args,
        **_step_progress(args),
    )


def report_step_start(args: Any, rollout_id: int | None = None) -> None:
    _post_framework_status(
        {
            **_run_context(args),
            "phase": SlimeStatus.ROLLOUT_LOGGING.value,
            **_step_progress(args, rollout_id),
            "rollout_id": rollout_id,
            "step_event": "start",
        },
        _STEP_EVENT_TIMEOUT_SECONDS,
    )


def report_weight_sync(args: Any) -> None:
    report_phase(
        SlimeStatus.WEIGHT_SYNC,
        args,
    )


def report_generate_rollouts(args: Any) -> None:
    report_phase(
        SlimeStatus.ROLLOUT_LOGGING,
        args,
    )


def report_step_complete(args: Any, rollout_id: int | None = None) -> None:
    if rollout_id is None:
        return
    _post_framework_status(
        {
            **_run_context(args),
            "phase": SlimeStatus.WEIGHT_SYNC.value,
            **_step_progress(args, rollout_id),
            "rollout_id": rollout_id,
            "step_event": "finish",
        },
        _STEP_EVENT_TIMEOUT_SECONDS,
    )


__all__ = [
    "before_log_prob_hook",
    "before_train_step_hook",
    "report_advantage_distribution",
    "report_generate_rollouts",
    "report_phase",
    "report_rollout_initializing",
    "report_rollout_samples",
    "report_step_start",
    "report_step_complete",
    "report_weight_sync",
    "log_eval_rollout_data",
    "log_rollout_data",
]
