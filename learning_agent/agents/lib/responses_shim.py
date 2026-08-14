#!/usr/bin/env python3
"""Responses->Chat shim: lets Codex CLI drive any OpenAI-compatible CHAT
endpoint.

Why this exists (2026-07-31): Codex CLI removed the chat-completions wire
(hard error since Feb 2026) and speaks only the Responses API. Our serving
stacks (SGLang, incl. the managed Modal Flash endpoint) serve /v1/responses
but do NOT support custom function tools on it (sgl-project/sglang#13292,
closed stale) — and an agent harness without tools is useless. This shim
listens on localhost, accepts Codex's /v1/responses traffic, translates it to
/v1/chat/completions against the upstream, and streams Responses-API SSE
events back. Self-owned and auditable — no third-party bridge in the
benchmark's agent path.

Scope is deliberately Codex-shaped: message/function_call/function_call_output
input items, flat function tools, instructions, streaming. Reasoning deltas
from the upstream (reasoning_content) are translated into Responses
reasoning items, so codex records the model's chain of thought in its own
trace (item type "reasoning") — nothing is dropped and no sidecar file is
needed. Reasoning items on the INPUT side are still skipped, so past
thinking never re-enters the model's context. Unknown
input item types are skipped. Anything else fails loudly.

Usage:
    python3 agents/lib/responses_shim.py --port 8299 \
        --upstream https://…/v1 [--api-key public-endpoint]
Health: GET /healthz -> {"ok": true}
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

UPSTREAM = ""
API_KEY = ""
TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=60.0)



def _rid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# ---- request translation: Responses -> Chat ---------------------------------

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text"):
            parts.append(c.get("text", ""))
    return "".join(parts)


def to_chat_request(body: dict) -> dict:
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    call_names: dict[str, str] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        t = it.get("type", "message")
        if t == "message":
            role = it.get("role", "user")
            if role == "system":  # top-level system already set; keep order anyway
                messages.append({"role": "system", "content": _content_to_text(it.get("content"))})
            else:
                messages.append({"role": role, "content": _content_to_text(it.get("content"))})
        elif t == "function_call":
            call_id = it.get("call_id") or it.get("id") or _rid("call")
            call_names[call_id] = it.get("name", "")
            arguments = it.get("arguments", "") or "{}"
            try:
                json.loads(arguments)
            except ValueError:
                # A conversation cut mid-stream (container preemption) can
                # persist a tool call with truncated arguments. The upstream
                # json-parses every historical arguments string and 400s the
                # whole request ("unexpected end of data"), bricking every
                # resume of that conversation (2026-08-13, one bad call among
                # 2546 messages). Repair to a note the model can read.
                arguments = json.dumps(
                    {"_truncated": "arguments lost when the container was interrupted"})
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": it.get("name", ""),
                             "arguments": arguments},
            }
            # Parallel tool calls arrive as consecutive function_call items and
            # must share ONE assistant message: strict chat templates (Kimi K3)
            # resolve tool results against the immediately preceding assistant
            # turn's tool_calls and reject split single-call messages with 400.
            last = messages[-1] if messages else None
            if last is not None and last.get("role") == "assistant" and last.get("tool_calls"):
                last["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [call]})
        elif t == "function_call_output":
            out = it.get("output", "")
            if not isinstance(out, str):
                out = _content_to_text(out) or json.dumps(out)
            msg = {"role": "tool",
                   "tool_call_id": it.get("call_id", ""),
                   "content": out}
            if call_names.get(msg["tool_call_id"]):
                msg["name"] = call_names[msg["tool_call_id"]]
            messages.append(msg)
        # reasoning / item_reference / unknown -> skipped on purpose

    chat: dict = {"model": body.get("model", ""), "messages": messages}
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "function":
            fn = tool.get("function") or {k: tool.get(k) for k in
                                          ("name", "description", "parameters", "strict")}
            fn = {k: v for k, v in fn.items() if v is not None}
            tools.append({"type": "function", "function": fn})
    if tools:
        chat["tools"] = tools
        if body.get("tool_choice") in ("auto", "none", "required"):
            chat["tool_choice"] = body["tool_choice"]
        if isinstance(body.get("parallel_tool_calls"), bool):
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("max_output_tokens"):
        chat["max_tokens"] = body["max_output_tokens"]
    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    return chat


# ---- response translation: Chat -> Responses --------------------------------

def _envelope(body: dict, status: str, output: list, usage: dict | None) -> dict:
    u = usage or {}
    return {
        "id": _rid("resp"), "object": "response", "created_at": int(time.time()),
        "model": body.get("model", ""), "status": status, "output": output,
        "error": None, "incomplete_details": None, "instructions": None,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "tool_choice": body.get("tool_choice", "auto"), "tools": body.get("tools") or [],
        "usage": {
            # prompt_tokens INCLUDES the cached prefix; cached_tokens is the
            # subset served from sglang's radix cache (prompt_tokens_details —
            # present when the server reports cache hits, null on cold turns).
            # codex forwards it as cached_input_tokens -> the observatory's
            # "cache read". There is no cache-write concept on this stack.
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": (
                (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)},
            "output_tokens_details": {"reasoning_tokens": u.get("reasoning_tokens", 0)},
        },
    }


def _message_item(text: str) -> dict:
    return {"type": "message", "id": _rid("msg"), "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _reasoning_item(rid: str, text: str) -> dict:
    return {"type": "reasoning", "id": rid,
            "summary": [{"type": "summary_text", "text": text}]}


def _fc_item(call_id: str, name: str, arguments: str) -> dict:
    return {"type": "function_call", "id": _rid("fc"), "call_id": call_id,
            "name": name, "arguments": arguments, "status": "completed"}


class _SSE:
    """Orders and frames Responses-API SSE events."""

    def __init__(self, wfile):
        self.w = wfile
        self.seq = 0

    def emit(self, etype: str, **fields):
        self.seq += 1
        data = {"type": etype, "sequence_number": self.seq}
        data.update(fields)
        payload = f"event: {etype}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.w.write(payload.encode("utf-8"))
        self.w.flush()


def stream_translate(handler, body: dict, chat_req: dict) -> None:
    """Upstream chat SSE -> downstream Responses SSE."""
    chat_req["stream"] = True
    sse = _SSE(handler.wfile)
    resp_env = _envelope(body, "in_progress", [], None)
    sse.emit("response.created", response=resp_env)
    sse.emit("response.in_progress", response=resp_env)

    out_index = -1
    text_buf: list[str] = []
    text_item_id = None
    reasoning_buf: list[str] = []
    reasoning_item_id = None
    reasoning_index = -1
    reasoning_items: list[dict] = []
    # tool calls keyed by upstream index
    calls: dict[int, dict] = {}
    usage = None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    def close_reasoning():
        nonlocal reasoning_item_id
        if reasoning_item_id is None:
            return
        text = "".join(reasoning_buf)
        sse.emit("response.reasoning_summary_text.done", item_id=reasoning_item_id,
                 output_index=reasoning_index, summary_index=0, text=text)
        sse.emit("response.reasoning_summary_part.done", item_id=reasoning_item_id,
                 output_index=reasoning_index, summary_index=0,
                 part={"type": "summary_text", "text": text})
        item = _reasoning_item(reasoning_item_id, text)
        sse.emit("response.output_item.done", output_index=reasoning_index, item=item)
        reasoning_items.append(item)
        reasoning_item_id = None

    def close_text():
        nonlocal text_item_id, out_index
        if text_item_id is None:
            return
        text = "".join(text_buf)
        sse.emit("response.output_text.done", item_id=text_item_id,
                 output_index=out_index, content_index=0, text=text)
        sse.emit("response.content_part.done", item_id=text_item_id,
                 output_index=out_index, content_index=0,
                 part={"type": "output_text", "text": text, "annotations": []})
        done_item = _message_item(text)
        done_item["id"] = text_item_id
        sse.emit("response.output_item.done", output_index=out_index, item=done_item)
        text_item_id = None

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{UPSTREAM}/chat/completions",
                               headers=headers, json=chat_req) as r:
                if r.status_code != 200:
                    detail = r.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"upstream {r.status_code}: {detail}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    r_delta = delta.get("reasoning_content")
                    if r_delta:
                        if reasoning_item_id is None:
                            out_index += 1
                            reasoning_index = out_index
                            reasoning_item_id = _rid("rs")
                            sse.emit("response.output_item.added",
                                     output_index=reasoning_index,
                                     item={"type": "reasoning", "id": reasoning_item_id,
                                           "summary": []})
                            sse.emit("response.reasoning_summary_part.added",
                                     item_id=reasoning_item_id, output_index=reasoning_index,
                                     summary_index=0,
                                     part={"type": "summary_text", "text": ""})
                        reasoning_buf.append(r_delta)
                        sse.emit("response.reasoning_summary_text.delta",
                                 item_id=reasoning_item_id, output_index=reasoning_index,
                                 summary_index=0, delta=r_delta)
                    content = delta.get("content")
                    if content:
                        if text_item_id is None:
                            close_reasoning()
                            out_index += 1
                            text_item_id = _rid("msg")
                            item = {"type": "message", "id": text_item_id,
                                    "role": "assistant", "status": "in_progress",
                                    "content": []}
                            sse.emit("response.output_item.added",
                                     output_index=out_index, item=item)
                            sse.emit("response.content_part.added", item_id=text_item_id,
                                     output_index=out_index, content_index=0,
                                     part={"type": "output_text", "text": "", "annotations": []})
                        text_buf.append(content)
                        sse.emit("response.output_text.delta", item_id=text_item_id,
                                 output_index=out_index, content_index=0, delta=content)

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in calls:
                            close_reasoning()
                            close_text()
                            out_index += 1
                            calls[idx] = {
                                "item_id": _rid("fc"),
                                "call_id": tc.get("id") or _rid("call"),
                                "name": (tc.get("function") or {}).get("name", ""),
                                "args": [], "output_index": out_index,
                            }
                            c = calls[idx]
                            sse.emit("response.output_item.added",
                                     output_index=c["output_index"],
                                     item={"type": "function_call", "id": c["item_id"],
                                           "call_id": c["call_id"], "name": c["name"],
                                           "arguments": "", "status": "in_progress"})
                        c = calls[idx]
                        fn = tc.get("function") or {}
                        if fn.get("name") and not c["name"]:
                            c["name"] = fn["name"]
                        if fn.get("arguments"):
                            c["args"].append(fn["arguments"])
                            sse.emit("response.function_call_arguments.delta",
                                     item_id=c["item_id"], output_index=c["output_index"],
                                     delta=fn["arguments"])

    except Exception as e:  # noqa: BLE001
        # The SSE headers are already on the wire, so an HTTP error status is
        # unsendable — and dying silently leaves codex waiting out its stream
        # idle timeout on an open socket (the 08-12 kimi "idle timeout waiting
        # for SSE" stalls, root cause a kimi-side 400). Emit a terminal event
        # codex can read and log the error where the operator can see it.
        print(f"[shim] upstream error: {e}", flush=True)
        if "unexpected end of data" in str(e):
            # 2026-08-13 truncation forensics: capture the exact request body
            # so the failure can be replayed from another vantage point
            import base64
            import zlib
            payload = json.dumps(chat_req, ensure_ascii=False).encode("utf-8")
            b64 = base64.b64encode(zlib.compress(payload)).decode()
            print(f"[shim] failing request body (zlib+b64, {len(payload)}B raw): {b64}",
                  flush=True)
        env = _envelope(body, "failed", [], None)
        env["error"] = {"code": "server_error", "message": f"shim upstream: {e}"}
        try:
            sse.emit("response.failed", response=env)
        except Exception:  # noqa: BLE001 — client already gone
            pass
        return

    close_reasoning()
    close_text()
    output = []
    if text_buf and not calls:
        pass  # close_text already emitted the done item; envelope filled below
    final_output = list(reasoning_items)
    if text_buf:
        final_output.append(_message_item("".join(text_buf)))
    for idx in sorted(calls):
        c = calls[idx]
        args = "".join(c["args"]) or "{}"
        sse.emit("response.function_call_arguments.done", item_id=c["item_id"],
                 output_index=c["output_index"], arguments=args)
        item = _fc_item(c["call_id"], c["name"], args)
        item["id"] = c["item_id"]
        sse.emit("response.output_item.done", output_index=c["output_index"], item=item)
        final_output.append(item)
    _ = output
    done_env = _envelope(body, "completed", final_output, usage)
    sse.emit("response.completed", response=done_env)


def complete_translate(body: dict, chat_req: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(f"{UPSTREAM}/chat/completions", headers=headers, json=chat_req)
    if r.status_code != 200:
        raise RuntimeError(f"upstream {r.status_code}: {r.text[:500]}")
    data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    output = []
    if msg.get("reasoning_content"):
        output.append(_reasoning_item(_rid("rs"), msg["reasoning_content"]))
    if msg.get("content"):
        output.append(_message_item(msg["content"]))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append(_fc_item(tc.get("id") or _rid("call"),
                               fn.get("name", ""), fn.get("arguments", "") or "{}"))
    return _envelope(body, "completed", output, data.get("usage"))


# ---- HTTP server -------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; codex is chatty
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/healthz", "/health"):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/responses"):
            self.send_error(404, "shim serves /v1/responses only")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            chat_req = to_chat_request(body)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                stream_translate(self, body, chat_req)
                # SSE stream has no length; close the connection to end it
                self.close_connection = True
            else:
                payload = json.dumps(complete_translate(body, chat_req)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 — translate everything to a 502 the client can read
            err = json.dumps({"error": {"message": f"shim: {e}", "type": "shim_error"}}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            except BrokenPipeError:
                pass


def main() -> None:
    global UPSTREAM, API_KEY
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True, help=".../v1 of a chat-completions endpoint")
    ap.add_argument("--api-key", default="", help="bearer for the upstream (optional)")
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    API_KEY = args.api_key
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[shim] /v1/responses on 127.0.0.1:{args.port} -> {UPSTREAM}/chat/completions",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    threading.stack_size(1 << 20)
    main()
