"""sglang_serve — launch an SGLang OpenAI-compatible server.

Sibling of vllm_serve.py — same shape, different engine. Wraps
`python -m sglang.launch_server`, prints the base_url the generators expect, then execs (or
subprocesses) it. SGLang is what the operator's eval harness uses to serve the task model, so
generating behind it keeps tokenizer/chat-template behaviour identical to eval time. Model
comes from --model or $TASK_MODEL (see ../README.md). The heavy `sglang` import is guarded,
so --help and --print-cmd work with nothing installed.

Note: eval applies the chat template with enable_thinking=False; if the model emits
<think>...</think> blocks and you need eval-matching output, strip them in the consumer.

    python3 toolbox/inference_tool/sglang_serve.py --model "$TASK_MODEL" --port 30000 --tp 2 --max-len 131072
    # -> base_url: http://0.0.0.0:30000/v1  (feed to the data_tool generators via --base-url)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

DEFAULT_MODEL = os.environ.get("TASK_MODEL")  # no model baked in; set $TASK_MODEL


def build_cmd(args: argparse.Namespace) -> list[str]:
    """Assemble the `python -m sglang.launch_server ...` argv from parsed flags."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tp", str(args.tp),
        "--mem-fraction-static", str(args.mem_fraction_static),
    ]
    if args.max_len:
        cmd += ["--context-length", str(args.max_len)]
    if args.served_name:
        cmd += ["--served-model-name", args.served_name]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.extra:
        cmd += args.extra
    return cmd


def base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}/v1"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Launch an SGLang OpenAI-compatible server "
                    "(wraps `python -m sglang.launch_server`).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF id or local path of the task model/base model to serve "
                         "(defaults to $TASK_MODEL)")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tp", "--tensor-parallel", dest="tp", type=int, default=1,
                    help="tensor-parallel size (GPUs); up to 8 on Modal H200 nodes")
    ap.add_argument("--max-len", "--context-length", dest="max_len", type=int,
                    default=131072, help="context length (0 = engine default)")
    ap.add_argument("--mem-fraction-static", type=float, default=0.85,
                    help="static KV-cache memory fraction (eval uses 0.85)")
    ap.add_argument("--served-name", default="",
                    help="override served model name reported at /v1/models")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--no-trust-remote-code", dest="trust_remote_code",
                    action="store_false")
    ap.add_argument("--print-cmd", action="store_true",
                    help="print the command + base_url and exit (no serving)")
    ap.add_argument("--no-exec", action="store_true",
                    help="run via subprocess (blocking) instead of exec-replacing")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args passed through to sglang.launch_server (after --)")
    args = ap.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    if not args.model:
        raise SystemExit(
            "no model specified: pass --model <task-model> or set $TASK_MODEL "
            "(no model is baked in; the task model is pinned per task by the operator)")

    cmd = build_cmd(args)
    url = base_url(args)
    print("command:", " ".join(cmd))
    print("base_url:", url)
    print(f"models endpoint: {url}/models   chat endpoint: {url}/chat/completions")

    if args.print_cmd:
        return

    # Soft, guarded probe — never required to build the command.
    try:
        import sglang  # noqa: F401  (heavy optional import, guarded)
        print(f"sglang version: {getattr(sglang, '__version__', 'unknown')}")
    except Exception:  # noqa: BLE001
        raise SystemExit(
            "`sglang` not importable. Install it (`pip install \"sglang[all]\"`) on the "
            "GPU host, or use --print-cmd to just emit the command.")

    print(f"starting SGLang… serving {args.model} at {url}", flush=True)
    if args.no_exec:
        sys.exit(subprocess.run(cmd, env=os.environ.copy()).returncode)
    os.execvp(cmd[0], cmd)  # replace this process with the launcher


if __name__ == "__main__":
    main()
