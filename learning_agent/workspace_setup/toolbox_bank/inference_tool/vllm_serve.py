"""vllm_serve — launch a vLLM OpenAI-compatible server.

Thin wrapper around the `vllm serve` CLI: assembles the command from a few high-signal
flags, prints the base_url the generators expect, then execs (or subprocesses) vllm. Model
comes from --model or $TASK_MODEL (see ../README.md). The heavy `vllm` import is guarded,
so --help and --print-cmd work with nothing installed; serving needs `vllm` on PATH.

IMAGE PIN: the Qwen3.5 task model needs vLLM >= 0.25. Run this inside
`vllm/vllm-openai:v0.27.1` (gpu_launcher --image; CUDA bundled) — the default
debian_slim/base images cannot serve it (../README.md has the full recipe).

    python3 toolbox/inference_tool/vllm_serve.py --model "$TASK_MODEL" --port 8000 --tp 2 --max-len 131072
    # -> base_url: http://0.0.0.0:8000/v1  (feed to the data_tool generators via --base-url)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_MODEL = os.environ.get("TASK_MODEL")  # no model baked in; set $TASK_MODEL


def build_cmd(args: argparse.Namespace) -> list[str]:
    """Assemble the `vllm serve ...` argv from parsed flags."""
    cmd = [
        "vllm", "serve", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if args.max_len:
        cmd += ["--max-model-len", str(args.max_len)]
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
        description="Launch a vLLM OpenAI-compatible server (wraps `vllm serve`).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF id or local path of the task model/base model to serve "
                         "(defaults to $TASK_MODEL)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tp", "--tensor-parallel", dest="tp", type=int, default=1,
                    help="tensor-parallel size (GPUs); up to 8 on Modal H200 nodes")
    ap.add_argument("--max-len", "--max-model-len", dest="max_len", type=int,
                    default=131072, help="max model/context length (0 = vllm default)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
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
                    help="extra args passed through to `vllm serve` (after --)")
    args = ap.parse_args()
    # argparse REMAINDER keeps a leading '--'; drop it.
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

    if shutil.which("vllm") is None:
        raise SystemExit(
            "`vllm` not found on PATH. Install it (`pip install vllm`) on the GPU host, "
            "or use --print-cmd to just emit the command.")

    # Soft, guarded version probe — never required to build the command.
    try:
        import vllm  # noqa: F401  (heavy optional import, guarded)
        print(f"vllm version: {getattr(vllm, '__version__', 'unknown')}")
    except Exception:  # noqa: BLE001
        pass

    print(f"starting vLLM… serving {args.model} at {url}", flush=True)
    if args.no_exec:
        sys.exit(subprocess.run(cmd, env=os.environ.copy()).returncode)
    os.execvp(cmd[0], cmd)  # replace this process with vllm


if __name__ == "__main__":
    main()
