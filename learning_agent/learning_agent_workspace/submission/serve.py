#!/usr/bin/env python3
"""Submission serving — put the trained task model behind an OpenAI-compatible endpoint.

This is the FIRST half of the submission contract: every post-eval path (QA
answers, tau2 rollouts, tb-lite terminal runs) talks to the task model through one
OpenAI-compatible endpoint, and this file is how the submission provides it.

    python submission/serve.py --weights /out/models/$LEARNING_AGENT_RUN_ID/<tag>/merged [--port 8000]
        serve and block (Ctrl-C to stop); prints the base_url when ready.

Importable (used by submission/agent.py and the operator harnesses):
    ensure_endpoint(weights, base_url="", port=8000) -> (base_url, model_name)
        returns an already-serving endpoint as-is, else starts vLLM on `weights`.

The serving stack is THE AGENT'S TO MODIFY (vLLM flags, sglang, LoRA adapters,
quantization — your call); the contract that must survive is just "an
OpenAI-compatible /v1 endpoint that serves the submitted task model". For QA
and env tasks scored through act(driver="tools") (e.g. alfworld), THIS file is
how the scored system serves — the operator calls build(weights=...) and your
serving rides into the score. tau2_* is the exception: tau2's native
orchestrator is served by the operator's pinned recipe, so only the weights
carry there.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import time
import urllib.request

# ---- agent-owned configuration ------------------------------------------- #
# The submitted checkpoint (local HF dir or a path on the mounted lab-out
# volume). submission/agent.py falls back to this when no endpoint is passed.
WEIGHTS = ""

# max_len=32768: fits the closed-book baseline; raise it (agent-owned) if your
# harness feeds long context per query.
MAX_MODEL_LEN = 32768


def serve_vllm(weights: str, port: int = 8000, max_len: int = MAX_MODEL_LEN) -> str:
    """Start a local vLLM OpenAI server for `weights`; return its base_url.
    Blocks until /v1/models responds. Requires vllm installed and local weights.

    --enable-auto-tool-choice + --tool-call-parser are LOAD-BEARING for env
    tasks: act(driver="tools") sends OpenAI function schemas, and without a
    parser the model's tool calls come back as prose and never reach the
    environment (the driver detects this and errors instead of scoring 0).
    The parser names match the Qwen3.5 student; change them if you serve a
    different family."""
    cmd = ["vllm", "serve", weights, "--served-model-name", weights,
           "--port", str(port), "--max-model-len", str(max_len),
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
           "--reasoning-parser", "qwen3"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}/v1"
    # Poll readiness every 5s for up to 30 min: a cold vLLM start (weight
    # load + CUDA-graph capture) routinely takes several minutes on the first serve.
    for _ in range(360):
        try:
            urllib.request.urlopen(f"{base}/models", timeout=2)
            return base
        except Exception:  # noqa: BLE001
            if proc.poll() is not None:
                raise SystemExit(f"vllm exited early (rc={proc.returncode}); "
                                 f"check weights path: {weights}")
            time.sleep(5)
    proc.terminate()
    raise SystemExit("vllm did not become ready in time")


def ensure_endpoint(weights: str = "", base_url: str = "", port: int = 8000) -> tuple[str, str]:
    """-> (base_url, model_name). Pass-through when an endpoint is already up;
    otherwise serve `weights` locally. The model name is what /chat/completions
    requests should send (vLLM registers the weights path as the model id)."""
    if base_url:
        return base_url, (weights or "task model")
    w = weights or WEIGHTS
    if not w:
        raise SystemExit("no endpoint and no weights: set WEIGHTS in submission/serve.py, "
                         "or pass --weights / --base-url")
    return serve_vllm(w, port=port), w


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the submitted task model (OpenAI-compatible).")
    ap.add_argument("--weights", default=WEIGHTS, help="checkpoint to serve (default: WEIGHTS)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    base, model = ensure_endpoint(weights=args.weights, port=args.port)
    print(f"[serve] ready: {base}  (model={model}) — Ctrl-C to stop", flush=True)
    signal.pause()  # vLLM runs as our child; keep the foreground process alive


if __name__ == "__main__":
    main()
