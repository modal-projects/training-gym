#!/usr/bin/env python3
"""completion_qa — the no-harness task agent (closed-book completion).

Pure LLM completion over questions — no search, no tools, no retrieval.

Reads a questions file (JSON list of {"id","question"}), queries an
OpenAI-compatible chat endpoint with the task's system prompt, and writes
answers.json ({id: answer}) — the exact submission/eval.py output contract.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def ask(base_url: str, api_key: str, model: str, sys_prompt: str,
        question: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
    payload = {
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "messages": ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
                    + [{"role": "user", "content": question}],
    }
    req = Request(f"{base_url.rstrip('/')}/chat/completions",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json",
                           **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    last = None
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=600) as r:
                data = json.load(r)
            return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        except Exception as e:  # noqa: BLE001 — retry transport/5xx uniformly
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"endpoint failed after {retries + 1} attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", required=True, help='JSON list of {"id","question"}')
    ap.add_argument("--out", required=True, help="answers.json ({id: answer})")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base")
    ap.add_argument("--model", required=True, help="served model id or /out/models/<tag>/merged")
    ap.add_argument("--api-key", default="", help="bearer for the endpoint (optional)")
    ap.add_argument("--sys-file", default="", help="system prompt file (e.g. tasks/<t>/sys.txt)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    sys_prompt = Path(args.sys_file).read_text() if args.sys_file else ""
    rows = json.loads(Path(args.questions).read_text())
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        answers = dict(zip(
            (r["id"] for r in rows),
            pool.map(lambda r: ask(args.base_url, args.api_key, args.model, sys_prompt,
                                   r["question"], args.temperature, args.max_tokens), rows)))
    Path(args.out).write_text(json.dumps(answers, ensure_ascii=False, indent=1) + "\n")
    print(f"[completion_qa] {len(answers)} answers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
