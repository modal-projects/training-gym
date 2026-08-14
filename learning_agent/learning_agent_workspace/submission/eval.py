#!/usr/bin/env python3
"""Submission QA entrypoint — answer a questions file with the submitted system.

Contract (fixed):
    python submission/eval.py --input questions.json --output answers.json
  input : JSON array of {"id": ..., "question": ...} (extra keys ignored)
  output: JSON object {id: answer} for every input id

This is now a THIN CLIENT over the two real contract surfaces:
  submission/serve.py  — the task model behind an OpenAI-compatible endpoint
  submission/agent.py  — build() -> the policy (answer / act / tool_turn)
Rewrite your harness THERE (agent.py answer/answer_batch), not here: post-eval
drives agent.build() directly for agentic tasks, and through this CLI for QA —
either way it scores the same object you developed against.

Baseline behavior (before you modify agent.py): serve WEIGHTS with vLLM (or use
an already-serving --base-url), answer each question with a ReAct search loop
over the task corpus (grep/glob/read_file, 15 tool turns; closed-book when the
track ships no corpus), write the answers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SUB_DIR = str(Path(__file__).resolve().parent)
if _SUB_DIR not in sys.path:
    sys.path.insert(0, _SUB_DIR)
from agent import build           # noqa: E402  (submission/agent.py)
from serve import serve_vllm      # noqa: E402,F401  (back-compat re-export)


def load_questions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise SystemExit(f"--input must be a JSON array of {{id, question}}: {path}")
    out = []
    for i, r in enumerate(rows):
        if "question" not in r:
            raise SystemExit(f"--input row {i} has no 'question' key")
        out.append({"id": r.get("id", f"q{i:04d}"), "question": r["question"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer a questions JSON with the submitted system.")
    ap.add_argument("--input", required=True, help="JSON array of {id, question}")
    ap.add_argument("--output", required=True, help="write {id: answer} JSON here")
    ap.add_argument("--weights", default="",
                    help="checkpoint to serve locally with vLLM (default: serve.py WEIGHTS)")
    ap.add_argument("--base-url", default="",
                    help="already-serving OpenAI-compatible endpoint (skips serving)")
    ap.add_argument("--model", default="",
                    help="model name for --base-url endpoints (default: --weights)")
    ap.add_argument("--backend", default="auto", choices=["auto", "openai", "cli-claude", "mock"],
                    help="mock = deterministic offline stub (contract tests only)")
    args = ap.parse_args()

    questions = load_questions(Path(args.input))
    agent = build(weights=args.weights, base_url=args.base_url, model=args.model,
                  backend="mock" if args.backend == "mock" else "")

    answers = agent.answer_batch(questions)
    missing = [q["id"] for q in questions if str(q["id"]) not in answers]
    if missing:
        raise SystemExit(f"contract violation: no answer for ids {missing[:5]}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(answers, indent=1))
    print(f"[submission] answered {len(answers)} questions -> {out}")


if __name__ == "__main__":
    main()
