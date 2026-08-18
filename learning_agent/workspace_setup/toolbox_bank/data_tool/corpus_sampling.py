"""Shared corpus-access helpers for the data-toolbox generators + eval_tool.

Two sampling styles, both stdlib-only and behaviour-identical to the copies they replace:

  * WHOLE-DOCUMENT  (paraphrase / reasoning / implications / annotation):
      list_docs() -> read_doc() -> doc_context() — one full file per row.
  * EVIDENCE-SPAN   (grounded_qa / gen_eval):
      iter_spans() yields {path,start_line,end_line,text} windows (grep-centred or
      random), wrapped for the prompt by span_context().

Extracted here so the samplers live in ONE place; the generators import from this
module instead of re-defining them. Paths are always relative to the corpus root.
"""
from __future__ import annotations

import random
import re
from pathlib import Path


# --------------------------------------------------------------------------- #
# whole-document access — one full document per file
# --------------------------------------------------------------------------- #
def list_docs(corpus: Path, glob: str) -> list[Path]:
    return sorted(p for p in corpus.glob(glob) if p.is_file())


def read_doc(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def doc_context(rel_path: str, text: str, description: str) -> str:
    """Wrap ONE full document as the generator's in-context source (system message)."""
    return f"{description}\n\nDocument: {rel_path}\n\n<document>\n{text}\n</document>"


# `iter_spans` samples over the same "files under corpus matching glob" set; the doc
# lister is identical, so expose it under the span-oriented name too (no behaviour change).
list_files = list_docs


# --------------------------------------------------------------------------- #
# evidence-span access (same spirit as harness/eval.py's grep/glob/read_file)
# --------------------------------------------------------------------------- #
def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").split("\n")


def span_from(lines: list[str], start: int, span_lines: int) -> tuple[int, int, str]:
    """Return (start_line, end_line, text) for a 1-indexed line window."""
    s = max(1, start)
    e = min(len(lines), s + span_lines - 1)
    text = "\n".join(f"{i:>5}: {lines[i - 1]}" for i in range(s, e + 1))
    return s, e, text


def iter_spans(corpus: Path, glob: str, grep: str | None, n: int,
               span_lines: int, rng: random.Random):
    """Yield up to `n` evidence spans as dicts {path, start_line, end_line, text}.

    With --grep: windows centred on regex matches. Without: random line windows
    from random files. `path` is relative to the corpus root."""
    files = list_files(corpus, glob)
    if not files:
        raise SystemExit(f"no files under {corpus} matching {glob!r}")

    if grep:
        try:
            rx = re.compile(grep)
        except re.error as e:
            raise SystemExit(f"bad --grep regex: {e}")
        hits = []
        for fp in files:
            lines = read_lines(fp)
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append((fp, lines, i))
        rng.shuffle(hits)
        for fp, lines, i in hits[:n]:
            s, e, text = span_from(lines, i - span_lines // 3, span_lines)
            yield {"path": str(fp.relative_to(corpus)),
                   "start_line": s, "end_line": e, "text": text}
        return

    # random-window sampling
    produced = 0
    guard = 0
    while produced < n and guard < n * 20:
        guard += 1
        fp = rng.choice(files)
        lines = read_lines(fp)
        if len(lines) < 3:
            continue
        start = rng.randint(1, max(1, len(lines) - span_lines // 2))
        s, e, text = span_from(lines, start, span_lines)
        if not text.strip():
            continue
        yield {"path": str(fp.relative_to(corpus)),
               "start_line": s, "end_line": e, "text": text}
        produced += 1


def span_context(span: dict) -> str:
    return (f"Source file: {span['path']} (lines {span['start_line']}-{span['end_line']})\n"
            f"--- EXCERPT ---\n{span['text']}\n--- END EXCERPT ---")
