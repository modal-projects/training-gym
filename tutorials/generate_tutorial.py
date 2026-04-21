#!/usr/bin/env python3
"""Generate .py + .ipynb tutorials from decorator-annotated source files.

Input files live in `tutorials/tutorial_generator/<name>.py` and declare cells
via top-level functions decorated with `@markdown` or `@code`:

  - `@markdown`: the function's docstring becomes one markdown cell.
  - `@code`:     the function's body (dedented) becomes one code cell.

Function names don't matter; cells appear in source order. The generator is
AST-based and deterministic — the same input always produces byte-identical
outputs, so it's safe to regenerate on every edit.

Outputs land at `tutorials/<name>/<name>.py` and `tutorials/<name>/<name>.ipynb`.

Usage:
    uv run python generate_tutorial.py                   # regenerate all
    uv run python generate_tutorial.py <input.py> ...    # regenerate specific
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import textwrap
import urllib.parse
from dataclasses import dataclass

TUTORIALS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TUTORIALS_DIR.parent
INPUT_DIR = TUTORIALS_DIR / "tutorial_generator"
OUTPUT_ROOT = TUTORIALS_DIR

# ── README table generation ─────────────────────────────────────────────────
# The tutorials/README.md table is auto-generated from a top-level
# `TUTORIAL_METADATA = { "framework": ..., "cluster_shape": ..., "summary": ...,
# "order": <int> }` dict declared in each source file under
# tutorial_generator/. Edit that dict to edit the table row — do not hand-edit
# the README.
_README_PATH = TUTORIALS_DIR / "README.md"
_README_BEGIN = "<!-- BEGIN TUTORIAL TABLE -->"
_README_END = "<!-- END TUTORIAL TABLE -->"
_REPO_SLUG = "modal-projects/training-gym"
_BRANCH = "joy/initial-setup"
_MODAL_WORKSPACE = "modal-labs/joy-dev"
_BADGE_IMG = "https://modal-cdn.com/open-in-modal.svg"

_MARKDOWN = "markdown"
_CODE = "code"
_SHELL = "shell"
_PY_ONLY = "py_only"
_NOTEBOOK_ONLY = "notebook_only"

# Targets a cell can appear in.
_PY = "py"
_NB = "notebook"


@dataclass
class Cell:
    kind: str  # "markdown" | "code"
    source: str
    targets: frozenset[str]  # subset of {"py", "notebook"}


def _decorator_name(node: ast.expr) -> str | None:
    """Return the bare decorator name, handling @foo, @mod.foo, and @foo(...)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _resolve_targets(deco_names: set[str | None]) -> frozenset[str]:
    py_only = _PY_ONLY in deco_names
    nb_only = _NOTEBOOK_ONLY in deco_names
    if py_only and nb_only:
        raise ValueError("@py_only and @notebook_only are mutually exclusive on the same cell")
    if py_only:
        return frozenset({_PY})
    if nb_only:
        return frozenset({_NB})
    return frozenset({_PY, _NB})


def _find_shell_command(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Return the string arg of a `@shell("…")` decorator, if any."""
    for deco in node.decorator_list:
        if not isinstance(deco, ast.Call):
            continue
        if _decorator_name(deco.func) != _SHELL:
            continue
        if not deco.args:
            continue
        arg0 = deco.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            return arg0.value
    return None


def _extract_cells(source: str) -> list[Cell]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    cells: list[Cell] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        deco_names = {_decorator_name(d) for d in node.decorator_list}
        targets = _resolve_targets(deco_names)

        # `@shell("…")` — emit the string verbatim as a code cell.
        shell_cmd = _find_shell_command(node)
        if shell_cmd is not None:
            cells.append(Cell(kind="code", source=shell_cmd, targets=targets))
            continue

        if _MARKDOWN in deco_names:
            doc = ast.get_docstring(node, clean=True) or ""
            cells.append(Cell(kind="markdown", source=doc, targets=targets))
        elif _CODE in deco_names:
            if not node.body:
                continue
            start = node.body[0].lineno - 1
            end = node.body[-1].end_lineno  # inclusive; slice stop = end
            body_src = "".join(lines[start:end])
            body_src = textwrap.dedent(body_src).rstrip("\n")
            cells.append(Cell(kind="code", source=body_src, targets=targets))
    return cells


def _render_py(cells: list[Cell], header: str) -> str:
    chunks: list[str] = [header.rstrip()] if header else []
    for cell in cells:
        if _PY not in cell.targets:
            continue
        if cell.kind == "markdown":
            chunks.append(
                "\n".join(f"# {ln}" if ln else "#" for ln in cell.source.splitlines())
            )
        else:
            chunks.append(cell.source)
    return "\n\n".join(c for c in chunks if c) + "\n"


def _nb_source_lines(text: str) -> list[str]:
    """Split into lines with trailing newlines (nbformat convention)."""
    if not text:
        return []
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _render_ipynb(cells: list[Cell]) -> str:
    visible = [c for c in cells if _NB in c.targets]
    nb_cells = []
    for i, cell in enumerate(visible):
        entry = {
            "cell_type": cell.kind,
            "id": f"cell-{i:03d}",
            "metadata": {},
            "source": _nb_source_lines(cell.source),
        }
        if cell.kind == "code":
            entry["execution_count"] = None
            entry["outputs"] = []
        nb_cells.append(entry)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1, sort_keys=True) + "\n"


def generate_one(
    input_path: pathlib.Path, output_root: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    source = input_path.read_text()
    cells = _extract_cells(source)
    name = input_path.stem
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    py_path = out_dir / f"{name}.py"
    ipynb_path = out_dir / f"{name}.ipynb"

    rel_src = input_path.relative_to(REPO_ROOT).as_posix()
    header = f"# Generated by generate_tutorial.py — do not edit directly.\n# Source: {rel_src}\n"
    py_path.write_text(_render_py(cells, header=header))
    ipynb_path.write_text(_render_ipynb(cells))
    return py_path, ipynb_path


def _extract_metadata(source: str) -> dict | None:
    """Return the top-level `TUTORIAL_METADATA = {...}` dict if present."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TUTORIAL_METADATA":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
                if isinstance(value, dict):
                    return value
    return None


def _render_launch_cell(name: str) -> str:
    raw_url = (
        f"https://raw.githubusercontent.com/{_REPO_SLUG}/{_BRANCH}"
        f"/tutorials/{name}/{name}.ipynb"
    )
    launch_url = (
        f"https://modal.com/notebooks/{_MODAL_WORKSPACE}"
        f"?import=url&url={urllib.parse.quote(raw_url, safe='')}"
    )
    # Explicit `rel="nofollow noopener noreferrer"` matches GitHub's own
    # auto-added rel, which ensures GitHub's markdown sanitizer preserves
    # `target="_blank"` rather than stripping it.
    return (
        f'<a href="{launch_url}" target="_blank" '
        f'rel="nofollow noopener noreferrer">'
        f'<img src="{_BADGE_IMG}" alt="Open in Modal"></a>'
    )


def _render_tutorial_table(entries: list[tuple[str, dict]]) -> str:
    lines = [
        "| Tutorial | Difficulty | Framework | Cluster shape | What it trains | Launch |",
        "|---|---|---|---|---|---|",
    ]
    for name, meta in entries:
        lines.append(
            f"| [`{name}`]({name}/{name}.ipynb)"
            f" | {meta.get('difficulty', '—')}"
            f" | {meta['framework']}"
            f" | {meta['cluster_shape']}"
            f" | {meta['summary']}"
            f" | {_render_launch_cell(name)} |"
        )
    return "\n".join(lines)


def _update_readme_table(entries: list[tuple[str, dict]]) -> bool:
    """Rewrite the `<!-- BEGIN TUTORIAL TABLE --> ... <!-- END TUTORIAL TABLE -->`
    region of tutorials/README.md. Returns True if the file changed."""
    if not _README_PATH.exists():
        print(f"Skipping README update: {_README_PATH} not found")
        return False
    content = _README_PATH.read_text()
    if _README_BEGIN not in content or _README_END not in content:
        print(
            f"Skipping README update: markers {_README_BEGIN!r} / {_README_END!r} "
            f"not found in {_README_PATH.relative_to(REPO_ROOT)}"
        )
        return False
    before, rest = content.split(_README_BEGIN, 1)
    _, after = rest.split(_README_END, 1)
    banner = (
        "<!-- Auto-generated by generate_tutorial.py from TUTORIAL_METADATA in "
        "each tutorial source. Edit metadata there, not here. -->"
    )
    new_body = (
        f"{_README_BEGIN}\n{banner}\n\n"
        f"{_render_tutorial_table(entries)}\n"
        f"{_README_END}"
    )
    new_content = f"{before}{new_body}{after}"
    if new_content == content:
        return False
    _README_PATH.write_text(new_content)
    return True


def _collect_all_metadata() -> list[tuple[str, dict]]:
    """Walk every tutorial source and return `(name, metadata)` sorted by
    display `order` (defaulting to alphabetical on name when unset)."""
    entries: list[tuple[str, dict]] = []
    for path in sorted(INPUT_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        meta = _extract_metadata(path.read_text())
        if meta is None:
            print(f"  (no TUTORIAL_METADATA in {path.relative_to(REPO_ROOT)})")
            continue
        entries.append((path.stem, meta))
    entries.sort(key=lambda item: (item[1].get("order", 10_000), item[0]))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=pathlib.Path,
        help="Specific input files (defaults to every *.py in tutorials/tutorial_generator/)",
    )
    args = parser.parse_args()

    inputs = args.inputs or sorted(
        p for p in INPUT_DIR.glob("*.py") if p.name != "__init__.py"
    )
    if not inputs:
        print(f"No input files found in {INPUT_DIR.relative_to(REPO_ROOT)}")
        return

    for inp in inputs:
        py, ipynb = generate_one(inp.resolve(), OUTPUT_ROOT)
        print(
            f"{inp.relative_to(REPO_ROOT) if inp.is_absolute() else inp} "
            f"→ {py.relative_to(REPO_ROOT)} + {ipynb.relative_to(REPO_ROOT)}"
        )

    # Rewrite the README table from metadata declared in each tutorial source.
    # Always iterates every source (not just `inputs`) so the table reflects
    # the whole catalog even when a single file is regenerated.
    entries = _collect_all_metadata()
    if entries and _update_readme_table(entries):
        print(f"{_README_PATH.relative_to(REPO_ROOT)} ← table regenerated ({len(entries)} rows)")


if __name__ == "__main__":
    main()
