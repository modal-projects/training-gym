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
from dataclasses import dataclass

TUTORIALS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TUTORIALS_DIR.parent
INPUT_DIR = TUTORIALS_DIR / "tutorial_generator"
OUTPUT_ROOT = TUTORIALS_DIR

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


if __name__ == "__main__":
    main()
