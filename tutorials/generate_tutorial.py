#!/usr/bin/env python3
"""Generate .py + .ipynb tutorials from decorator-annotated source files.

Input files live in `tutorials/tutorial_generator/<bucket>/<name>.py` where
`<bucket>` is one of `intro`, `rl`, `sft`, `misc`, and declare cells via
top-level functions decorated with `@markdown` or `@code`:

  - `@markdown`: the function's docstring becomes one markdown cell.
  - `@code`:     the function's body (dedented) becomes one code cell.

Function names don't matter; cells appear in source order. The generator is
AST-based and deterministic — the same input always produces byte-identical
outputs, so it's safe to regenerate on every edit.

Outputs land at `tutorials/<bucket>/<name>/<name>.py` and
`tutorials/<bucket>/<name>/<name>.ipynb`.

Usage:
    uv run generate_tutorial.py                   # regenerate all
    uv run generate_tutorial.py <input.py> ...    # regenerate specific
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import symtable
import textwrap
import urllib.parse
from dataclasses import dataclass

TUTORIALS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TUTORIALS_DIR.parent
INPUT_DIR = TUTORIALS_DIR / "tutorial_generator"
OUTPUT_ROOT = TUTORIALS_DIR

# ── README table generation ─────────────────────────────────────────────────
# The repo-root README.md table is auto-generated from a top-level
# `TUTORIAL_METADATA = { "framework": ..., "cluster_shape": ..., "summary": ...,
# "order": <int> }` dict declared in each source file under
# tutorial_generator/. Edit that dict to edit the table row — do not hand-edit
# the README.
_README_PATH = REPO_ROOT / "README.md"
_README_BEGIN = "<!-- BEGIN TUTORIAL TABLE -->"
_README_END = "<!-- END TUTORIAL TABLE -->"
_REPO_SLUG = "modal-projects/training-gym"
_BADGE_IMG = "https://modal-cdn.com/open-in-modal.svg"
_BRANCH = "main"
_DOCS_BASE = "https://gym.modal.dev"

_MARKDOWN = "markdown"
_CODE = "code"
_SHELL = "shell"
_PY_ONLY = "py_only"
_NOTEBOOK_ONLY = "notebook_only"

# Targets a cell can appear in.
_PY = "py"
_NB = "notebook"

# Buckets the tutorial catalog is grouped into. Display order in the README
# sections and ordering within each bucket fall back to meta["order"].
_BUCKETS = ("intro", "rl", "sft", "singlenode", "agent", "multinode", "misc")
_DOCS_PAGE_BUCKETS = frozenset({"intro", "rl", "sft", "agent", "misc"})
_BUCKET_DISPLAY = {
    "intro": "Intro",
    "rl": "RL",
    "sft": "SFT",
    "singlenode": "Single Node",
    "agent": "Agents",
    "multinode": "Multinode",
    "misc": "Misc",
}
# Injected before every tutorial's first code cell so missing secrets fail
# fast locally instead of mid-launch on a Modal worker.
_DEFAULT_REQUIRED_MODAL_SECRETS = (
    {
        "name": "huggingface-secret",
        "key": "HF_TOKEN",
    },
)
_NOTEBOOK_GPU_NOTE_MARKDOWN = (
    "> **Note:** you do **not** need to attach a GPU to this notebook. All training and\n"
    "> serving happens on Modal-managed GPU workers spun up by the SDK — the notebook\n"
    "> itself only needs to issue API calls."
)
_MULTINODE_DISCLAIMER_MARKDOWN = (
    "> **Multi-node workspace required:** This is a multi-node example. To run it,\n"
    "> your Modal workspace must have multi-node enabled. Contact\n"
    "> [support@modal.com](mailto:support@modal.com) to enable multi-node."
)


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
        raise ValueError(
            "@py_only and @notebook_only are mutually exclusive on the same cell"
        )
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


def _extract_cells(
    source: str,
    *,
    required_modal_secrets: tuple[dict[str, str], ...],
    include_multinode_disclaimer: bool = False,
) -> list[Cell]:
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
    cells = _inject_secret_check(cells, required_modal_secrets)
    if include_multinode_disclaimer:
        cells = _inject_multinode_disclaimer(cells)
    return cells


def _inject_multinode_disclaimer(cells: list[Cell]) -> list[Cell]:
    """Prepend the multi-node workspace warning to multi-node tutorials."""
    return [
        Cell(
            kind="markdown",
            source=_MULTINODE_DISCLAIMER_MARKDOWN,
            targets=frozenset({_PY, _NB}),
        ),
        *cells,
    ]


def _secret_check_markdown(required_modal_secrets: tuple[dict[str, str], ...]) -> str:
    if len(required_modal_secrets) == 1:
        secret = required_modal_secrets[0]
        return (
            "## Prerequisites\n"
            "\n"
            f"This tutorial requires a Modal Secret named `{secret['name']}` containing your\n"
            f"`{secret['key']}`. Create one at [modal.com/secrets](https://modal.com/secrets) if you\n"
            "haven't already — the cell below fails fast with instructions otherwise."
        )

    secret_lines = "\n".join(
        f"- `{secret['name']}` containing `{secret['key']}`"
        for secret in required_modal_secrets
    )
    return (
        "## Prerequisites\n"
        "\n"
        "This tutorial requires these Modal Secrets:\n"
        f"{secret_lines}\n"
        "\n"
        "Create them at [modal.com/secrets](https://modal.com/secrets) if you "
        "haven't already — the cell below fails fast with instructions otherwise."
    )


def _secret_check_code(required_modal_secrets: tuple[dict[str, str], ...]) -> str:
    if len(required_modal_secrets) == 1:
        secret = required_modal_secrets[0]
        if secret == _DEFAULT_REQUIRED_MODAL_SECRETS[0]:
            return (
                "import modal\n"
                "\n"
                "try:\n"
                '    modal.Secret.from_name("huggingface-secret").hydrate()\n'
                "except modal.exception.NotFoundError as e:\n"
                "    raise RuntimeError(\n"
                "        \"Missing Modal Secret 'huggingface-secret'. Create one at \"\n"
                '        "https://modal.com/secrets with an HF_TOKEN entry, then re-run."\n'
                "    ) from e"
            )
        return (
            "import modal\n"
            "\n"
            "try:\n"
            f'    modal.Secret.from_name("{secret["name"]}", required_keys=["{secret["key"]}"]).hydrate()\n'
            "except modal.exception.NotFoundError as e:\n"
            "    raise RuntimeError(\n"
            f"        \"Missing Modal Secret '{secret['name']}'. Create one at \"\n"
            f'        "https://modal.com/secrets with a {secret["key"]} entry, then re-run."\n'
            "    ) from e"
        )

    secret_list = ",\n".join(
        f'    ("{secret["name"]}", "{secret["key"]}")'
        for secret in required_modal_secrets
    )
    return (
        "import modal\n"
        "\n"
        "for secret_name, required_key in [\n"
        f"{secret_list},\n"
        "]:\n"
        "    try:\n"
        "        modal.Secret.from_name(\n"
        "            secret_name, required_keys=[required_key]\n"
        "        ).hydrate()\n"
        "    except modal.exception.NotFoundError as e:\n"
        "        raise RuntimeError(\n"
        "            f\"Missing Modal Secret '{secret_name}'. Create one at \"\n"
        '            f"https://modal.com/secrets with a {required_key} entry, then re-run."\n'
        "        ) from e"
    )


def _inject_secret_check(
    cells: list[Cell], required_modal_secrets: tuple[dict[str, str], ...]
) -> list[Cell]:
    """Insert the Modal secret precheck right before the first code cell."""
    both = frozenset({_PY, _NB})
    nb_only = frozenset({_NB})
    prereq = [
        Cell(
            kind="markdown",
            source=_secret_check_markdown(required_modal_secrets),
            targets=both,
        ),
        Cell(kind="markdown", source=_NOTEBOOK_GPU_NOTE_MARKDOWN, targets=nb_only),
        Cell(
            kind="code", source=_secret_check_code(required_modal_secrets), targets=both
        ),
    ]
    for i, cell in enumerate(cells):
        if cell.kind == "code":
            return cells[:i] + prereq + cells[i:]
    return cells + prereq


def _render_py(cells: list[Cell], header: str) -> str:
    chunks: list[str] = [header.rstrip()] if header else []
    module_chunks: list[str] = []
    main_chunks: list[str] = []
    pending_comment_lines: list[str] = []
    globals_used_by_defs = _py_globals_used_by_definitions(cells)

    for cell in cells:
        if _PY not in cell.targets:
            continue
        if cell.kind == "markdown":
            pending_comment_lines.extend(
                f"# {ln}" if ln else "#" for ln in cell.source.splitlines()
            )
            continue

        blocks = _split_py_code_cell(
            cell.source,
            globals_used_by_defs=globals_used_by_defs,
        )
        if not blocks:
            continue

        pending_comment_block = "\n".join(pending_comment_lines).rstrip()
        pending_comment_lines = []
        for i, (scope, segment) in enumerate(blocks):
            block = segment
            if i == 0 and pending_comment_block:
                block = f"{pending_comment_block}\n\n{block}"
            if scope == "module":
                module_chunks.append(block)
            else:
                main_chunks.append(block)

    if pending_comment_lines:
        module_chunks.append("\n".join(pending_comment_lines))

    chunks.extend(module_chunks)

    if main_chunks:
        chunks.append("import modal\n\ntutorial_cli_app = modal.App()")
        indented = "\n\n".join(textwrap.indent(chunk, "    ") for chunk in main_chunks)
        chunks.append(f"def _main_impl() -> None:\n{indented}")
        chunks.append(
            "@tutorial_cli_app.local_entrypoint()\n"
            "def main() -> None:\n"
            "    _main_impl()"
        )
        chunks.append('if __name__ == "__main__":\n    main()')

    return "\n\n".join(c for c in chunks if c) + "\n"


def _py_globals_used_by_definitions(cells: list[Cell]) -> set[str]:
    py_code = [
        cell.source for cell in cells if _PY in cell.targets and cell.kind == "code"
    ]
    if not py_code:
        return set()

    table = symtable.symtable("\n\n".join(py_code), "<generated_tutorial>", "exec")
    refs: set[str] = set()

    def _walk(child_table: symtable.SymbolTable) -> None:
        if child_table.get_type() != "module":
            for symbol in child_table.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    refs.add(symbol.get_name())
        for nested in child_table.get_children():
            _walk(nested)

    _walk(table)
    return refs


def _split_py_code_cell(
    source: str,
    *,
    globals_used_by_defs: set[str],
) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    if not tree.body:
        return []

    lines = source.splitlines(keepends=True)
    blocks: list[tuple[str, str]] = []

    stmt_scopes = _resolve_module_scope_statements(tree.body, globals_used_by_defs)
    group_start = 0

    for i in range(1, len(tree.body) + 1):
        same_scope = i < len(tree.body) and stmt_scopes[i] == stmt_scopes[group_start]
        if same_scope:
            continue

        start = 0 if group_start == 0 else tree.body[group_start].lineno - 1
        end = tree.body[i].lineno - 1 if i < len(tree.body) else len(lines)
        segment = "".join(lines[start:end]).rstrip("\n")
        if segment:
            scope = "module" if stmt_scopes[group_start] else "main"
            blocks.append((scope, segment))
        group_start = i

    return blocks


def _resolve_module_scope_statements(
    statements: list[ast.stmt],
    globals_used_by_defs: set[str],
) -> list[bool]:
    stmt_scopes = [
        _stmt_belongs_at_module_scope(stmt, globals_used_by_defs) for stmt in statements
    ]

    changed = True
    while changed:
        changed = False
        names_required_by_module_scope = set(globals_used_by_defs)
        for stmt, belongs_at_module_scope in zip(statements, stmt_scopes, strict=True):
            if belongs_at_module_scope:
                names_required_by_module_scope.update(_module_scope_references(stmt))

        for i, stmt in enumerate(statements):
            if stmt_scopes[i]:
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and (
                _assigned_names(stmt) & names_required_by_module_scope
            ):
                stmt_scopes[i] = True
                changed = True

    return stmt_scopes


def _stmt_belongs_at_module_scope(
    stmt: ast.stmt,
    globals_used_by_defs: set[str],
) -> bool:
    if isinstance(
        stmt,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    ):
        return True
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return bool(_assigned_names(stmt) & globals_used_by_defs)
    return False


def _module_scope_references(stmt: ast.stmt) -> set[str]:
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        nodes: list[ast.AST] = []
        if isinstance(stmt, ast.Assign):
            nodes.extend(stmt.targets)
            nodes.append(stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            nodes.append(stmt.target)
            nodes.append(stmt.annotation)
            if stmt.value is not None:
                nodes.append(stmt.value)
        else:
            nodes.append(stmt.target)
            nodes.append(stmt.value)
        return _referenced_names(nodes)

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nodes = [*stmt.decorator_list, *stmt.args.defaults, *stmt.args.kw_defaults]
        if stmt.returns is not None:
            nodes.append(stmt.returns)
        return _referenced_names([node for node in nodes if node is not None])

    if isinstance(stmt, ast.ClassDef):
        return _referenced_names([*stmt.decorator_list, *stmt.bases, *stmt.keywords])

    return _referenced_names([stmt])


def _referenced_names(nodes: list[ast.AST]) -> set[str]:
    refs: set[str] = set()

    class ReferenceVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                refs.add(node.id)

    visitor = ReferenceVisitor()
    for node in nodes:
        visitor.visit(node)
    return refs


def _assigned_names(stmt: ast.Assign | ast.AnnAssign | ast.AugAssign) -> set[str]:
    if isinstance(stmt, ast.Assign):
        names: set[str] = set()
        for target in stmt.targets:
            names.update(_names_in_target(target))
        return names
    return _names_in_target(stmt.target)


def _names_in_target(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names.update(_names_in_target(elt))
        return names
    return set()


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


def _bucket_for(input_path: pathlib.Path) -> str:
    """Return the bucket subfolder name for a given tutorial source.

    Expects the `tutorials/tutorial_generator/<bucket>/<name>.py` layout. Raises
    if a source file is placed outside a bucket subdirectory.
    """
    rel = input_path.resolve().relative_to(INPUT_DIR.resolve())
    parts = rel.parts
    if len(parts) != 2:
        raise ValueError(
            f"Tutorial source {input_path} is not inside a bucket subdirectory of "
            f"{INPUT_DIR.relative_to(REPO_ROOT)}. Expected "
            f"tutorial_generator/<bucket>/<name>.py."
        )
    bucket = parts[0]
    if bucket not in _BUCKETS:
        raise ValueError(
            f"Tutorial source {input_path} lives under unknown bucket {bucket!r}. "
            f"Expected one of {_BUCKETS}."
        )
    return bucket


def generate_one(
    input_path: pathlib.Path, output_root: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    source = input_path.read_text()
    name = input_path.stem
    metadata = _extract_metadata(source) or {}
    bucket = _bucket_for(input_path)
    cells = _extract_cells(
        source,
        required_modal_secrets=_required_modal_secrets(metadata),
        include_multinode_disclaimer=bucket == "multinode",
    )
    out_dir = output_root / bucket / name
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


def _required_modal_secrets(metadata: dict) -> tuple[dict[str, str], ...]:
    extra_secrets = metadata.get("required_modal_secrets", ())
    if not isinstance(extra_secrets, (list, tuple)):
        raise ValueError("required_modal_secrets must be a list of {name, key} dicts")

    secrets = list(_DEFAULT_REQUIRED_MODAL_SECRETS)
    for secret in extra_secrets:
        if not isinstance(secret, dict):
            raise ValueError("required_modal_secrets must contain {name, key} dicts")
        name = secret.get("name")
        key = secret.get("key")
        if not isinstance(name, str) or not isinstance(key, str):
            raise ValueError("required_modal_secrets entries need string name and key")
        secrets.append({"name": name, "key": key})

    return tuple(secrets)


def _render_launch_cell(bucket: str, name: str) -> str:
    notebook_url = (
        f"https://github.com/{_REPO_SLUG}/blob/{_BRANCH}"
        f"/tutorials/{bucket}/{name}/{name}.ipynb"
    )
    launch_url = (
        f"https://modal.com/notebooks/new/{urllib.parse.quote(notebook_url, safe='')}"
    )
    # Explicit `rel="nofollow noopener noreferrer"` matches GitHub's own
    # auto-added rel, which ensures GitHub's markdown sanitizer preserves
    # `target="_blank"` rather than stripping it.
    return (
        f'<a href="{launch_url}" target="_blank" '
        f'rel="nofollow noopener noreferrer">'
        f'<img src="{_BADGE_IMG}" alt="Open in Modal"></a>'
    )


def _render_tutorial_table(
    bucket: str, bucket_entries: list[tuple[str, str, dict]]
) -> str:
    lines = [
        "| Tutorial | Summary | Difficulty | Framework | Launch |",
        "|---|---|---|---|---|",
    ]
    for _, name, meta in bucket_entries:
        summary = str(meta["summary"]).rstrip(".").replace("|", r"\|")
        difficulty = str(meta.get("difficulty", "—")).replace("|", r"\|")
        framework = str(meta["framework"]).replace("|", r"\|")
        launch = _render_launch_cell(bucket, name)
        if bucket in _DOCS_PAGE_BUCKETS:
            link = f"{_DOCS_BASE}/tutorials/{bucket}/{name}/"
        else:
            link = f"tutorials/{bucket}/{name}/{name}.ipynb"
        lines.append(
            f"| [`{name}`]({link}) | {summary} | "
            f"{difficulty} | {framework} | {launch} |"
        )
    return "\n".join(lines)


def _render_tutorial_sections(entries: list[tuple[str, str, dict]]) -> str:
    """Render the catalog as one H3 section per bucket, each with a table."""
    lines: list[str] = []
    for bucket in _BUCKETS:
        bucket_entries = [entry for entry in entries if entry[0] == bucket]
        if not bucket_entries:
            continue
        if lines:
            lines.append("")
        lines.append(f"### {_BUCKET_DISPLAY.get(bucket, bucket)}")
        lines.append("")
        lines.append(_render_tutorial_table(bucket, bucket_entries))
    return "\n".join(lines)


def _update_readme_table(entries: list[tuple[str, str, dict]]) -> bool:
    """Rewrite the `<!-- BEGIN TUTORIAL TABLE --> ... <!-- END TUTORIAL TABLE -->`
    region of the repo-root README.md. Returns True if the file changed."""
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
        f"{_render_tutorial_sections(entries)}\n"
        f"{_README_END}"
    )
    new_content = f"{before}{new_body}{after}"
    if new_content == content:
        return False
    _README_PATH.write_text(new_content)
    return True


def _iter_source_files() -> list[pathlib.Path]:
    """Return every tutorial source under `tutorial_generator/<bucket>/`.

    Skips `__init__.py` at any level and files that aren't inside a known
    bucket subdirectory (so a stray top-level file produces a clear error in
    `_bucket_for` rather than a silent no-op).
    """
    return sorted(
        p
        for p in INPUT_DIR.rglob("*.py")
        if p.name != "__init__.py"
    )


def _collect_all_metadata() -> list[tuple[str, str, dict]]:
    """Walk every tutorial source and return `(bucket, name, metadata)` sorted by
    bucket display order, then `order`, then alphabetical name."""
    entries: list[tuple[str, str, dict]] = []
    for path in _iter_source_files():
        try:
            bucket = _bucket_for(path)
        except ValueError as exc:
            print(f"  (skipping {path.relative_to(REPO_ROOT)}: {exc})")
            continue
        meta = _extract_metadata(path.read_text())
        if meta is None:
            print(f"  (no TUTORIAL_METADATA in {path.relative_to(REPO_ROOT)})")
            continue
        entries.append((bucket, path.stem, meta))
    bucket_rank = {b: i for i, b in enumerate(_BUCKETS)}
    entries.sort(
        key=lambda item: (
            bucket_rank.get(item[0], len(_BUCKETS)),
            item[2].get("order", 10_000),
            item[1],
        )
    )
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

    inputs = args.inputs or _iter_source_files()
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
        print(
            f"{_README_PATH.relative_to(REPO_ROOT)} ← table regenerated ({len(entries)} rows)"
        )


if __name__ == "__main__":
    main()
