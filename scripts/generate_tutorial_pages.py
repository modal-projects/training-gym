"""Generate individual Starlight pages for each tutorial source."""

from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

from api_reference_manifest import CLASS_REFERENCE_PATHS

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_SRC_DIR = ROOT / "tutorials" / "tutorial_generator"
DEFAULT_OUTPUT_DIR = ROOT / "docs-next" / "src" / "content" / "docs" / "tutorials"
REPO_URL = "https://github.com/modal-projects/training-gym"

BUCKETS = ["intro", "rl", "sft", "misc"]
BUCKET_LABELS = {
    "intro": "Getting Started",
    "rl": "Reinforcement Learning",
    "sft": "Supervised Fine-Tuning",
    "misc": "Infrastructure",
}


def extract_metadata(source_path: Path) -> dict | None:
    """Extract TUTORIAL_METADATA dict from a tutorial source file via AST."""
    try:
        tree = ast.parse(source_path.read_text())
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TUTORIAL_METADATA":
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


def extract_cells(source_path: Path) -> list[tuple[str, str]]:
    """Extract (cell_type, content) from decorated tutorial functions."""
    tree = ast.parse(source_path.read_text())
    source_lines = source_path.read_text().splitlines()
    cells: list[tuple[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.decorator_list:
            continue

        decorator = node.decorator_list[0]
        if isinstance(decorator, ast.Name):
            dec_name = decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            dec_name = decorator.func.id
        else:
            continue

        if dec_name == "markdown":
            docstring = ast.get_docstring(node)
            if docstring:
                cells.append(("markdown", textwrap.dedent(docstring).strip()))
        elif dec_name == "code":
            body_lines = []
            first_line = node.body[0].lineno - 1
            if (
                isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))
            ):
                first_line = node.body[1].lineno - 1 if len(node.body) > 1 else node.end_lineno or first_line
            last_line = (node.end_lineno or first_line + 1)
            raw = source_lines[first_line:last_line]
            if raw:
                indent = len(raw[0]) - len(raw[0].lstrip())
                body_lines = [line[indent:] if len(line) > indent else "" for line in raw]
            cells.append(("code", "\n".join(body_lines).strip()))
        elif dec_name == "shell":
            if isinstance(decorator, ast.Call) and decorator.args:
                arg = decorator.args[0]
                if isinstance(arg, ast.Constant):
                    cells.append(("shell", str(arg.value)))

    return cells


def generate_tutorial_page(
    source_path: Path,
    bucket: str,
    metadata: dict,
) -> str:
    """Generate a Starlight markdown page from a tutorial source."""
    name = source_path.stem
    cells = extract_cells(source_path)

    api_classes = metadata.get("api_classes", [])
    for cls_name in api_classes:
        if cls_name not in CLASS_REFERENCE_PATHS:
            print(f"  WARNING: {source_path.name} references '{cls_name}' in api_classes but it is not in the manifest", file=sys.stderr)

    title = metadata.get("summary", name)
    found_h1 = False
    for cell_type, content in cells:
        if cell_type == "markdown" and not found_h1:
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    found_h1 = True
                    break

    lines = [
        "---",
        f"title: \"{title}\"",
        f"description: \"{metadata.get('summary', '')}\"",
        "---",
        "",
    ]

    first_markdown = True
    for cell_type, content in cells:
        if cell_type == "markdown":
            if first_markdown and found_h1:
                content_lines = content.splitlines()
                content_lines = [l for i, l in enumerate(content_lines)
                                 if not (l.startswith("# ") and i == 0)]
                content = "\n".join(content_lines).strip()
                first_markdown = False
            if content:
                lines.append(content)
                lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(content)
            lines.append("```")
            lines.append("")
        elif cell_type == "shell":
            lines.append("```bash")
            lines.append(content)
            lines.append("```")
            lines.append("")

    if api_classes:
        lines.append("---")
        lines.append("")
        lines.append("## Related API Reference")
        lines.append("")
        for cls_name in api_classes:
            ref_path = CLASS_REFERENCE_PATHS.get(cls_name)
            if ref_path:
                lines.append(f"- [`{cls_name}`]({ref_path})")
            else:
                lines.append(f"- `{cls_name}`")
        lines.append("")

    py_path = f"tutorials/{bucket}/{name}/{name}.py"
    nb_path = f"tutorials/{bucket}/{name}/{name}.ipynb"
    nb_url = f"https://modal.com/notebooks/new/{REPO_URL}/blob/joy/initial-setup/{nb_path}"
    lines.append(f"**Source:** [`{py_path}`]({REPO_URL}/blob/joy/initial-setup/{py_path})")
    lines.append(f' | <a href="{nb_url}" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>')
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tutorial Starlight pages.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for tutorial pages.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    generated = 0

    for bucket in BUCKETS:
        bucket_dir = TUTORIAL_SRC_DIR / bucket
        if not bucket_dir.is_dir():
            continue

        for source_path in sorted(bucket_dir.glob("*.py")):
            if source_path.name.startswith("_") or source_path.name == "__init__.py":
                continue

            metadata = extract_metadata(source_path)
            if metadata is None:
                print(f"  SKIP {source_path.name}: no TUTORIAL_METADATA found")
                continue

            content = generate_tutorial_page(source_path, bucket, metadata)
            out_dir = output_dir / bucket
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{source_path.stem}.md"
            if out_path.exists():
                print(f"  WARNING: overwriting existing tutorial page {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}", file=sys.stderr)
            out_path.write_text(content)
            print(f"  {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")
            generated += 1

    print(f"\nGenerated {generated} tutorial pages")


if __name__ == "__joy/initial-setup__":
    joy/initial-setup()
