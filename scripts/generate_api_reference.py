"""Generate Starlight API reference pages from the curated class manifest."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import re
import shutil
import sys
from dataclasses import fields as dataclass_fields
from enum import Enum
from pathlib import Path
from typing import Any, get_type_hints

from api_reference_manifest import API_REFERENCE_MANIFEST, GROUPS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs-next" / "src" / "content" / "docs" / "reference"
REPO_URL = "https://github.com/modal-projects/training-gym"


def _is_dataclass(cls: type) -> bool:
    return hasattr(cls, "__dataclass_fields__")


def _get_class_attrs(cls: type) -> dict[str, tuple[type, Any]]:
    """Extract class-level attributes with their types and defaults."""
    hints = {}
    try:
        hints = get_type_hints(cls)
    except Exception:
        for klass in cls.__mro__:
            hints.update(getattr(klass, "__annotations__", {}))

    attrs: dict[str, tuple[type, Any]] = {}

    if _is_dataclass(cls):
        MISSING = dataclasses.MISSING
        for f in dataclass_fields(cls):
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:
                default = f.default_factory()
            else:
                default = inspect.Parameter.empty
            attrs[f.name] = (hints.get(f.name, Any), default)
    else:
        init_defaults: dict[str, Any] = {}
        init_method = getattr(cls, "__init__", None)
        if init_method:
            try:
                sig = inspect.signature(init_method)
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    if param.default is not param.empty:
                        init_defaults[pname] = param.default
            except (ValueError, TypeError):
                pass

        docstring = inspect.getdoc(cls) or ""
        effective_defaults = _extract_effective_defaults(docstring)

        for name, type_hint in hints.items():
            if name.startswith("_"):
                continue
            default = getattr(cls, name, inspect.Parameter.empty)
            if default is inspect.Parameter.empty and name in init_defaults:
                default = init_defaults[name]
            if default is None and name in effective_defaults:
                default = effective_defaults[name]
            attrs[name] = (type_hint, default)

    return attrs


def _extract_effective_defaults(docstring: str) -> dict[str, _EffectiveDefault]:
    """Extract effective defaults documented as 'Default ``Foo()``.' in docstrings."""
    results: dict[str, _EffectiveDefault] = {}
    pattern = re.compile(r"^(\w+)\s*:\s*\S+")
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        m = pattern.match(lines[i])
        if m:
            field_name = m.group(1)
            i += 1
            while i < len(lines) and lines[i].strip() and not pattern.match(lines[i]):
                line = lines[i].strip()
                default_match = re.search(r"Default\s+``([^`]+\(\))``", line)
                if default_match:
                    results[field_name] = _EffectiveDefault(default_match.group(1))
                i += 1
        else:
            i += 1
    return results


def _rst_to_md(text: str) -> str:
    """Convert reST-style double backticks to Markdown single backticks."""
    return re.sub(r"``(.*?)``", r"`\1`", text)


def _orders_within_group() -> dict[str, int]:
    next_order: dict[str, int] = {}
    orders: dict[str, int] = {}
    for entry in API_REFERENCE_MANIFEST:
        group = entry["group"]
        orders[entry["class_name"]] = next_order.get(group, 0)
        next_order[group] = next_order.get(group, 0) + 1
    return orders


def _page_heading(order: int, title: str) -> list[str]:
    return [
        "---",
        f"order: {order}",
        "---",
        "",
        f"# {title}",
        "",
    ]


def _format_type(type_hint: Any) -> str:
    """Format a type hint for display."""
    raw = str(type_hint)
    raw = raw.replace("typing.", "")
    raw = raw.replace("modal_training_gym.common.models.base.", "")
    raw = raw.replace("modal_training_gym.common.dataset.", "")
    raw = raw.replace("modal_training_gym.common.wandb.", "")
    raw = raw.replace("modal_training_gym.common.eval.", "")
    raw = raw.replace("modal_training_gym.common.deployment.", "")
    raw = raw.replace("modal_training_gym.deploy_recipes.sglang_recipe.recipe.", "")
    raw = raw.replace("modal_training_gym.deploy_recipes.vllm_recipe.recipe.", "")
    raw = raw.replace("modal_training_gym.train_recipes.slime_recipe.recipe.", "")
    raw = re.sub(r"<(?:class|enum) '([^']+)'>", r"\1", raw)
    raw = raw.replace("|", "\\|")
    return raw


class _EffectiveDefault:
    """Wrapper for docstring-documented effective defaults."""

    def __init__(self, text: str):
        self.text = text


def _format_default(val: Any) -> str:
    if val is inspect.Parameter.empty:
        return ""
    if isinstance(val, _EffectiveDefault):
        return f"`{val.text}`"
    if isinstance(val, str):
        return f'`"{val}"`' if val else '`""`'
    if isinstance(val, bool):
        return f"`{val}`"
    if isinstance(val, Enum):
        return f"`{val.value}`"
    if isinstance(val, (int, float)):
        return f"`{val}`"
    if val is None:
        return "`None`"
    if isinstance(val, list) and not val:
        return "`[]`"
    if isinstance(val, dict) and not val:
        return "`{}`"
    return f"`{val!r}`"


def _parse_docstring_groups(docstring: str) -> list[tuple[str, list[str]]]:
    """Parse docstring for ## group headers, returning (group_name, [lines])."""
    groups: list[tuple[str, list[str]]] = []
    current_group: str | None = None
    current_lines: list[str] = []

    for line in docstring.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("## ")
            and not stripped.startswith("## Fields")
            and not stripped.startswith("## Methods")
        ):
            if current_group is not None:
                groups.append((current_group, current_lines))
            current_group = stripped[3:].strip()
            current_lines = []
        elif current_group is not None:
            current_lines.append(line)

    if current_group is not None:
        groups.append((current_group, current_lines))

    return groups


def _parse_docstring_groups_from_mro(cls: type) -> list[tuple[str, list[str]]]:
    """Parse docstring groups across the class MRO, subclass sections first."""
    seen_groups: set[str] = set()
    result: list[tuple[str, list[str]]] = []

    for klass in cls.__mro__:
        if klass is object:
            continue
        doc = inspect.getdoc(klass) or ""
        if not doc:
            continue
        groups = _parse_docstring_groups(doc)
        for group_name, group_lines in groups:
            if group_name not in seen_groups:
                result.append((group_name, group_lines))
                seen_groups.add(group_name)

    return result


def _extract_field_docs_from_mro(cls: type) -> dict[str, str]:
    """Extract field docs from docstrings across the class MRO."""
    merged: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        doc = inspect.getdoc(klass) or ""
        if doc:
            merged.update(_extract_field_docs(doc))
    return merged


def _extract_field_docs(docstring: str) -> dict[str, str]:
    """Extract field-level docs from structured docstring sections."""
    field_docs: dict[str, str] = {}
    pattern = re.compile(r"^(\w+)\s*:\s*\S+")
    lines = docstring.splitlines()

    i = 0
    while i < len(lines):
        m = pattern.match(lines[i])
        if m:
            field_name = m.group(1)
            desc_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() and not pattern.match(lines[i]):
                desc_lines.append(lines[i].strip())
                i += 1
            cleaned = " ".join(desc_lines)
            cleaned = re.sub(r"``(.*?)``", r"`\1`", cleaned)
            field_docs[field_name] = cleaned
        else:
            i += 1

    return field_docs


def _render_field_table(
    attrs: dict[str, tuple[Any, Any]],
    field_docs: dict[str, str],
    label: str = "Field",
) -> list[str]:
    """Render a `| <label> | Type | Default | Description |` markdown table."""
    lines = [
        f"| {label} | Type | Default | Description |",
        f"|{'-' * (len(label) + 2)}|------|---------|-------------|",
    ]
    for name, (type_hint, default) in attrs.items():
        desc = field_docs.get(name, "")
        lines.append(
            f"| `{name}` | `{_format_type(type_hint)}` | {_format_default(default)} | {desc} |"
        )
    return lines


def _page_preamble(cls: type, entry: dict, order: int) -> list[str]:
    """Shared page opening: frontmatter, import block, summary, inheritance."""
    docstring = inspect.getdoc(cls) or ""
    # Render all intro prose before the first section header.  Section
    # headers are either Markdown `## ...` headings or reST-style
    # underlined headings (e.g. `Attributes\n-----------`) that would
    # otherwise be duplicated by the generated ## Fields / ## Methods
    # tables on behavior pages.
    intro_lines: list[str] = []
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            break
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            title = line.strip()
            if (
                title
                and next_line
                and re.match(r"^[-=~]+$", next_line)
                and len(next_line) >= len(title)
            ):
                break
        intro_lines.append(line)
        i += 1
    first_para = "\n".join(intro_lines).strip()
    module_path = entry["module"]

    lines = [
        *_page_heading(order, entry["sidebar_label"]),
        "```python",
        f"from {module_path} import {entry['class_name']}",
        "```",
        "",
    ]

    if first_para:
        lines.append(_rst_to_md(first_para))
        lines.append("")

    parent_classes = [
        b.__name__
        for b in cls.__mro__[1:]
        if b.__name__ not in ("object", "type")
        and b.__module__.startswith("modal_training_gym")
    ]
    if parent_classes:
        lines.append(
            f"**Inherits from:** {', '.join(f'`{p}`' for p in parent_classes)}"
        )
        lines.append("")

    return lines


def _methods_section(cls: type) -> list[str]:
    """Render the ## Methods section, or nothing if the class has no methods."""
    methods = _get_methods(cls)
    if not methods:
        return []
    lines = ["## Methods", ""]
    for name, sig, doc in methods:
        lines.append(f"### `{name}{sig}`")
        lines.append("")
        if doc:
            lines.append(_rst_to_md(doc))
            lines.append("")
    return lines


def _page_footer(entry: dict) -> list[str]:
    lines: list[str] = []
    module_path = entry["module"]
    lines.append(
        f"**Source:** [`{module_path.replace('.', '/')}.py`]({REPO_URL}/blob/main/{module_path.replace('.', '/')}.py)"
    )
    lines.append("")
    return lines


def _get_methods(cls: type) -> list[tuple[str, str, str]]:
    """Get public methods with their signatures and docstrings."""
    methods = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        if attr is None or not callable(attr):
            continue
        if isinstance(attr, property):
            continue
        if not getattr(attr, "__module__", "").startswith("modal_training_gym"):
            continue
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            sig = None
        doc = inspect.getdoc(attr) or ""
        first_line = doc.splitlines()[0] if doc else ""
        sig_str = str(sig) if sig else "()"
        methods.append((name, sig_str, first_line))
    return methods


def generate_config_data_page(cls: type, entry: dict, order: int) -> str:
    """Generate markdown for a config/data class."""
    attrs = _get_class_attrs(cls)
    field_docs = _extract_field_docs_from_mro(cls)
    groups = _parse_docstring_groups_from_mro(cls)

    lines = _page_preamble(cls, entry, order)

    if not attrs:
        return "\n".join(lines)

    if len(attrs) > 15 and not groups:
        print(
            f"  WARNING: {entry['class_name']} has {len(attrs)} fields but no ## group headers in docstring",
            file=sys.stderr,
        )

    if groups:
        group_field_names: set[str] = set()
        for group_name, group_lines in groups:
            group_attrs: dict[str, tuple[Any, Any]] = {}
            for gl in group_lines:
                stripped = gl.strip()
                for attr_name in list(attrs.keys()):
                    if attr_name in group_field_names:
                        continue
                    if re.match(rf"^{attr_name}\s*:", stripped):
                        group_attrs[attr_name] = attrs[attr_name]
                        group_field_names.add(attr_name)

            lines.append(f"## {group_name}")
            lines.append("")
            lines.extend(_render_field_table(group_attrs, field_docs))
            lines.append("")

        ungrouped = {k: v for k, v in attrs.items() if k not in group_field_names}
        if ungrouped:
            lines.append("## Other Fields")
            lines.append("")
            lines.extend(_render_field_table(ungrouped, field_docs))
            lines.append("")
    else:
        lines.append("## Fields")
        lines.append("")
        lines.extend(_render_field_table(attrs, field_docs))
        lines.append("")

    lines.extend(_methods_section(cls))
    lines.extend(_page_footer(entry))

    return "\n".join(lines)


def generate_behavior_page(cls: type, entry: dict, order: int) -> str:
    """Generate markdown for a behavior class."""
    lines = _page_preamble(cls, entry, order)

    init = getattr(cls, "__init__", None)
    if init:
        try:
            sig = inspect.signature(init)
            params = {k: v for k, v in sig.parameters.items() if k != "self"}
        except (ValueError, TypeError):
            params = {}

        if params:
            lines.append("## Constructor")
            lines.append("")
            lines.append("```python")
            param_strs = []
            for pname, param in params.items():
                if param.kind == param.VAR_KEYWORD:
                    param_strs.append(f"**{pname}")
                elif param.default is not param.empty:
                    param_strs.append(f"{pname}={param.default!r}")
                else:
                    param_strs.append(pname)
            lines.append(f"{entry['class_name']}({', '.join(param_strs)})")
            lines.append("```")
            lines.append("")

            param_docs = _extract_field_docs_from_mro(cls)
            lines.append("| Parameter | Type | Default | Description |")
            lines.append("|-----------|------|---------|-------------|")
            for pname, param in params.items():
                if param.kind == param.VAR_KEYWORD:
                    continue
                type_str = (
                    _format_type(param.annotation)
                    if param.annotation is not param.empty
                    else ""
                )
                default_str = (
                    _format_default(param.default)
                    if param.default is not param.empty
                    else "*required*"
                )
                desc = param_docs.get(pname, "")
                lines.append(f"| `{pname}` | `{type_str}` | {default_str} | {desc} |")
            lines.append("")

    attrs = _get_class_attrs(cls)
    if attrs:
        field_docs = _extract_field_docs_from_mro(cls)
        lines.append("## Attributes")
        lines.append("")
        lines.extend(_render_field_table(attrs, field_docs, label="Attribute"))
        lines.append("")

    lines.extend(_methods_section(cls))
    lines.extend(_page_footer(entry))

    return "\n".join(lines)


def generate_index_page(manifest: list[dict]) -> str:
    """Generate the API reference index page."""
    lines = [
        *_page_heading(0, "Reference"),
    ]

    for group_key, group_info in sorted(GROUPS.items(), key=lambda x: x[1]["order"]):
        group_entries = [e for e in manifest if e["group"] == group_key]
        if not group_entries:
            continue

        lines.append(f"## {group_info['label']}")
        lines.append("")
        lines.append("| Class | Description |")
        lines.append("|-------|-------------|")

        for entry in group_entries:
            try:
                mod = importlib.import_module(entry["module"])
                cls = getattr(mod, entry["class_name"])
                doc = inspect.getdoc(cls) or ""
                first_line = doc.splitlines()[0] if doc else ""
            except Exception:
                first_line = ""

            slug = entry["class_name"].lower()
            link = f"/reference/{group_key}/{slug}"
            lines.append(
                f"| [`{entry['sidebar_label']}`]({link}) | {_rst_to_md(first_line)} |"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API reference pages.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for reference pages.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    errors = []
    generated = 0
    orders = _orders_within_group()

    for entry in API_REFERENCE_MANIFEST:
        try:
            mod = importlib.import_module(entry["module"])
            cls = getattr(mod, entry["class_name"])
        except (ImportError, AttributeError) as e:
            errors.append(f"ERROR: {entry['class_name']} ({entry['module']}): {e}")
            continue

        order = orders[entry["class_name"]]
        if entry["class_type"] == "config_data":
            content = generate_config_data_page(cls, entry, order)
        elif not isinstance(cls, type):
            doc = inspect.getdoc(cls) or ""
            sig_str = ""
            try:
                sig_str = f"`{entry['class_name']}{inspect.signature(cls)}`"
            except (ValueError, TypeError):
                pass
            title = entry.get("sidebar_label", entry["class_name"])
            lines = _page_heading(order, title)
            if sig_str:
                lines += [sig_str, ""]
            if doc:
                lines += [doc, ""]
            content = "\n".join(lines) + "\n"
        else:
            content = generate_behavior_page(cls, entry, order)

        group_dir = output_dir / entry["group"]
        group_dir.mkdir(parents=True, exist_ok=True)
        slug = entry["class_name"].lower()
        out_path = group_dir / f"{slug}.md"
        out_path.write_text(content)
        generated += 1
        print(
            f"  {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}"
        )

    index_content = generate_index_page(API_REFERENCE_MANIFEST)
    index_path = output_dir / "index.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index_content)
    print(
        f"  {index_path.relative_to(ROOT) if index_path.is_relative_to(ROOT) else index_path}"
    )

    print(f"\nGenerated {generated} reference pages + 1 index page")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
