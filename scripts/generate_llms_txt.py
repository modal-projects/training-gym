"""Generate docs-next/public/llms.txt from tutorial + API catalogs.

uv run scripts/generate_llms_txt.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from api_reference_manifest import (
    API_REFERENCE_MANIFEST,
    CLASS_REFERENCE_PATHS,
    GROUPS,
)
from tutorial_index import TutorialEntry, load_tutorial_index

ROOT = Path(__file__).resolve().parents[1]
GUIDES_DIR = ROOT / "docs-next" / "src" / "content" / "docs" / "guides"
DEFAULT_OUTPUT = ROOT / "docs-next" / "public" / "llms.txt"

SITE = "https://gym.modal.dev"
REPO = "https://github.com/modal-projects/training-gym"


def _collect_guides() -> list[tuple[str, str, str, int]]:
    """Return (slug, title, description, sidebar order) for authored guides."""
    guides: list[tuple[str, str, str, int]] = []
    for path in sorted(GUIDES_DIR.rglob("*.md")):
        text = path.read_text()
        if not text.startswith("---\n"):
            raise ValueError(f"Guide is missing frontmatter: {path}")
        parts = text.split("---\n", 2)
        if len(parts) != 3:
            raise ValueError(f"Guide has invalid frontmatter: {path}")
        frontmatter = parts[1]

        try:
            metadata = yaml.safe_load(frontmatter)
        except yaml.YAMLError as exc:
            raise ValueError(f"Guide has invalid YAML frontmatter: {path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Guide frontmatter must be a mapping: {path}")

        title = metadata.get("title")
        if not isinstance(title, str) or not title:
            raise ValueError(f"Guide frontmatter is missing title: {path}")
        description = metadata.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"Guide description must be a string: {path}")

        sidebar = metadata.get("sidebar", {})
        if not isinstance(sidebar, dict):
            raise ValueError(f"Guide sidebar must be a mapping: {path}")
        sidebar_order = sidebar.get("order", 10_000)
        if type(sidebar_order) is not int:
            raise ValueError(f"Guide sidebar order must be an integer: {path}")

        slug = path.relative_to(GUIDES_DIR).with_suffix("").as_posix()
        if slug != "index":
            guides.append((slug, title, description, sidebar_order))

    guides.sort(key=lambda guide: (guide[3], guide[1].lower()))
    return guides


def _render(
    tutorials: tuple[TutorialEntry, ...],
    guides: list[tuple[str, str, str, int]],
) -> str:
    lines: list[str] = [
        "# Modal Training Gym",
        "",
        "> Python SDK for RL post-training on Modal. Compose a model, dataset, and",
        "> recipe (`SlimeRecipe` / Miles) via `TrainConfig`, then call `.train()` /",
        "> `.launch()` — cluster topology, Ray/NCCL, volumes, and checkpointing are handled.",
        "",
        "Requires Python 3.12. Install the package as `modal-training-gym`",
        "(import as `modal_training_gym`). Prefer `TrainConfig` + recipe over older",
        "framework-specific launcher APIs.",
        "",
        f"Docs: {SITE}/",
        f"Repo: {REPO}",
        "",
        "## Docs",
        "",
        f"- [Overview]({SITE}/): Product overview and getting started",
        f"- [Guides]({SITE}/guides/): Concepts and practical workflows",
        f"- [Tutorials]({SITE}/tutorials/): Runnable Python guides",
        f"- [API Reference]({SITE}/reference/): Public class reference",
        f"- [CLI Reference]({SITE}/reference/cli/): `modal-training-gym` CLI",
        f"- [Support]({SITE}/support/): Support and contribution notes",
        "",
        "## Guides",
        "",
    ]

    for slug, title, description, _ in guides:
        suffix = f": {description}" if description else ""
        lines.append(f"- [{title}]({SITE}/guides/{slug}/){suffix}")

    lines.extend(
        [
            "",
            "## Tutorials",
            "",
        ]
    )

    for tutorial in tutorials:
        lines.append(f"- [{tutorial.title}]({SITE}/tutorials/{tutorial.slug}/)")
    lines.append("")

    lines.extend(
        [
            "## API Reference",
            "",
            f"- [Overview]({SITE}/reference/): Index of public classes",
            f"- [CLI Reference]({SITE}/reference/cli/): CLI commands and flags",
            "",
        ]
    )

    by_group: dict[str, list[dict]] = defaultdict(list)
    for entry in API_REFERENCE_MANIFEST:
        by_group[entry["group"]].append(entry)

    for group_key, group_meta in sorted(
        GROUPS.items(), key=lambda item: item[1]["order"]
    ):
        entries = by_group.get(group_key)
        if not entries:
            continue
        lines.append(f"### {group_meta['label']}")
        lines.append("")
        for entry in entries:
            class_name = entry["class_name"]
            label = entry.get("sidebar_label") or class_name
            path = CLASS_REFERENCE_PATHS[class_name]
            lines.append(f"- [{label}]({SITE}{path}): `{class_name}`")
        lines.append("")

    lines.extend(
        [
            "## Optional",
            "",
            f"- [AGENTS.md]({REPO}/blob/main/AGENTS.md): Agent working rules for this repo",
            f"- [skills/]({REPO}/tree/main/skills): Packaged agent skills (model-support, etc.)",
            f"- [examples/quickstart.py]({REPO}/blob/main/examples/quickstart.py): Minimal TrainConfig example",
            "",
            "<!-- Generated by scripts/generate_llms_txt.py; do not edit by hand. -->",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate docs-next/public/llms.txt from documentation sources."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for llms.txt",
    )
    args = parser.parse_args()

    tutorials = load_tutorial_index()
    if not tutorials:
        raise SystemExit("No tutorials found")
    guides = _collect_guides()
    if not guides:
        raise SystemExit("No Markdown guides found")

    text = _render(tutorials, guides)
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)

    n_api = len(API_REFERENCE_MANIFEST)
    try:
        display_path = out_path.relative_to(ROOT)
    except ValueError:
        display_path = out_path
    print(f"Wrote {display_path}")
    print(f"  guides: {len(guides)}")
    print(f"  tutorials: {len(tutorials)}")
    print(f"  api classes: {n_api}")


if __name__ == "__main__":
    main()
