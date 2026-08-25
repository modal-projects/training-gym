"""Generate docs-next/public/llms.txt from tutorial + API catalogs.

uv run python -m scripts.generate_llms_txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from scripts.api_reference_manifest import (
    API_REFERENCE_MANIFEST,
    CLASS_REFERENCE_PATHS,
    GROUPS,
)
from scripts.tutorial_index import TutorialEntry, load_tutorial_index

ROOT = Path(__file__).resolve().parents[1]
GUIDES_DIR = ROOT / "docs-next" / "src" / "content" / "docs" / "guides"
DEFAULT_OUTPUT = ROOT / "docs-next" / "public" / "llms.txt"

SITE = "https://gym.modal.dev"
REPO = "https://github.com/modal-projects/training-gym"


def _site_url(*parts: str) -> str:
    path = "/".join(part.strip("/") for part in parts if part)
    return f"{SITE}/{path}" if path else f"{SITE}/"


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _collect_guides() -> list[tuple[str, str, int]]:
    """Return (slug, title, order) for authored guides."""
    guides: list[tuple[str, str, int]] = []
    for path in sorted(GUIDES_DIR.rglob("*.md")):
        text = path.read_text()
        if not text.startswith("---\n"):
            raise ValueError(f"Guide is missing frontmatter: {path}")
        parts = text.split("---\n", 2)
        if len(parts) != 3:
            raise ValueError(f"Guide has invalid frontmatter: {path}")
        frontmatter = parts[1]
        body = parts[2]

        try:
            metadata = yaml.safe_load(frontmatter)
        except yaml.YAMLError as exc:
            raise ValueError(f"Guide has invalid YAML frontmatter: {path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Guide frontmatter must be a mapping: {path}")

        order = metadata.get("order")
        if type(order) is not int:
            raise ValueError(f"Guide frontmatter requires an integer order: {path}")
        title = _first_heading(body)
        if not title:
            raise ValueError(f"Guide is missing an H1 heading: {path}")

        slug = path.relative_to(GUIDES_DIR).with_suffix("").as_posix()
        if slug != "index":
            guides.append((slug, title, order))

    guides.sort(key=lambda guide: (guide[2], guide[1].lower()))
    return guides


def _render(
    tutorials: tuple[TutorialEntry, ...],
    guides: list[tuple[str, str, int]],
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
        f"Docs: {_site_url()}",
        f"Repo: {REPO}",
        "",
        "## Docs",
        "",
        f"- [Home]({_site_url()}): Docs home",
        f"- [Guides]({_site_url('guides')}): Concepts and practical workflows",
        f"- [Tutorials]({_site_url('tutorials')}): Runnable Python guides",
        f"- [Reference]({_site_url('reference')}): Public class reference",
        f"- [CLI Reference]({_site_url('reference/cli')}): `modal-training-gym` CLI",
        "",
        "## Guides",
        "",
    ]

    for slug, title, _order in guides:
        lines.append(f"- [{title}]({_site_url('guides', slug)})")

    lines.extend(
        [
            "",
            "## Tutorials",
            "",
        ]
    )

    for tutorial in tutorials:
        lines.append(f"- [{tutorial.title}]({_site_url('tutorials', tutorial.slug)})")
    lines.append("")

    lines.extend(
        [
            "## Reference",
            "",
            f"- [Reference]({_site_url('reference')}): Index of public classes",
            f"- [CLI Reference]({_site_url('reference/cli')}): CLI commands and flags",
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
            lines.append(f"- [{label}]({_site_url(path)}): `{class_name}`")
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
