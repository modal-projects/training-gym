from pathlib import Path
import sys

import pytest
import yaml

from scripts.api_reference_manifest import CLASS_REFERENCE_PATHS
from scripts.generate_docs_pages import generate_starlight
from scripts.generate_llms_txt import (
    GUIDES_DIR,
    README,
    _collect_guides,
    _guide_section,
    _readme_heading_and_intro,
    _render,
    flatten_doc_id,
)
from scripts.tutorial_index import parse_tutorial

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs-next/src/content/docs"
GUIDE_PAGES = tuple(
    path for path in sorted(GUIDES_DIR.rglob("*.md")) if path.stem != "index"
)
CLI_PAGES = tuple(sorted(DOCS_DIR.glob("reference/cli.md")))
AUTHORED_PAGES = GUIDE_PAGES + CLI_PAGES


@pytest.mark.parametrize("order", ["+0", "-0", "1_0"])
def test_parse_tutorial_rejects_non_decimal_order(tmp_path: Path, order: str) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text(f"# ---\n# order: {order}\n# ---\n# # Example\n")

    with pytest.raises(
        ValueError,
        match="frontmatter requires a non-negative integer order",
    ):
        parse_tutorial(tutorial)


def _frontmatter_lines(text: str) -> list[str]:
    assert text.startswith("---\n"), text[:40]
    parts = text.split("---\n", 2)
    assert len(parts) == 3
    return [line for line in parts[1].splitlines() if line.strip()]


def test_homepage_frontmatter_is_order_only(tmp_path: Path) -> None:
    generate_starlight(tmp_path)
    text = (tmp_path / "index.md").read_text()
    assert _frontmatter_lines(text) == ["order: 0"]
    assert "\n# Training Gym\n" in text


def test_authored_pages_use_order_and_h1() -> None:
    assert GUIDE_PAGES, f"no guide pages under {GUIDES_DIR}"
    assert CLI_PAGES, f"missing {DOCS_DIR / 'reference/cli.md'}"
    for path in AUTHORED_PAGES:
        text = path.read_text()
        lines = _frontmatter_lines(text)
        assert len(lines) == 1, path
        assert lines[0].startswith("order: ")
        body = text.split("---\n", 2)[2]
        assert any(line.startswith("# ") for line in body.splitlines()), path


def test_collect_guides_orders_by_section_then_order() -> None:
    assert GUIDE_PAGES, f"no guide pages under {GUIDES_DIR}"
    orders_by_section: dict[str, list[int]] = {}
    expected: list[tuple[str, int, str, str]] = []

    for path in GUIDE_PAGES:
        text = path.read_text()
        parts = text.split("---\n", 2)
        metadata = yaml.safe_load(parts[1])
        assert isinstance(metadata, dict), path
        order = metadata.get("order")
        assert type(order) is int, path
        title = next(
            line[2:].strip() for line in parts[2].splitlines() if line.startswith("# ")
        )
        slug = path.relative_to(GUIDES_DIR).with_suffix("").as_posix()
        section = _guide_section(slug)
        orders_by_section.setdefault(section, []).append(order)
        expected.append((section, order, title.lower(), slug))

    for section, orders in orders_by_section.items():
        assert len(orders) == len(set(orders)), section

    expected.sort()
    guides = _collect_guides()
    assert [slug for slug, _, _ in guides] == [slug for _, _, _, slug in expected]


def test_readme_heading_and_intro_skips_h1_and_badges() -> None:
    markdown = (
        "# Training Gym\n"
        "\n"
        "[![ci](https://img.shields.io/badge/ci-ok)](https://example.com)\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "Second paragraph.\n"
        "\n"
        "## Quickstart\n"
    )
    assert _readme_heading_and_intro(markdown) == (
        "Training Gym",
        "First paragraph.\n\nSecond paragraph.",
    )


def test_render_groups_guides_by_section() -> None:
    text = _render(
        (),
        [
            ("start/model", "Model", 0),
            ("start/dataset", "Dataset", 1),
            ("tools/wandb-integration", "Weights & Biases integration", 2),
        ],
    )
    heading, intro = _readme_heading_and_intro(README.read_text())
    assert text.startswith(f"# {heading}\n")
    start = text.index("### Start")
    tools = text.index("### Tools")
    assert start < tools
    assert text.index("[Model]", start) < text.index("[Dataset]", start) < tools
    assert intro in text
    assert "https://gym.modal.dev/guides/model)" in text
    assert "https://gym.modal.dev/guides/dataset)" in text
    assert "https://gym.modal.dev/guides/wandb-integration)" in text
    assert "/guides/start/" not in text
    assert "/guides/tools/" not in text
    for line in text.splitlines():
        if line.startswith("- ["):
            assert "): " not in line


def test_flatten_doc_id_keeps_section_drops_grouping_folders() -> None:
    assert flatten_doc_id("guides/tools/dashboard.md") == "guides/dashboard"
    assert flatten_doc_id("guides/start/model.md") == "guides/model"
    assert flatten_doc_id("guides/tools/agent.md") == "guides/agent"
    assert flatten_doc_id("reference/core/trainconfig.md") == "reference/trainconfig"
    assert flatten_doc_id("reference/cli.md") == "reference/cli"
    assert flatten_doc_id("tutorials/rl_basics.md") == "tutorials/rl_basics"
    assert flatten_doc_id("reference/index.md") == "reference"
    assert flatten_doc_id("index.md") == "index"


def test_agent_guide_stem_is_agent() -> None:
    assert (GUIDES_DIR / "tools" / "agent.md").is_file()
    assert not (GUIDES_DIR / "tools" / "agent-driven-training.md").exists()


def test_class_reference_paths_are_section_plus_stem() -> None:
    for path in CLASS_REFERENCE_PATHS.values():
        parts = path.strip("/").split("/")
        assert parts[0] == "reference"
        assert len(parts) == 2


def test_api_reference_orders_follow_manifest() -> None:
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_api_reference import _orders_within_group, _page_heading

    assert _page_heading(0, "Reference") == [
        "---",
        "order: 0",
        "---",
        "",
        "# Reference",
        "",
    ]
    orders = _orders_within_group()
    assert orders["ModelConfig"] == 0
    assert orders["HFModelConfiguration"] == 1
    assert orders["ToolCall"] == 0
    assert orders["TrainConfig"] == 0
    assert orders["Endpoint"] == 0


def test_mobile_sidebar_does_not_accent_every_row() -> None:
    css = (ROOT / "docs-next/src/styles/custom.css").read_text()
    assert "li:has(.active) > :is(" not in css
