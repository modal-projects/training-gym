from pathlib import Path
import sys

import pytest

from scripts.generate_docs_pages import generate_starlight
from scripts.generate_llms_txt import _collect_guides
from scripts.tutorial_index import parse_tutorial

ROOT = Path(__file__).resolve().parents[1]
AUTHORED_PAGES = (
    ROOT / "docs-next/src/content/docs/guides/tools/observability-dashboard.md",
    ROOT / "docs-next/src/content/docs/guides/tools/trackio-integration.md",
    ROOT / "docs-next/src/content/docs/guides/tools/wandb-integration.md",
    ROOT / "docs-next/src/content/docs/reference/cli.md",
)


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
    for path in AUTHORED_PAGES:
        text = path.read_text()
        lines = _frontmatter_lines(text)
        assert len(lines) == 1, path
        assert lines[0].startswith("order: ")
        body = text.split("---\n", 2)[2]
        assert any(line.startswith("# ") for line in body.splitlines()), path


def test_collect_guides_uses_h1_and_order() -> None:
    guides = _collect_guides()
    assert [guide[0] for guide in guides] == [
        "tools/agent-driven-training",
        "tools/observability-dashboard",
        "tools/wandb-integration",
        "tools/trackio-integration",
    ]
    assert guides[0][1] == "Agent-driven training"
    assert guides[0][2] == 0
    assert guides[1][1] == "The observability dashboard"
    assert guides[1][2] == 1
    assert guides[2][1] == "Weights & Biases integration"
    assert guides[2][2] == 2
    assert guides[3][1] == "Trackio integration"
    assert guides[3][2] == 3


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
