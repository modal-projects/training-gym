from pathlib import Path

import pytest

from scripts.tutorial_index import parse_tutorial


@pytest.mark.parametrize("order", ["+0", "-0", "1_0"])
def test_parse_tutorial_rejects_non_decimal_order(tmp_path: Path, order: str) -> None:
    tutorial = tmp_path / "example.py"
    tutorial.write_text(f"# ---\n# order: {order}\n# ---\n# # Example\n")

    with pytest.raises(
        ValueError,
        match="frontmatter requires a non-negative integer order",
    ):
        parse_tutorial(tutorial)
