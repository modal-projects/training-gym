from __future__ import annotations

import pytest

from scripts.generate_all import GENERATORS
from scripts.generate_docs_pages import wrap_catalogs


def test_wrap_catalogs_wraps_models_table() -> None:
    src = (
        "intro\n"
        "<!-- BEGIN MODELS TABLE -->\n"
        "| Model |\n"
        "<!-- END MODELS TABLE -->\n"
        "outro\n"
    )
    out = wrap_catalogs(src)
    assert 'class="catalog models-catalog"' in out
    assert "--catalog-columns:" in out
    assert "minmax(0, 1.7fr)" in out
    assert out.index("<!-- BEGIN MODELS TABLE -->") > out.index("<div")
    assert out.index("</div>") > out.index("<!-- END MODELS TABLE -->")


def test_wrap_catalogs_wraps_tutorial_table() -> None:
    src = "<!-- BEGIN TUTORIAL TABLE -->\n| Tutorial |\n<!-- END TUTORIAL TABLE -->\n"
    out = wrap_catalogs(src)
    assert 'class="catalog tutorial-catalog"' in out
    assert "minmax(0, 1.6fr)" in out


def test_wrap_catalogs_leaves_unmarked_markdown_unchanged() -> None:
    src = "no catalogs here\n"
    assert wrap_catalogs(src) == src


def test_wrap_catalogs_rejects_one_sided_markers() -> None:
    with pytest.raises(ValueError, match="models-catalog"):
        wrap_catalogs("<!-- BEGIN MODELS TABLE -->\nno end\n")


def test_generate_all_runs_models_table_before_docs_pages() -> None:
    scripts = [cmd[-1] for cmd in GENERATORS]
    assert scripts.index("scripts/generate_models_table.py") < scripts.index(
        "scripts/generate_docs_pages.py"
    )
