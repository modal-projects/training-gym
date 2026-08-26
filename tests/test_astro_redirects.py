"""Astro meta-refresh map from dist/tarball, and nginx 302 locations."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = '<meta http-equiv="refresh" content="0;url=/guides/start/model/">'
PAGE = "<html><title>Start</title></html>"
TARGET = "/guides/start/model/"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _site(tmp_path: Path) -> tuple[Path, Path]:
    dist = tmp_path / "dist"
    (dist / "guides").mkdir(parents=True)
    (dist / "guides" / "index.html").write_text(STUB)
    page_dir = dist / "guides" / "start" / "model"
    page_dir.mkdir(parents=True)
    (page_dir / "index.html").write_text(PAGE)
    tarball = tmp_path / "docs.tar.gz"
    subprocess.check_call(["tar", "-czf", str(tarball), "-C", str(dist), "."])
    return dist, tarball


def test_refresh_map_and_nginx_slash_locations(tmp_path: Path) -> None:
    dist, tarball = _site(tmp_path)
    astro = _load("astro_redirects", ROOT / "docs-next" / "astro_redirects.py")
    expected = {"/guides": TARGET}
    assert astro.refresh_map(dist) == expected
    assert astro.refresh_map_from_tarball(tarball) == expected

    previews = _load(
        "frontend_previews",
        ROOT / "scripts" / "previews" / "frontend_previews.py",
    )
    snippet = previews._docs_nginx_refresh_inc(expected)
    assert f"location = /guides {{ return 302 {TARGET}$is_args$args; }}" in snippet
    assert f"location = /guides/ {{ return 302 {TARGET}$is_args$args; }}" in snippet
