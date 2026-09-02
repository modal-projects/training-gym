"""Map Astro meta-refresh index.html stubs to HTTP redirect targets."""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

_ASTRO_REDIRECT = re.compile(
    r'<meta http-equiv="refresh" content="\d+;url=([^"]+)">',
    re.IGNORECASE,
)

DIRECTORY_HOME_REDIRECTS = frozenset(
    {
        "/guides",
        "/reference",
        "/tutorials",
        "/tutorials/agent",
        "/tutorials/rl",
    }
)


def redirect_status(request_path: str) -> int:
    return 302 if request_path in DIRECTORY_HOME_REDIRECTS else 301


def _request_path(parent: Path) -> str:
    return "/" if parent == Path(".") else f"/{parent.as_posix()}"


def _target(html: str) -> str | None:
    match = _ASTRO_REDIRECT.search(html)
    return None if match is None else match.group(1)


def refresh_map(dist: Path) -> dict[str, str]:
    redirects: dict[str, str] = {}
    for index_path in dist.rglob("index.html"):
        target = _target(index_path.read_text())
        if target is None:
            continue
        redirects[_request_path(index_path.parent.relative_to(dist))] = target
    return redirects


def refresh_map_from_tarball(tarball: Path) -> dict[str, str]:
    redirects: dict[str, str] = {}
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            name = member.name
            if member.isfile() and Path(name).name == "index.html":
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                target = _target(extracted.read().decode())
                if target is None:
                    continue
                redirects[_request_path(Path(name).parent)] = target
    return redirects
