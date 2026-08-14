"""Workspace snapshotter — bounded inline file tree per schema WS_* limits."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .. import schema

EXCLUDE_DIRS = set(schema.WS_EXCLUDE_DIRS) | {".obs"}
SAFE_DOTENV_NAMES = {".env.example", ".env.sample", ".env.template"}


def _is_sensitive_name(name: str) -> bool:
    """Keep dotenv credentials out of the public workspace artifact.

    Template files are useful documentation and contain placeholders; runtime
    dotenv files may contain API keys and must never reach workspace.json.
    """
    lower = name.lower()
    return (lower == ".env" or lower == ".envrc" or lower.startswith(".env.")) \
        and lower not in SAFE_DOTENV_NAMES


def _is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\x00" in f.read(8192)


def snapshot(ws_root) -> dict:
    ws_root = Path(ws_root).resolve()
    files: list[dict] = []
    total_bytes = 0
    inline_left = schema.WS_TOTAL_INLINE_MAX_BYTES

    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ws_root, followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames if d not in EXCLUDE_DIRS and not _is_sensitive_name(d)
        )
        for fn in sorted(filenames):
            if _is_sensitive_name(fn):
                continue
            paths.append(Path(dirpath) / fn)

    for p in sorted(paths):
        try:
            if not p.is_file() or p.is_symlink():
                continue
            size = p.stat().st_size
        except OSError:
            continue
        total_bytes += size
        entry = {"path": p.relative_to(ws_root).as_posix(), "size": size,
                 "inline": False, "content": None, "truncated": False}
        if size <= schema.WS_INLINE_MAX_BYTES and size <= inline_left:
            try:
                if not _is_binary(p):
                    entry["content"] = p.read_text(errors="replace")
                    entry["inline"] = True
                    inline_left -= size
            except OSError:
                pass  # read error -> stays inline=false
        files.append(entry)

    return {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(ws_root),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "inlined_files": sum(1 for f in files if f["inline"]),
        "files": files,
    }
