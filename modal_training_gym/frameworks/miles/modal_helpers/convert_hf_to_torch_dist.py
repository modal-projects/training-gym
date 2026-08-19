"""Thin wrapper around miles' convert_hf_to_torch_dist.py for Modal volumes.

The upstream script's shutil.move(iter_0000001 -> release) only sees local shards,
which poisons the volume state for multi-node conversions (the rename propagates a
deletion of iter_0000001/ that wipes other nodes' committed shards).

Setting SKIP_RELEASE_RENAME=1 suppresses the rename so all nodes commit
to iter_0000001/ additively. Megatron loads from iter_0000001/ via the
tracker file just fine.

Pipeline-parallel auto-inflation is suppressed upstream via CONVERT_KEEP_PP1,
which the launcher sets when it pins PP/TP explicitly (slime has no such env
var, hence the source patch in its wrapper).

When SKIP_RELEASE_RENAME is unset this wrapper is a transparent pass-through.
"""

from __future__ import annotations

import os

_UPSTREAM = "/root/miles/tools/convert_hf_to_torch_dist.py"


def _load_upstream_source() -> str:
    with open(_UPSTREAM) as f:
        return f.read()


def main() -> None:
    src = _load_upstream_source()
    if os.environ.get("SKIP_RELEASE_RENAME"):
        src = src.replace(
            "shutil.move(source_dir, target_dir)",
            "pass  # SKIP_RELEASE_RENAME",
        )
        src = src.replace(
            'f.write("release")',
            'f.write("1")  # SKIP_RELEASE_RENAME: keep iter_0000001',
        )
    exec(
        compile(src, _UPSTREAM, "exec"), {"__name__": "__main__", "__file__": _UPSTREAM}
    )


if __name__ == "__main__":
    main()
