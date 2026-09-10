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

CONVERT_DEQUANT_HF_WEIGHTS=1 dequantizes DeepSeek block-scaled fp8/fp4 weights
as mbridge reads them (see hf_block_dequant), for checkpoints that ship without
a bf16 export.

When neither variable is set this wrapper is a transparent pass-through.
"""

from __future__ import annotations

import os
import sys

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
    if os.environ.get("CONVERT_DEQUANT_HF_WEIGHTS"):
        # torchrun runs this file as a script, so its directory heads sys.path
        # while the package itself may not be importable under PYTHONPATH.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hf_block_dequant

        hf_block_dequant.install()
    exec(
        compile(src, _UPSTREAM, "exec"), {"__name__": "__main__", "__file__": _UPSTREAM}
    )


if __name__ == "__main__":
    main()
