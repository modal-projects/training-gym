from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATCHER = (
    Path(__file__).parents[1]
    / "modal_training_gym"
    / "frameworks"
    / "slime"
    / "modal_helpers"
    / "patches"
    / "model_specific_patches"
    / "qwen3_5_vl"
    / "patch_qwen3_5_hf_to_megatron.py"
)


@pytest.fixture(scope="session")
def patcher():
    spec = importlib.util.spec_from_file_location(
        "patch_qwen3_5_hf_to_megatron", PATCHER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_moved_anchor_raises(patcher, tmp_path):
    work = tmp_path / "checkpoint.py"
    work.write_text(
        patcher.CHECKPOINT_OLD.replace(
            "    from megatron.bridge import AutoBridge\n", "", 1
        )
    )
    with pytest.raises(RuntimeError, match="anchor matched 0 times"):
        patcher._patch_file(work, patcher.EDITS[patcher.CHECKPOINT])
