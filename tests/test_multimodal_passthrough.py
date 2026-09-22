"""Multimodal dataset write + TrainConfig/recipe CLI contracts for the modality port."""

import pytest
from pydantic import ValidationError

from modal_training_gym.common.dataset import MultimodalDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import (
    Gemma4_26B_A4B,
    Qwen3_4B,
    Qwen3_6_27B,
    Qwen3_VL_8B,
)
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
    Gemma4_26B_A4B_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_27b import Qwen3_6_27B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b import Qwen3_VL_8B_Recipe


def _mm(modality):
    return MultimodalDataset(
        rows=[{"prompt": "p", "media": ["ref"], "label": "l"}],
        modality=modality,
    )


def _flags(args):
    return {
        args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")
    }


def test_media_column_must_be_distinct():
    with pytest.raises(TrainingGymConfigError, match="media_column"):
        MultimodalDataset(rows=[], modality="image", media_column="prompt")


def test_train_config_rejects_modality_mismatch():
    with pytest.raises(ValidationError, match="cannot serve image"):
        TrainConfig(
            dataset=_mm("image"),
            model=Qwen3_4B(),
            recipe=Qwen3_4B_Recipe(),
        )


def test_qwen36_image_cli_keeps_torch_dist_ref_load():
    args = Qwen3_6_27B_Recipe().cli_args(dataset=_mm("image"), model=Qwen3_6_27B())
    assert "/checkpoints/Qwen3.6-27B_torch_dist_tp1pp1" in args
    assert "--megatron-to-hf-mode" not in args


def test_explicit_bridge_recipe():
    flags = _flags(
        Qwen3_VL_8B_Recipe().cli_args(dataset=_mm("image"), model=Qwen3_VL_8B())
    )
    assert flags["--megatron-to-hf-mode"] == "bridge"


def test_gemma_image_cli():
    args = Gemma4_26B_A4B_Recipe(rm_type="gemma_math").cli_args(
        dataset=_mm("image"), model=Gemma4_26B_A4B()
    )
    flags = _flags(args)
    assert flags["--qkv-format"] == "bshd"
    assert "--sglang-enable-multimodal" in args


def test_write_rejects_remote_media_urls(tmp_path):
    ds = MultimodalDataset(
        rows=[
            {
                "prompt": "p",
                "media": ["https://example.test/clip.wav"],
                "label": "l",
            }
        ],
        modality="audio",
    )
    with pytest.raises(TrainingGymConfigError, match=r"source_rows\(\)"):
        ds.write(str(tmp_path / "train.jsonl"))


def test_write_rejects_missing_media_paths(tmp_path):
    missing = tmp_path / "does-not-exist.png"
    ds = MultimodalDataset(
        rows=[{"prompt": "p", "media": [str(missing)], "label": "l"}],
        modality="image",
    )
    with pytest.raises(TrainingGymConfigError, match="not an existing file"):
        ds.write(str(tmp_path / "train.jsonl"))
