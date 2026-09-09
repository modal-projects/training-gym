import pytest
from pydantic import ValidationError

from modal_training_gym.common.dataset import MultimodalDataset
from modal_training_gym.common.models import (
    Gemma4_26B_A4B,
    Inkling_Small,
    Qwen3_4B,
    Qwen3_5_4B,
    Qwen3_6_27B,
    Qwen3_ASR_1_7B,
    Qwen3_VL_8B,
)
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
    Gemma4_26B_A4B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.inkling import (
    Inkling_Small_LoRA_Recipe,
    Inkling_Small_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4B_Miles_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_5_4b import Qwen3_5_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_27b import Qwen3_6_27B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b import (
    Qwen3_ASR_1_7B_Recipe,
)
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


def test_train_config_rejects_video():
    with pytest.raises(ValidationError, match="cannot serve video"):
        TrainConfig(
            dataset=_mm("video"),
            model=Qwen3_VL_8B(),
            recipe=Qwen3_VL_8B_Recipe(),
        )


@pytest.mark.parametrize("modality", ["audio", "image"])
def test_train_config_rejects_media_on_text_model(modality):
    with pytest.raises(ValidationError, match=f"cannot serve {modality}"):
        TrainConfig(
            dataset=_mm(modality),
            model=Qwen3_4B(),
            recipe=Qwen3_4B_Recipe(),
        )


def test_train_config_accepts_qwen35_image():
    TrainConfig(
        dataset=_mm("image"),
        model=Qwen3_5_4B(),
        recipe=Qwen3_5_4B_Recipe(),
    )


def test_train_config_rejects_inkling_text_mode_with_audio():
    with pytest.raises(ValidationError, match="cannot serve audio"):
        TrainConfig(
            dataset=_mm("audio"),
            model=Inkling_Small(),
            recipe=Inkling_Small_Recipe(),
        )


def test_train_config_accepts_inkling_audio():
    TrainConfig(
        dataset=_mm("audio"),
        model=Inkling_Small(),
        recipe=Inkling_Small_Recipe(modality="audio"),
    )


def test_qwen35_image_cli_freezes_vision_and_keeps_thd():
    args = Qwen3_5_4B_Recipe().cli_args(dataset=_mm("image"), model=Qwen3_5_4B())
    flags = _flags(args)
    assert flags["--qkv-format"] == "thd"
    assert flags["--freeze-params-name-list"] == "vision_model"
    assert "--use-dynamic-batch-size" in args
    assert "--micro-batch-size" not in args


def test_qwen36_image_disables_eagle():
    recipe = Qwen3_6_27B_Recipe()
    image_args = recipe.cli_args(dataset=_mm("image"), model=Qwen3_6_27B())
    text_flags = _flags(recipe.cli_args(model=Qwen3_6_27B()))
    assert "--sglang-speculative-algorithm" not in image_args
    assert text_flags["--sglang-speculative-algorithm"] == "EAGLE"


def test_miles_image_enables_multimodal_without_slime_keys():
    args = Qwen3_5_4B_Miles_Recipe().cli_args(dataset=_mm("image"), model=Qwen3_5_4B())
    assert "--sglang-enable-multimodal" in args
    assert "--freeze-params-name-list" not in args


def test_vl_always_emits_bshd():
    flags = _flags(Qwen3_VL_8B_Recipe().cli_args(model=Qwen3_VL_8B()))
    assert flags["--qkv-format"] == "bshd"
    assert flags["--freeze-params-name-list"] == "vision_model"


def test_asr_audio_cli_uses_bshd():
    args = Qwen3_ASR_1_7B_Recipe().cli_args(
        dataset=_mm("audio"), model=Qwen3_ASR_1_7B()
    )
    flags = _flags(args)
    assert flags["--qkv-format"] == "bshd"
    assert "--use-dynamic-batch-size" not in args
    assert flags["--micro-batch-size"] == "1"


def test_gemma_image_cli_uses_bshd():
    recipe = Gemma4_26B_A4B_Recipe(modality="vision", rm_type="gemma_math")
    args = recipe.cli_args(dataset=_mm("image"), model=Gemma4_26B_A4B())
    flags = _flags(args)
    assert flags["--qkv-format"] == "bshd"
    assert "--use-dynamic-batch-size" not in args
    assert flags["--micro-batch-size"] == "1"


@pytest.mark.parametrize(
    ("modality", "recipe_modality"),
    [("image", "vision"), ("audio", "audio")],
)
def test_inkling_lora_media_keeps_thd_and_dynamic_packing(modality, recipe_modality):
    recipe = Inkling_Small_LoRA_Recipe(modality=recipe_modality)
    args = recipe.cli_args(dataset=_mm(modality), model=Inkling_Small())
    flags = _flags(args)
    assert flags["--qkv-format"] == "thd"
    assert "--use-dynamic-batch-size" in args
    assert "--micro-batch-size" not in args
