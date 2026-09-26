from __future__ import annotations

import warnings

import pytest

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.memory_estimate import _arch_from_hf, maybe_warn_gpu_oom
from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
from modal_training_gym.common.models.validation import VALIDATION_CONFIGS
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


@pytest.mark.parametrize(
    "entry",
    [e for e in VALIDATION_CONFIGS if e.model_config.architecture is not None],
    ids=lambda e: e.name,
)
def test_presets_do_not_warn(entry) -> None:
    model = entry.model_config()
    recipe = (
        SlimeRecipe if entry.framework is Framework.SLIME else MilesRecipe
    ).get_base_recipe(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    assert not [w for w in caught if "Estimated peak" in str(w.message)]


def test_known_mistake_warns() -> None:
    model, recipe = Qwen3_4B(), SlimeRecipe.get_base_recipe(Qwen3_4B())
    recipe.max_tokens_per_gpu = 65536
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maybe_warn_gpu_oom(recipe, model)
    msg = next(str(w.message) for w in caught if "Estimated peak" in str(w.message))
    assert "max_tokens_per_gpu=65536" in msg
    assert "recompute_granularity" not in msg
    assert "tensor_model_parallel_size" not in msg


def test_invalid_model_name_without_arch_does_not_raise() -> None:
    model = Qwen3_4B()
    model.architecture, model.model_name = None, ""
    maybe_warn_gpu_oom(SlimeRecipe.get_base_recipe(Qwen3_4B()), model)


def test_hf_cfg_detects_moe(tmp_path, monkeypatch) -> None:
    p = tmp_path / "config.json"
    p.write_text(
        '{"num_hidden_layers":4,"hidden_size":256,"num_attention_heads":4,'
        '"intermediate_size":512,"num_experts":8,"moe_intermediate_size":128,'
        '"vocab_size":1000}'
    )
    monkeypatch.setattr(
        "modal_training_gym.common.memory_estimate.hf_hub_download",
        lambda **_: str(p),
    )
    assert _arch_from_hf("org/moe").num_experts == 8
