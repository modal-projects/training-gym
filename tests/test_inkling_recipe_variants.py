"""Inkling-Small ships two recipe variants behind one ``model_name``.

``TrainConfig`` defaults ``merge_model_recipe=True``, so ``_resolve_recipe`` merges
whatever ``get_base_recipe`` returns onto the caller's recipe and rebuilds it as
``type(base)``. If the lookup keyed only off ``model_name`` it would hand a LoRA
caller the full-parameter preset, silently dropping the LoRA-only fields and
force-applying full-parameter offload settings.
"""

from modal_training_gym import (
    Inkling_Small,
    Inkling_Small_LoRA_Recipe,
    Inkling_Small_Recipe,
)
from modal_training_gym.common.train import _resolve_recipe
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def test_lora_caller_keeps_the_lora_preset() -> None:
    assert (
        type(Inkling_Small_LoRA_Recipe.get_base_recipe(Inkling_Small()))
        is Inkling_Small_LoRA_Recipe
    )


def test_full_parameter_caller_keeps_the_full_parameter_preset() -> None:
    assert (
        type(Inkling_Small_Recipe.get_base_recipe(Inkling_Small()))
        is Inkling_Small_Recipe
    )


def test_bare_miles_recipe_defaults_to_the_full_parameter_preset() -> None:
    assert type(MilesRecipe.get_base_recipe(Inkling_Small())) is Inkling_Small_Recipe


def test_resolving_a_lora_recipe_preserves_lora_only_fields() -> None:
    model = Inkling_Small()
    lora = Inkling_Small_LoRA_Recipe()
    resolved = _resolve_recipe(model, lora, merge_model_recipe=True)

    assert type(resolved) is Inkling_Small_LoRA_Recipe
    assert resolved.lora_rank == lora.lora_rank
    assert resolved.sglang_max_loras_per_batch == lora.sglang_max_loras_per_batch
    assert resolved.sglang_max_lora_rank == lora.sglang_max_lora_rank
    assert resolved.no_offload_rollout is True
    assert resolved.no_offload_train is True


def test_resolving_a_lora_recipe_does_not_inherit_full_parameter_offload() -> None:
    full = Inkling_Small_Recipe()
    resolved = _resolve_recipe(
        Inkling_Small(), Inkling_Small_LoRA_Recipe(), merge_model_recipe=True
    )

    assert full.optimizer_cpu_offload is True
    assert full.no_save_optim is True
    assert resolved.optimizer_cpu_offload is not True
    assert resolved.no_save_optim is not True
    assert resolved.use_dynamic_batch_size is True


def test_resolving_a_full_parameter_recipe_still_gets_full_parameter_settings() -> None:
    resolved = _resolve_recipe(
        Inkling_Small(), Inkling_Small_Recipe(), merge_model_recipe=True
    )

    assert type(resolved) is Inkling_Small_Recipe
    assert resolved.optimizer_cpu_offload is True
    assert resolved.no_save_optim is True
    assert resolved.no_load_optim is True
    assert resolved.use_dynamic_batch_size is False
