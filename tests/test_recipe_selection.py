import dataclasses
import importlib
import inspect
import pkgutil
from typing import Any

import pytest

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
    Gemma4_26B_A4B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4B_Miles_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_5_4b import Qwen3_5_4B_Recipe

_RECIPE_PACKAGES = (
    "modal_training_gym.train_recipes.slime_recipe",
    "modal_training_gym.train_recipes.miles_recipe",
)


def _dataset() -> HuggingFaceDataset:
    return HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
        input_format="text",
    )


def _config(recipe: SlimeRecipe) -> TrainConfig:
    return TrainConfig(dataset=_dataset(), model=Qwen3_4B(), recipe=recipe)


def _model_recipe_classes() -> list[type[SlimeRecipe | MilesRecipe]]:
    classes = []
    for package_name in _RECIPE_PACKAGES:
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(package.__path__, f"{package_name}."):
            if module_info.name.endswith(".recipe"):
                continue
            module = importlib.import_module(module_info.name)
            classes.extend(
                recipe_cls
                for _, recipe_cls in inspect.getmembers(module, inspect.isclass)
                if recipe_cls.__module__ == module.__name__
                and "Recipe" in recipe_cls.__name__
                and issubclass(recipe_cls, (SlimeRecipe, MilesRecipe))
            )
    return classes


@pytest.mark.parametrize(
    "recipe_cls",
    _model_recipe_classes(),
    ids=lambda recipe_cls: recipe_cls.__name__,
)
def test_model_recipes_only_override_framework_defaults(
    recipe_cls: type[SlimeRecipe | MilesRecipe],
) -> None:
    baseline_cls = recipe_cls.__mro__[1]  # mro[1] corresponds to the parent class
    defaults = {}
    for field in dataclasses.fields(baseline_cls):
        if field.default is dataclasses.MISSING:
            if field.default_factory is dataclasses.MISSING:
                continue
            defaults[field.name] = field.default_factory()
        else:
            defaults[field.name] = field.default
    recipe = recipe_cls()
    direct_fields = set(recipe_cls.__annotations__) & defaults.keys()

    redundant = {
        field for field in direct_fields if getattr(recipe, field) == defaults[field]
    }
    assert redundant == set()


def test_gemma_recipe_disables_unsupported_recompute_and_routing_replay() -> None:
    recipe = Gemma4_26B_A4B_Recipe()

    # Recompute rejects Gemma's tuple decoder output. Routing replay expects
    # num_experts_per_tok, which Gemma names top_k_experts.
    assert recipe.recompute_granularity is None
    assert recipe.recompute_method is None
    assert recipe.recompute_num_layers is None
    assert recipe.use_rollout_routing_replay is False


def test_slime_recipe_is_constructible_without_kwargs() -> None:
    recipe = SlimeRecipe()

    assert recipe.num_rollout == 1
    assert recipe.save_interval == recipe.num_rollout
    assert recipe.optimizer_cpu_offload is False
    assert recipe.overlap_cpu_optimizer_d2h_h2d is False
    assert recipe.use_precision_aware_optimizer is False


def test_generic_recipe_uses_framework_defaults_for_known_model() -> None:
    config = _config(SlimeRecipe())

    recipe = config._prepare_recipe()
    assert recipe.gpu_type == "H100"
    assert recipe.tensor_model_parallel_size == 1
    assert recipe.rollout_num_gpus_per_engine == 1
    assert recipe.colocate is True
    assert recipe.num_rollout == 1
    assert recipe.save_interval == recipe.num_rollout
    assert recipe.n_samples_per_prompt == 2
    assert recipe.rollout_batch_size == 2


def test_miles_recipe_uses_shared_sampling_defaults() -> None:
    recipe = MilesRecipe()

    assert recipe.n_samples_per_prompt == 2
    assert recipe.save_interval == recipe.num_rollout
    assert recipe.rollout_batch_size == 2


def test_model_recipe_uses_its_class_defaults() -> None:
    recipe = _config(Qwen3_4B_Recipe())._prepare_recipe()

    assert recipe.actor_num_gpus_per_node == 1
    assert recipe.max_tokens_per_gpu == 8192


def test_miles_4b_inherits_one_gpu_and_enables_cpu_offload() -> None:
    recipe = Qwen3_5_4B_Miles_Recipe()

    assert recipe.actor_num_gpus_per_node == 1
    assert recipe.optimizer_cpu_offload is True
    assert recipe.overlap_cpu_optimizer_d2h_h2d is True
    assert recipe.use_precision_aware_optimizer is True
    assert recipe.tensor_model_parallel_size == 1


def test_slime_35_4b_keeps_one_gpu_offload() -> None:
    recipe = Qwen3_5_4B_Recipe()

    assert recipe.actor_num_gpus_per_node == 1
    assert recipe.optimizer_cpu_offload is True
    assert recipe.sglang_mem_fraction_static == 0.7


def test_prepare_recipe_does_not_mutate_stored_launch_callables() -> None:
    def image_overlay(image: Any) -> Any:
        return image

    def custom_rm_function(*args: Any, **kwargs: Any) -> float:
        return 1.0

    recipe = SlimeRecipe(
        image_overlay=image_overlay,
        custom_rm_function=custom_rm_function,
    )
    config = _config(recipe)

    first = config._prepare_recipe()
    first.image_overlay = None
    first.custom_rm_function = None
    second = config._prepare_recipe()

    assert recipe.image_overlay is image_overlay
    assert recipe.custom_rm_function is custom_rm_function
    assert second.image_overlay is image_overlay
    assert second.custom_rm_function is custom_rm_function
