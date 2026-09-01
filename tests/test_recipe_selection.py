from typing import Any

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4b_Recipe


_SLIME_RECIPE_KW = {
    "gpu_type": "H100",
    "colocate": True,
    "tensor_model_parallel_size": 1,
    "sequence_parallel": False,
    "rollout_num_gpus_per_engine": 1,
    "num_rollout": 1,
    "rollout_batch_size": 16,
    "rollout_max_response_len": 4096,
    "rollout_temperature": 1.0,
    "save_interval": 10,
}


def _dataset() -> HuggingFaceDataset:
    return HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
        apply_chat_template=True,
    )


def _config(recipe: SlimeRecipe) -> TrainConfig:
    return TrainConfig(dataset=_dataset(), model=Qwen3_4B(), recipe=recipe)


def test_generic_recipe_uses_framework_defaults_for_known_model() -> None:
    config = _config(SlimeRecipe(**_SLIME_RECIPE_KW))

    assert config._prepare_recipe().n_samples_per_prompt == 2


def test_model_recipe_uses_its_class_defaults() -> None:
    config = _config(Qwen3_4b_Recipe())

    assert config._prepare_recipe().n_samples_per_prompt == 8


def test_prepare_recipe_does_not_mutate_stored_launch_callables() -> None:
    def image_overlay(image: Any) -> Any:
        return image

    def custom_rm_function(*args: Any, **kwargs: Any) -> float:
        return 1.0

    recipe = SlimeRecipe(
        **_SLIME_RECIPE_KW,
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
