from modal_training_gym.deploy_recipes.base import BaseDeployRecipe, DeployRecipeType
from modal_training_gym.deploy_recipes.vllm_recipe import (
    VllmRecipe,
    Qwen3_0_6b_VllmRecipe,
    Qwen3_1_7b_VllmRecipe,
    Qwen3_4b_VllmRecipe,
    Qwen3_8b_VllmRecipe,
    Qwen3_14b_VllmRecipe,
    Qwen3_30b_VllmRecipe,
    Qwen3_32b_VllmRecipe,
)
from modal_training_gym.deploy_recipes.sglang_recipe import (
    SglangRecipe,
    DeepSeek_V4_Flash_SglangRecipe,
    Qwen3_0_6b_SglangRecipe,
    Qwen3_1_7b_SglangRecipe,
    Qwen3_4b_SglangRecipe,
    Qwen3_8b_SglangRecipe,
    Qwen3_14b_SglangRecipe,
    Qwen3_30b_SglangRecipe,
    Qwen3_32b_SglangRecipe,
)

__all__ = [
    "BaseDeployRecipe",
    "DeepSeek_V4_Flash_SglangRecipe",
    "DeployRecipeType",
    "Qwen3_0_6b_SglangRecipe",
    "Qwen3_0_6b_VllmRecipe",
    "Qwen3_1_7b_SglangRecipe",
    "Qwen3_1_7b_VllmRecipe",
    "Qwen3_4b_SglangRecipe",
    "Qwen3_4b_VllmRecipe",
    "Qwen3_8b_SglangRecipe",
    "Qwen3_8b_VllmRecipe",
    "Qwen3_14b_SglangRecipe",
    "Qwen3_14b_VllmRecipe",
    "Qwen3_30b_SglangRecipe",
    "Qwen3_30b_VllmRecipe",
    "Qwen3_32b_SglangRecipe",
    "Qwen3_32b_VllmRecipe",
    "SglangRecipe",
    "VllmRecipe",
]
