from modal_training_gym.deploy_recipes.vllm_recipe.recipe import VllmRecipe
from modal_training_gym.deploy_recipes.vllm_recipe.qwen3_0_6b import Qwen3_0_6b_VllmRecipe
from modal_training_gym.deploy_recipes.vllm_recipe.qwen3_4b import Qwen3_4b_VllmRecipe
from modal_training_gym.deploy_recipes.vllm_recipe.qwen3_30b import Qwen3_30b_VllmRecipe

__all__ = [
    "VllmRecipe",
    "Qwen3_0_6b_VllmRecipe",
    "Qwen3_4b_VllmRecipe",
    "Qwen3_30b_VllmRecipe",
]
