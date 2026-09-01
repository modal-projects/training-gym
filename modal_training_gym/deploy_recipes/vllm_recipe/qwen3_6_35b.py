from dataclasses import dataclass

from modal_training_gym.deploy_recipes.vllm_recipe.recipe import VllmRecipe


@dataclass
class Qwen3_6_35B_VllmRecipe(VllmRecipe):
    """Qwen3.6-35B-A3B (MoE) vLLM recipe."""

    pass
