from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe
from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
    Gemma4_26B_A4B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.inkling import (
    Inkling_Small_LoRA_Recipe,
    Inkling_Small_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.moonlight_16b_a3b import (
    Moonlight_16B_A3B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.nemotron3_ultra_550b_a55b import (
    Nemotron3_Ultra_550B_A55B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4B_Miles_Recipe,
)

__all__ = [
    "MilesRecipe",
    "Gemma4_26B_A4B_Recipe",
    "Inkling_Small_Recipe",
    "Inkling_Small_LoRA_Recipe",
    "Moonlight_16B_A3B_Recipe",
    "Nemotron3_Ultra_550B_A55B_Recipe",
    "Qwen3_5_4B_Miles_Recipe",
]
