from abc import ABC
from enum import Enum


class RecipeType(Enum):
    SLIME = "slime"


class BaseTrainRecipe(ABC):
    recipe_type: RecipeType
