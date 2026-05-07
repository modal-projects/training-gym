from abc import ABC
from dataclasses import field
from enum import Enum


class DeployRecipeType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


class BaseDeployRecipe(ABC):
    recipe_type: DeployRecipeType
    # TODO: add as env variables at serve time
    env_vars: dict[str, str] = field(default_factory=dict)

