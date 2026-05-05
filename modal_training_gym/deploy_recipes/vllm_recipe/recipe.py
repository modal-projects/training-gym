from __future__ import annotations

from dataclasses import dataclass

from modal_training_gym.common import GPUType
from modal_training_gym.deploy_recipes.base import BaseDeployRecipe, DeployRecipeType


@dataclass
class VllmRecipe(BaseDeployRecipe):
    """vLLM serving configuration.

    ## Fields

    gpu : GPUType | None
        GPU type for the serving container. Default ``None`` (inferred).
    n_gpu : int | None
        Number of GPUs (tensor-parallel degree for vLLM). Default ``None``.
    extra_vllm_args : list[str] | None
        Additional CLI args passed to ``vllm serve``. Default ``None``.
    environment_name : str | None
        Modal environment to deploy into. Default ``None``.
    deploy_strategy : str
        Modal deployment strategy. Default ``"rolling"``.
    """

    recipe_type: DeployRecipeType = DeployRecipeType.VLLM
    gpu: GPUType | None = None
    n_gpu: int | None = None
    extra_vllm_args: list[str] | None = None
    environment_name: str | None = None
    deploy_strategy: str = "rolling"
