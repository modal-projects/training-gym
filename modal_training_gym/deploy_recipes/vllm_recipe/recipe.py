from __future__ import annotations

from dataclasses import dataclass

from modal_training_gym.common import GPUType
from modal_training_gym.deploy_recipes.base import BaseDeployRecipe, DeployRecipeType


@dataclass
class VllmRecipe(BaseDeployRecipe):
    """vLLM server settings.

    Args:
        gpu:
            GPU type for server containers.
        n_gpu:
            Number of GPUs and tensor-parallel degree.
        extra_vllm_args:
            Additional arguments for ``vllm serve``.
        environment_name:
            Modal deployment environment.
        deploy_strategy:
            Modal deployment strategy.
        startup_timeout:
            Maximum time in seconds for container startup.
    """

    recipe_type: DeployRecipeType = DeployRecipeType.VLLM
    gpu: GPUType | None = None
    n_gpu: int | None = None
    extra_vllm_args: list[str] | None = None
    environment_name: str | None = None
    deploy_strategy: str = "rolling"
    startup_timeout: int = 20 * 60
