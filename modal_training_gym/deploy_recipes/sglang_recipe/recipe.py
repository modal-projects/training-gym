from __future__ import annotations

from dataclasses import dataclass

from modal_training_gym.common import GPUType
from modal_training_gym.deploy_recipes.base import BaseDeployRecipe, DeployRecipeType


@dataclass
class SglangRecipe(BaseDeployRecipe):
    """SGLang serving configuration.

    ## Fields

    gpu : GPUType | None
        GPU type for the serving container. Default ``"H100"``.
    tp : int | None
        Tensor parallelism degree. Default ``None`` (SGLang infers from GPU count).
    dp : int | None
        Data parallelism degree. Enables ``--enable-dp-attention`` when set.
        Default ``None``.
    context_length : int | None
        Maximum context length. Default ``None`` (model default).
    mem_fraction_static : float | None
        Fraction of GPU memory for KV cache. Default ``None`` (SGLang default).
    chunked_prefill_size : int | None
        Chunked prefill token budget. Default ``None``.
    max_running_requests : int | None
        Max concurrent requests per worker. Default ``None``.
    sglang_image : str
        Docker image tag for the SGLang container. Default is a recent nightly.
    extra_server_args : dict[str, str] | None
        Additional ``--flag value`` pairs passed to ``sglang.launch_server``.
        Use an empty string value for boolean flags (e.g. ``{"--trust-remote-code": ""}``).
        Default ``None``.
    environment_name : str | None
        Modal environment to deploy into. Default ``None``.
    deploy_strategy : str
        Modal deployment strategy. Default ``"rolling"``.
    """

    recipe_type: DeployRecipeType = DeployRecipeType.SGLANG

    gpu: GPUType = "H100"
    tp: int | None = None
    dp: int | None = None
    context_length: int | None = None
    mem_fraction_static: float | None = None
    chunked_prefill_size: int | None = None
    max_running_requests: int | None = None
    sglang_image: str = "lmsysorg/sglang:v0.5.12-cu130"
    extra_server_args: dict[str, str] | None = None
    environment_name: str | None = None
    deploy_strategy: str = "rolling"

    def server_args(self, *, served_model_name: str) -> dict[str, str]:
        """Build the ``--flag value`` dict for ``SGLangEndpoint``."""
        args: dict[str, str] = {"--served-model-name": served_model_name}
        if self.context_length is not None:
            args["--context-length"] = str(self.context_length)
        if self.mem_fraction_static is not None:
            args["--mem-fraction-static"] = str(self.mem_fraction_static)
        if self.chunked_prefill_size is not None:
            args["--chunked-prefill-size"] = str(self.chunked_prefill_size)
        if self.max_running_requests is not None:
            args["--max-running-requests"] = str(self.max_running_requests)
        if self.extra_server_args:
            args.update(self.extra_server_args)
        return args
