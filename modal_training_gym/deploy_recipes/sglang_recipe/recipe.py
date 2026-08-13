from __future__ import annotations

from dataclasses import dataclass, field

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
        Data parallelism degree. Emitted as ``--dp-size`` and enables
        ``--enable-dp-attention`` when set. Default ``None``.
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
    env_vars : dict[str, str]
        Extra environment variables baked into the serving image (e.g. DeepGEMM
        MegaMoE knobs). Merged on top of the base HF cache env. Default empty.
    install_transformers_from_git : bool
        If ``True`` (default), the serve image ``pip install``s transformers
        from GitHub so brand-new architectures (historically DeepSeek-V4 on
        older SGLang tags) are recognized by ``AutoConfig``. Set ``False``
        when the chosen ``sglang_image`` already ships a compatible
        transformers — otherwise the git install double-registers configs
        (e.g. ``qwen3_asr``) and the server crashloops on import.
    environment_name : str | None
        Modal environment to deploy into. Default ``None``.
    deploy_strategy : str
        Modal deployment strategy. Default ``"rolling"``.
    startup_timeout : int
        Seconds the server container is allowed to spend in startup before
        Modal kills it — gates both Modal's container ``startup_timeout`` and
        the SGLang health-check poll. Bump this for very large models whose
        weight load exceeds the default (e.g. GLM-4.7 at 355B, Kimi-K2.5 at
        ~1T). Default ``1200`` (20 minutes).
    """

    recipe_type: DeployRecipeType = DeployRecipeType.SGLANG

    gpu: GPUType = "H100"
    tp: int | None = None
    dp: int | None = None
    context_length: int | None = None
    mem_fraction_static: float | None = None
    chunked_prefill_size: int | None = None
    max_running_requests: int | None = None
    sglang_image: str = "lmsysorg/sglang:v0.5.12"
    extra_server_args: dict[str, str] | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    install_transformers_from_git: bool = True
    environment_name: str | None = None
    deploy_strategy: str = "rolling"
    startup_timeout: int = 20 * 60

    def server_args(self, *, served_model_name: str) -> dict[str, str]:
        """Build the ``--flag value`` dict for the SGLang launch command."""
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
