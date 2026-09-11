from __future__ import annotations

from dataclasses import field
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

_MM_MODEL_PROVIDER = "miles_plugins.models.inkling.model.inkling_mm_model_provider"
_EPHEMERAL_DISK_MIB = 768 * 1024


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _InklingSmallRecipe(MilesRecipe):
    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | {"modality"}

    modality: Literal["text", "vision"] = "text"

    docker_image: str = "radixark/miles:dev-202608041247"
    gpu_type: str = "H200"
    memory: tuple[int, int] = (1024, int(2 * 1024 * 1024))

    miles_model_script: str = "scripts/models/inkling-small.sh"
    model_name: str = "inkling"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CONVERT_KEEP_PP1": "1",
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            "SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM": "false",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "MILES_SGLANG_DUMMY_LOAD": "0",
            "SGLANG_SERVER_ENGINE_ROLLOUT_RETURN_LOGPROB": "1",
            "RAY_memory_monitor_refresh_ms": "0",
            "NCCL_MNNVL_ENABLE": "1",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_RAS_ENABLE": "0",
        }
    )

    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Inkling-Small_torch_dist"
    conversion_tensor_model_parallel_size: int = 8
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    convert_ephemeral_disk_mb: int | None = 1024 * 1024

    actor_num_nodes: int = 4
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 8
    decoder_last_pipeline_num_layers: int = 7
    expert_model_parallel_size: int = 4

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    transformer_impl: str = "transformer_engine"
    bf16: bool = True
    no_bias_dropout_fusion: bool = True
    distributed_timeout_minutes: int = 30

    rm_type: str = "math"
    balance_data: bool = True
    rollout_num_gpus_per_engine: int = 16
    skip_eval_before_train: bool = True

    eps_clip_c: float = 3.0
    use_tis: bool = True
    use_rollout_routing_replay: bool = True

    use_distributed_optimizer: bool = True
    no_check_for_nan_in_loss_and_grad: bool = True

    sglang_attention_backend: str = "fa4"
    sglang_moe_runner_backend: str = "triton"
    sglang_mamba_scheduler_strategy: str = "extra_buffer"
    sglang_enable_multimodal: bool = True
    sglang_context_length: int = 4096
    sglang_disable_custom_all_reduce: bool = True

    custom_model_provider_path: str | None = None

    def _fields(
        self,
        dataset=None,
        eval_dataset=None,
        dataset_path=None,
        eval_dataset_path=None,
        model=None,
    ) -> dict[str, Any]:
        fields = super()._fields(
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            model=model,
        )
        # inkling-small.sh pins the *text* provider. Both providers are the same
        # function; the multimodal one just passes mm_towers=True, which calls
        # wire_mm_towers() to build the vision/audio towers and load them straight
        # from --hf-checkpoint (they never live in the torch_dist checkpoint, so no
        # re-conversion is needed). The data side needs no switch: miles selects
        # InklingTrainProcessor off the checkpoint's model_type and forwards its
        # patch tensors into forward() generically.
        if (
            self.modality == "vision"
            and not self.custom_model_provider_path
            and "custom_model_provider_path" not in self._escape_hatch_keys()
        ):
            fields["custom_model_provider_path"] = _MM_MODEL_PROVIDER
        return fields


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_Small_Recipe(_InklingSmallRecipe):
    """Inkling-Small full-parameter recipe."""

    lr: float = 5e-5

    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1

    offload_train: bool = True
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/tmp/train_offload"
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    sglang_mem_fraction_static: float = 0.65
    sglang_max_running_requests: int = 64
    sglang_max_total_tokens: int = 327680

    no_save_optim: bool = True
    no_load_optim: bool = True
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "Inkling_Small_Recipe":
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_Small_LoRA_Recipe(_InklingSmallRecipe):
    """Inkling-Small rank-32 LoRA recipe."""

    lr: float = 2e-4

    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    target_modules: str | None = "all-linear"
    experts_shared_outer_loras: bool = True
    sglang_lora_backend: str | None = "triton"
    sglang_lora_use_virtual_experts: bool = True
    sglang_max_loras_per_batch: int = 1
    sglang_max_lora_rank: int = 32

    max_tokens_per_gpu: int = 4096

    sglang_ep_size: int = 16
    sglang_mem_fraction_static: float = 0.65
    sglang_max_running_requests: int = 32
    sglang_max_total_tokens: int = 320000
    sglang_cuda_graph_max_bs: int = 64
    sglang_max_mamba_cache_size: int = 256
    no_offload_rollout: bool = True
    no_offload_train: bool = True
