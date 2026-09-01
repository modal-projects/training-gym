"""Miles recipes for the Inkling family.

Both recipes mirror the upstream-validated Inkling-Small profile from
``scripts/run_inkling.py`` + ``docs/models/thinkingmachines/inkling-small.md``:
4 nodes x 8 H200 colocated, TP4 / SP / PP8 / EP4 / ETP1 (DP1), one 16-GPU SGLang
engine per 2 nodes, GRPO with truncated importance sampling and Rollout Routing
Replay.

``Inkling_Small_Recipe`` is full-parameter, ``Inkling_Small_LoRA_Recipe`` is
rank-32 all-linear LoRA; upstream validates both on this same cluster and gates
both in CI on a 4-layer slice. They differ on more than the adapter: full-param
pins a fixed micro-batch and offloads the optimizer, LoRA uses dynamic token
packing and keeps everything resident, syncing only the adapter (~4 s vs ~50 s
per rollout).

See ``.gym/new_models/Inkling_Small/model_setup.md`` for the derivation, the
deliberate deviations from ``run_inkling.py``, and expected step timings.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

# Same function as the text provider with mm_towers=True; see _fields below.
_MM_MODEL_PROVIDER = "miles_plugins.models.inkling.model.inkling_mm_model_provider"
_EPHEMERAL_DISK_MIB = 768 * 1024


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _InklingSmallRecipe(MilesRecipe):
    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | {"modality"}

    # Selects the model provider below, the same way Gemma4_26B_A4B_Recipe's flag picks
    # its mode. A vision run also needs a MultimodalDataset with apply_chat_template
    # off and its images materialized as files.
    modality: Literal["text", "vision"] = "text"

    # Inkling landed upstream on 2026-08-03 (miles 5c517599 / 92ccb87d), well after
    # MilesRecipe's default image was built, so this recipe pins its own. Do not use
    # the `radixark/miles:inkling` tag from the upstream docs: it is arm64-only
    # (GB300) and will not run on Modal H100/H200.
    docker_image: str = "radixark/miles:dev-202608041247"
    gpu_type: str = "H200"
    # Full-param offloads the optimizer to host RAM; LoRA inherits the request
    # harmlessly.
    memory: tuple[int, int] = (1024, int(2 * 1024 * 1024))

    # Inkling's args (relative attention, ShortConv, shared-expert sink, the custom
    # model provider) are not representable as a ModelArchitecture, so the launcher
    # sources upstream's script and passes ${MODEL_ARGS[@]} verbatim.
    miles_model_script: str = "scripts/models/inkling-small.sh"
    # Selects miles' megatron→HF weight mapping (miles/backends/megatron_utils/
    # megatron_to_hf/inkling.py); shared by Inkling and Inkling-Small.
    model_name: str = "inkling"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            # Keep HF → torch_dist at PP1. Without it convert_hf_to_torch_dist.py
            # auto-bumps PP toward the rank count and rewrites the decoder split.
            "CONVERT_KEEP_PP1": "1",
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            "SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM": "false",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "MILES_SGLANG_DUMMY_LOAD": "0",
            "SGLANG_SERVER_ENGINE_ROLLOUT_RETURN_LOGPROB": "1",
            "RAY_memory_monitor_refresh_ms": "0",
            # Upstream sets these for GB300. MNNVL is absent on Modal H200, where
            # enabling it is a no-op rather than a hang: NCCL probes for the fabric
            # and falls back to NVLink/PCIe. NCCL_NVLS_ENABLE is the one that had to
            # deviate.
            "NCCL_MNNVL_ENABLE": "1",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_RAS_ENABLE": "0",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    # "raw" (not the "bridge" default) turns on HF → torch_dist conversion. miles
    # then sets load=ref_load with no-load-optim/rng + finetune on a fresh run,
    # which is exactly run_inkling.py's --load {torch_dist}. No reference model is
    # materialized: that is gated on kl_coef/use_kl_loss, both off for GRPO here.
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Inkling-Small_torch_dist"
    # hf_checkpoint is intentionally unset — it is inherited from the attached
    # ModelConfig, so pointing the recipe at a sliced checkpoint is a model swap.

    # Convert on one node at TP8 / PP1 / EP8 / ETP1, per the upstream doc. The
    # training layout (TP4 PP8) would need 4 nodes and a PP8 split; torch_dist
    # reshards on load, so conversion parallelism is free to differ.
    conversion_tensor_model_parallel_size: int = 8
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    # A Volume buffers writes to container-local disk before they are committed, so
    # writing this ~550 GB checkpoint needs that much scratch even though nothing is
    # staged. Undersizing it exhausts the buffer mid-write and surfaces as the zip
    # writer's "unexpected pos", not as a clean ENOSPC.
    convert_ephemeral_disk_mb: int | None = 1024 * 1024

    # ── Cluster + parallelism ────────────────────────────────────────────────
    actor_num_nodes: int = 4
    actor_num_gpus_per_node: int = 8
    colocate: bool = True
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 8
    # 42 = 7x5 + 7 across 8 stages.
    decoder_last_pipeline_num_layers: int = 7
    expert_model_parallel_size: int = 4
    expert_tensor_parallel_size: int = 1

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    transformer_impl: str = "transformer_engine"
    bf16: bool = True
    no_bias_dropout_fusion: bool = True
    distributed_timeout_minutes: int = 30

    # ── Rollout + reward ─────────────────────────────────────────────────────
    rm_type: str = "math"
    balance_data: bool = True
    num_rollout: int = 100
    rollout_batch_size: int = 64
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 2048
    rollout_temperature: float = 1.0
    global_batch_size: int = 128
    # One engine spans 2 nodes.
    rollout_num_gpus_per_engine: int = 16
    skip_eval_before_train: bool = True

    # ── GRPO ─────────────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    eps_clip_c: float = 3.0
    use_tis: bool = True
    # R3: replays the rollout's routed expert IDs (including across media-expanded
    # multimodal sequences) so train and inference routing agree.
    use_rollout_routing_replay: bool = True

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer: str = "adam"
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    use_distributed_optimizer: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    no_check_for_nan_in_loss_and_grad: bool = True

    # ── SGLang ───────────────────────────────────────────────────────────────
    sglang_attention_backend: str = "fa4"
    sglang_moe_runner_backend: str = "triton"
    # ShortConv state reuses the mamba cache machinery.
    sglang_mamba_scheduler_strategy: str = "extra_buffer"
    sglang_enable_multimodal: bool = True
    sglang_context_length: int = 4096
    sglang_disable_custom_all_reduce: bool = True

    # Resolved from ``modality`` by ``_fields``. Set it explicitly to pin a provider
    # regardless of the mode. MODEL_ARGS precede the recipe's flags on the command line,
    # so whichever value lands here overrides the text provider baked into the model
    # script.
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
    """Full-parameter GRPO on Inkling-Small (4 nodes x 8 H200)."""

    lr: float = 5e-5

    # Dynamic token packing exposes a PP-p2p x EP-all-to-all NCCL launch-order race
    # on varlen shapes, so upstream pins a fixed micro-batch for full-parameter runs.
    # This overrides MilesRecipe's use_dynamic_batch_size=True default.
    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1

    # 276 B of params, grads and fp32 optimizer state does not fit 32 H200s. The
    # optimizer offloads to host RAM and the paused actor spills to node-local disk,
    # which is upstream's launcher default; putting the actor in host RAM as well
    # exceeded it.
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/tmp/train_offload"
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    sglang_mem_fraction_static: float = 0.65
    sglang_max_running_requests: int = 64
    sglang_max_total_tokens: int = 327680

    # Save policy weights only, not distributed-optimizer state. Megatron saves both by
    # default: 276 B x 2 B (bf16 params) + 276 B x 12 B (fp32 main params + two Adam
    # moments) is ~3.9 TB, i.e. ~966 GB per node, and a Modal Volume buffers all of it
    # to container-local disk because nothing commits mid-save, which exhausts it.
    # Params alone are ~138 GB per node. Trade-off: a resumed run restarts the Adam
    # moments rather than continuing them.
    no_save_optim: bool = True
    # Must be paired with no_save_optim: Megatron's load_checkpoint does
    # `optimizer.load_state_dict(state_dict['optimizer'])` unconditionally unless this
    # is set (checkpointing.py:1809), so resuming a params-only checkpoint dies with
    # KeyError: 'optimizer'.
    no_load_optim: bool = True
    # Headroom for the Volume's local write buffer during the ~138 GB/node params
    # save, on top of the CPU-offloaded optimizer's own host/disk usage.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "Inkling_Small_Recipe":
        """Keep the reservation when a caller supplies their own kwargs.

        Passing ``{"secrets": [...]}`` would otherwise drop it and the params-only
        save would exhaust local disk. A caller who names ``ephemeral_disk`` wins.
        """
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
    """Rank-32 all-linear LoRA GRPO on Inkling-Small (4 nodes x 8 H200).

    The base model stays frozen in both runtimes and only the adapter is synced to
    the colocated engine after each step, so no optimizer offload is needed.
    """

    # Upstream's launcher default is 5e-6, which upstream itself documents as reading
    # like "not learning": the zero-initialised B factors need hundreds of rollouts to
    # accumulate a visible delta-W. 2e-4 is the validated value.
    lr: float = 2e-4

    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    target_modules: str | None = "all-linear"
    # Routed experts share one outer factor; the expert-specific factors follow EP.
    experts_shared_outer_loras: bool = True
    sglang_lora_backend: str | None = "triton"
    sglang_lora_use_virtual_experts: bool = True
    # RL serving holds exactly the current policy's adapter.
    sglang_max_loras_per_batch: int = 1
    sglang_max_lora_rank: int = 32

    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 4096

    sglang_ep_size: int = 16
    sglang_mem_fraction_static: float = 0.65
    sglang_max_running_requests: int = 32
    sglang_max_total_tokens: int = 320000
    sglang_cuda_graph_max_bs: int = 64
    sglang_max_mamba_cache_size: int = 256
    # At a 16-GPU engine there is enough headroom to keep both resident.
    no_offload_rollout: bool = True
    no_offload_train: bool = True
