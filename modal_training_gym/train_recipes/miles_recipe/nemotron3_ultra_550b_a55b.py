"""Miles recipe for NVIDIA Nemotron-3-Ultra-550B-A55B.

Mirrors the upstream-validated profile from
``scripts/run_nemotron_3_ultra_550b_a55b.py`` and
``docs/models/nemotron/nemotron-3-ultra.md`` at ``--num-nodes 16``:
16 nodes x 8 H200 colocated, TP8 / SP / PP4 / EP32 / ETP1 (DP4), four 32-GPU
SGLang engines running EP32 + DP-attention, GRPO on DAPO-Math-17k with a
CPU-offloaded Adam.

Full-parameter only — upstream ships no LoRA model script for this model, so
there is no validated adapter profile to mirror.

**No image bump**: nemotron support (the model scripts, the ``NemotronHBridge``
shim, sglang's ``nemotron_h`` runner) is already inside ``MilesRecipe``'s default
image, and this recipe carries no `patch_files` of its own.

It did, however, surface two image-wide fixes that now apply to every miles
image, both driven by this model's scale rather than by anything nemotron-specific:

- ``megatron_patches/patch_dist_ckpt_fork_retry`` — Megatron's torch_dist writer
  forks 2 helper processes per rank and that fork intermittently returns EAGAIN
  on Modal, killing the run at its checkpoint save.
- ``miles/modal_helpers/patches/patch_sglang_load_barrier`` — SGLang allows 8 min
  between the first and last rank finishing weight load. Only the node that
  downloaded the checkpoint reads it from page cache; the rest pull ~1 TB off a
  Modal Volume at ~1 GiB/s, so a multi-node engine cannot make that deadline.

See ``.gym/new_models/Nemotron3_Ultra_550B_A55B/model_setup.md`` for the
derivation, the deviations from ``run_nemotron_3_ultra_550b_a55b.py``, and
expected step timings.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import ModelConfig, Nemotron3_Ultra_550B_A55B
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

# The Volume buffers writes to container-local disk, so downloading the ~1.12 TB
# checkpoint into the shared HF cache needs that much scratch in the train
# container. Bridge mode does the download inside train() — there is no separate
# conversion container to size instead.
_EPHEMERAL_DISK_MIB = 2 * 1024 * 1024


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Nemotron3_Ultra_550B_A55B_Recipe(MilesRecipe):
    """Full-parameter GRPO on Nemotron-3-Ultra-550B-A55B (16 nodes x 8 H200)."""

    model_config_class: ClassVar[type[ModelConfig]] = Nemotron3_Ultra_550B_A55B

    gpu_type: str = "H200"
    # Upstream's optimizer offload keeps fp32 main params and both Adam moments in
    # host RAM. See model_setup.md for the per-node estimate and for why
    # use_distributed_optimizer stays off (upstream leaves it off).
    memory: tuple[int, int] = (1024, 2 * 1024 * 1024)

    # nemotron_h is a hybrid Mamba2 + attention + latent-MoE stack: the block
    # pattern, the Mamba dimensions and moe_latent_size=2048 are not representable
    # as a ModelArchitecture, so the launcher runs miles' model_args_utils.py for
    # this name and splices the printed ${MODEL_ARGS[@]} in ahead of the recipe's
    # own flags. Upstream ships model_args() as a .py module, not a .sh script,
    # which is why this is miles_model_name rather than miles_model_script.
    miles_model_name: str = "nemotron-3-ultra-550b-a55b"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            # The nemotron DP-attention path uses existing kernels; upstream skips
            # the blanket sgl-kernel version guard, which otherwise refuses to
            # start the engines.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # 0, overriding MilesRecipe's default of 1. NVLS is intra-node NVLink
            # SHARP; a rollout engine here spans 4 nodes, so its tensor-parallel
            # group cannot use it, and enabling it made ncclCommInitRank fail with
            # "NCCL error: invalid usage" while building _TP, killing engine init
            # (`weary-sole-4b773f2e4618`). Inkling-Small -- the only other
            # multi-node miles recipe here -- reached the same conclusion: "NCCL
            # NVLS_ENABLE is the one that had to deviate" on Modal.
            "NCCL_NVLS_ENABLE": "0",
            # The NCCL error message itself asks for this, and a failed engine
            # bring-up otherwise surfaces only as "invalid usage" with no detail.
            # WARN is quiet unless something is wrong.
            "NCCL_DEBUG": "WARN",
            # Ray's memory monitor kills actors under host-memory pressure, which
            # is the regime this model runs in (CPU-offloaded optimizer for 550 B).
            # Inkling-Small disables it for the same reason.
            "RAY_memory_monitor_refresh_ms": "0",
            # An engine spans 4 nodes; the three that did not download the
            # checkpoint read ~1 TB off the Modal Volume at ~1 GiB/s, so weight
            # load takes ~30 min (measured: 28 min on `poky-coyote-b2814b94964f`).
            # SGLang's post-load barrier allows 480 s by default and killed the
            # first 16-node attempt. `patch_sglang_load_barrier` makes that
            # configurable, defaulting to upstream's 480 — models with single-node
            # engines keep fast dead-rank detection; this one opts in.
            "MILES_LOAD_BARRIER_TIMEOUT_S": "3600",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    # AutoBridge plus miles' NemotronH shim read the HF checkpoint directly:
    # miles' load_checkpoint dispatches on what --ref-load points at, and an HF
    # directory routes to _load_checkpoint_hf. There is no offline torch_dist
    # conversion step, so none of the conversion_* fields apply.
    megatron_to_hf_mode: str = "bridge"
    ref_load: str = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    # hf_checkpoint is intentionally unset — it comes from the attached
    # ModelConfig, so pointing the recipe at another checkpoint is a model swap.
    save_interval: int = 50
    # Params only. 550 B x 2 B (bf16 params) plus fp32 main params and two Adam
    # moments is multiple terabytes, and a Modal Volume buffers all of it to
    # container-local disk because nothing commits mid-save.
    no_save_optim: bool = True
    # Must be paired with no_save_optim: Megatron's load_checkpoint does
    # `optimizer.load_state_dict(state_dict['optimizer'])` unconditionally unless
    # this is set, so resuming a params-only checkpoint dies with
    # KeyError: 'optimizer'. Trade-off: a resumed run restarts the Adam moments.
    no_load_optim: bool = True
    # Room for the Volume's write buffer during the 1.12 TB download.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    # ── Cluster + parallelism ────────────────────────────────────────────────
    actor_num_nodes: int = 16
    actor_num_gpus_per_node: int = 8
    colocate: bool = True
    # Mamba n_groups=8 requires n_groups % tp == 0, so attention/Mamba tensor
    # parallelism cannot exceed 8. Scale comes from PP and EP instead.
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 32
    expert_tensor_parallel_size: int = 1

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    # Not a MilesRecipe field: declaring it here is what emits
    # --log-probs-chunk-size, which keeps the log-prob pass inside the memory
    # budget at this shape.
    log_probs_chunk_size: int = 128
    # Also recipe-declared. Megatron's 10-minute default collective timeout does
    # not survive a cold 1.12 TB bridge load spread over 128 ranks.
    distributed_timeout_minutes: int = 60

    # ── Rollout + reward ─────────────────────────────────────────────────────
    rm_type: str | None = "deepscaler"
    balance_data: bool = True
    rollout_shuffle: bool = True
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0
    # The 550 B (~1.1 TB BF16) does not fit one 8-GPU engine.
    rollout_num_gpus_per_engine: int = 32
    skip_eval_before_train: bool = True
    # Four 32-GPU engines each load ~280 GB of weights and capture CUDA graphs
    # before serving; miles' zero-second default grace period kills them
    # mid-startup.
    #
    # 3600, not 1800: an engine spans 4 nodes, and the three that did not download
    # the checkpoint read it cold off the Modal Volume at ~1 GiB/s (measured), so
    # load alone can run 20-40 min. `patch_sglang_load_barrier` raises SGLang's own
    # post-load barrier to 3600 s for the same reason — leaving this at 1800 would
    # just move the failure from the barrier to the health checker.
    rollout_health_check_first_wait: int = 3600
    # Left off deliberately, unlike every other large-MoE recipe here: upstream
    # does not enable routing replay for the 108-layer Ultra because the routing
    # capturer needs a fix for per-layer top-22 under DP-attention.
    # Train/rollout logprob diff is ~0.01 without it.
    use_rollout_routing_replay: bool = False

    # ── GRPO ─────────────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # Spill the paused actor to node-local disk, not host RAM. miles defaults
    # --offload-train-target to "cpu", which puts the offloaded trainer in host
    # memory *on top of* the CPU-offloaded optimizer above. Inkling-Small
    # documents that as already exceeding the node at 276 B; this model is twice
    # that, and leaving the default killed a 16-node run
    # (`poky-coyote-b2814b94964f`): four Megatron actors hit
    # "!!!!!!! Segfault encountered !!!!!!!" inside `offload_train`, Ray reported
    # them unavailable, and the gym silently retried the whole run.
    #
    # /tmp is on the container's overlay filesystem on Modal, not a tmpfs, so it
    # is genuinely disk — upstream warns that a tmpfs path defeats this.
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/tmp/train_offload"

    # ── Batching + precision ─────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 1024
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    attention_backend: str = "auto"

    # ── SGLang ───────────────────────────────────────────────────────────────
    # attn_tp = rollout_num_gpus_per_engine / sglang_dp_size = 8, which satisfies
    # the same Mamba n_groups=8 rule the training side has.
    sglang_ep_size: int = 32
    sglang_dp_size: int = 4
    sglang_enable_dp_attention: bool = True
    sglang_mem_fraction_static: float = 0.7

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "Nemotron3_Ultra_550B_A55B_Recipe":
        """Keep the reservation when a caller supplies their own kwargs.

        Passing ``{"secrets": [...]}`` would otherwise drop it and the run would
        die part-way through the 1.12 TB download. A caller who names
        ``ephemeral_disk`` still wins.
        """
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self
