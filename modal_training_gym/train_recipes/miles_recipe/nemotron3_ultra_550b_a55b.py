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

It did, however, surface image-wide fixes that now apply to every miles image,
driven by this model's scale rather than by anything nemotron-specific:

- ``megatron_patches/patch_dist_ckpt_fork_retry`` — Megatron's torch_dist writer
  forks 2 helper processes per rank and that fork intermittently returns EAGAIN
  on Modal, killing the run at its checkpoint save.
- ``megatron_patches/patch_dist_ckpt_read_retry`` — torch_dist reads off a
  Modal Volume intermittently fail with EINVAL on a subset of ranks while the
  files are intact; retry with reopen instead of losing the run.
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
    # host RAM: measured (cgroup accounting, 16 nodes) ~1360 GiB steady state plus
    # ~320 GiB during a checkpoint save. 2600 GiB keeps the save peak well under
    # the cap on a 2.84 TiB host.
    memory: tuple[int, int] = (1024, 2600 * 1024)

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
            # The nemotron DP-attention path uses existing kernels; upstream skips
            # the blanket sgl-kernel version guard, which otherwise refuses to
            # start the engines.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # 0, overriding MilesRecipe's default of 1. NVLS is intra-node NVLink
            # SHARP; a rollout engine here spans 4 nodes, so its tensor-parallel
            # group cannot use it and ncclCommInitRank fails with "invalid usage"
            # while building _TP. Inkling-Small deviates for the same reason.
            "NCCL_NVLS_ENABLE": "0",
            # A failed engine bring-up otherwise surfaces only as "invalid usage"
            # with no detail. INIT,NET keeps INFO scoped to bring-up and the
            # network path — chatty only during init — to capture why the NCCL
            # net plugin fails to initialize on individual hosts (the recurring
            # per-node "Failed to initialize any NET plugin" fleet issue).
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
            # NCCL's RAS reliability thread segfaults while ranks wind down after
            # the final checkpoint save, failing a run whose work is already
            # done. RAS only reports health, so disabling it costs nothing.
            # Inkling-Small sets it for the same reason.
            "NCCL_RAS_ENABLE": "0",
            # Ray's memory monitor kills actors under host-memory pressure, which
            # is the regime this model runs in (CPU-offloaded optimizer for 550 B).
            # Inkling-Small disables it for the same reason.
            "RAY_memory_monitor_refresh_ms": "0",
            # An engine spans 4 nodes; the three that did not download the
            # checkpoint read ~1 TB off the Modal Volume at ~1 GiB/s, so weight
            # load alone takes ~30 min. SGLang's post-load barrier allows 480 s by
            # default; `patch_sglang_load_barrier` makes it configurable so models
            # with single-node engines keep fast dead-rank detection while this
            # one opts in.
            "MILES_LOAD_BARRIER_TIMEOUT_S": "3600",
            # Pace checkpoint shard writes (patch_dist_ckpt_write_throttle,
            # MB/s per writer process; 16 writers/node => ~0.5 GiB/s/node).
            # Unpaced, the 64 GiB/node burst outruns the Volume mount's upload
            # and the congestion resets TCP connections between cluster
            # containers, failing the run at its save. Reads at ~1 GiB/s/node
            # are proven safe; this keeps writes inside that envelope for a
            # ~2 min save instead of ~1 min.
            "MILES_CKPT_WRITE_BWLIMIT_MBPS": "32",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    # AutoBridge plus miles' NemotronH shim read the HF checkpoint directly:
    # miles' load_checkpoint dispatches on what --ref-load points at, and an HF
    # directory routes to _load_checkpoint_hf. There is no offline torch_dist
    # conversion step, so none of the conversion_* fields apply.
    #
    # The bridge logs "Unrecognized mapping type for mtp.*" on every rank: the
    # checkpoint ships an MTP (speculative-draft) head whose layer norms have no
    # bridge mapping. Benign while MTP is neither trained (mtp_num_layers unset)
    # nor served (no sglang_speculative_algorithm); enabling speculative decoding
    # with this head requires fixing that mapping first.
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
    # Not a MilesRecipe field: declaring it emits --ckpt-fully-parallel-load.
    # Megatron's default load has every DP replica read its full TP/PP slice, so
    # a resume pulls DP x 1.1 TB through the Volume mount at once; fully-parallel
    # load reads each shard once cluster-wide and broadcasts. Safe with
    # params-only checkpoints — Megatron's dp_zero_gather_scatter conflict only
    # arises when optimizer state is loaded, which no_load_optim disables.
    ckpt_fully_parallel_load: bool = True
    # Room for the Volume's write buffer during the 1.12 TB download.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    # ── Weight sync ──────────────────────────────────────────────────────────
    # The colocated sync pays a convert -> serialize -> IPC round trip per
    # chunk, so 1.1 TB through miles' 512 MB default is ~9,000 round trips:
    # measured 598-688 s per sync. 2 GiB — upstream's tuning for the comparable
    # GLM5-744B and Kimi-K2.5 — measured 330 s. The remaining floor is the
    # bridge HF export; a direct megatron_to_hf mapping for nemotron_h is the
    # path to Inkling-class (~40 s) syncs.
    update_weight_buffer_size: int | None = 2 * 1024**3

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
    # Small rollout batch/sample sizes can idle DP ranks;
    # see sglang#34535: https://github.com/sgl-project/sglang/pull/34535
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
    # On, deviating from upstream's script, because the host cannot hold the
    # un-sharded state: TP8 x PP4 = 32 model-parallel ranks put ~17.2 B params on
    # each, and at 12 B/param of fp32 master plus two Adam moments that is
    # ~1.50 TiB per 8-rank node — the host OOMs at the checkpoint save. Sharding
    # across DP=4 brings it to ~0.38 TiB per node; mathematically equivalent, it
    # changes where state lives, not the update. Inkling-Small already pairs it
    # with CPU offload.
    use_distributed_optimizer: bool = True

    # Spill the paused actor to node-local disk, not host RAM: miles' default
    # --offload-train-target of "cpu" puts the offloaded trainer in host memory
    # on top of the CPU-offloaded optimizer above, which segfaults the actors.
    # Inkling-Small documents the same at half this size. /tmp is on the
    # container's overlay filesystem on Modal, not a tmpfs, so it is genuinely
    # disk — upstream warns that a tmpfs path defeats this.
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
