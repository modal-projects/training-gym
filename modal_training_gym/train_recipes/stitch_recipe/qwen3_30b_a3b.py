"""Qwen3-30B-A3B NVFP4 GRPO, disaggregated — the port of stitch's own recipe.

A faithful port of ``cookbook/miles_disagg/configs/qwen3_30b_a3b_nvfp4_46.py``
(the humansand.ai NVFP4 "4-bitter" recipe, miles PR #1261) onto the training-gym
launcher: 1×8 B200 miles trainer publishing sparse deltas, rollouts served by a
Flash pool of single-B200 SGLang replicas.

The precision contract is the substance of this recipe, and it is a *pair* of
environment blocks that must agree:

- ``_NVTE_QUANT_ENV`` drives TransformerEngine — the trainer's NVFP4 forward
  (per-token row-scaled experts, dequantized backward, 4-over-6 block scaling),
  its weight export, *and* the served baseline's conversion (via ``prep_env``).
- ``_FLASHINFER_QUANT_ENV`` drives the pool's activation quantization kernels
  with the same conventions (MAE error mode, e4m3-max 256, fast math off), which
  is what keeps rollout logprobs close to the trainer's.

Weight bytes are exact by construction: miles' export and its
``tools/convert_hf_to_nvfp4.py`` share one TE-direct quantizer, so the served
packing equals the export packing and a sparse XOR delta stays sparse. Drift in
either block shows up as a checksum failure on the first delta apply.

Only the layers that matter are quantized: the last 7 of 48 stay BF16, and the
NVFP4 matchers select the routed-expert GEMMs (Qwen3 MoE has no shared expert).
"""

from __future__ import annotations

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe
from modal_training_gym.train_recipes.miles_recipe.recipe import CHECKPOINTS_PATH
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    MEGATRON_R3_DISPATCH_PATCH,
    MEGATRON_RESHARDABLE_STEP_PATCH,
)
from modal_training_gym.train_recipes.stitch_recipe.recipe import StitchRecipe
from modal_training_gym.train_recipes.stitch_recipe.serve import StitchServeConfig
from modal_training_gym.train_recipes.stitch_recipe.train import StitchTrainConfig

# Where prepare_checkpoints leaves the two checkpoints this recipe needs.
BF16_CHECKPOINT_PATH = str(CHECKPOINTS_PATH / "qwen3-30b-a3b-bf16")
NVFP4_CHECKPOINT_PATH = str(CHECKPOINTS_PATH / "qwen3-30b-a3b-nvfp4-46")

# The TE side of the quantizer contract: training, export, and the served
# baseline's conversion all run under exactly these settings.
_NVTE_QUANT_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MAE",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    # The BF16 backward differentiates DQ(Q(w)) — the same quantized function the
    # forward evaluated.
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
}

# The sampler side of the same contract, for the pool's activation quantization.
_FLASHINFER_QUANT_ENV = {
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MAE",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
}

# NVFP4 per-tensor recipe + the matchers that pick which GEMMs get it. Materialized
# to a node-local YAML file by the launcher, because every Ray actor re-reads it.
_TE_PRECISION_CONFIG = {
    "configs": {
        "nvfp4": {
            "transformer_engine_config_type": "TEQuantizationParams",
            "training_recipe": {"fp4_quantization_recipe": "nvfp4"},
        },
        "bf16": {
            "transformer_engine_config_type": "TEQuantizationParams",
            "training_recipe": {},
        },
    },
    "matchers": {
        "routed_experts_fc1_nvfp4": {
            "type": "glob",
            "enabled": True,
            "pattern": "*.mlp.experts.linear_fc1",
            "config": "nvfp4",
        },
        "routed_experts_fc2_nvfp4": {
            "type": "glob",
            "enabled": True,
            "pattern": "*.mlp.experts.linear_fc2",
            "config": "nvfp4",
        },
        "default_bf16": {
            "type": "glob",
            "enabled": True,
            "pattern": "*",
            "config": "bf16",
        },
    },
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Stitch_Train(StitchTrainConfig):
    """1×8 B200 miles actor cluster for Qwen3-30B-A3B under NVFP4 QAT.

    Topology, algorithm, and precision come from stitch's
    ``qwen3_30b_a3b_nvfp4_46`` config verbatim; the architecture comes from miles'
    model script rather than ``ModelArchitecture`` flags.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu_type: str = "B200"
    region: str | None = "us"
    # CPU-offloaded optimizer state for 128 experts wants host RAM headroom.
    memory: tuple[int, int] | None = (128 * 1024, 2 * 1024 * 1024)
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    num_gpus_per_node: int = 8
    # R3 routing replay needs Megatron's dropless-dispatch fix; the reshardable
    # step lets the CPU-offloaded optimizer survive a changed DP layout.
    megatron_runtime_patches: list[str] = field(
        default_factory=lambda: [
            MEGATRON_R3_DISPATCH_PATCH,
            MEGATRON_RESHARDABLE_STEP_PATCH,
        ]
    )

    # ── Checkpoints: BF16 masters to train from, NVFP4 base to serve ─────────
    miles_model_script: str = "scripts/models/qwen3-30B-A3B.sh"
    model_name: str = "qwen3moe"  # megatron_to_hf export dispatch
    source_hf_checkpoint: str | None = "Qwen/Qwen3-30B-A3B"
    hf_checkpoint: str = NVFP4_CHECKPOINT_PATH
    bf16_checkpoint_path: str = BF16_CHECKPOINT_PATH
    served_checkpoint_format: str = "nvfp4"
    prep_env: dict[str, str] = field(default_factory=lambda: dict(_NVTE_QUANT_ENV))
    megatron_to_hf_mode: str = "bridge"

    # ── NVFP4 QAT ───────────────────────────────────────────────────────────
    fp4_format: str = "e2m1"
    fp4_recipe: str = "nvfp4"
    fp4_param_gather: bool = False
    te_precision_config_file: dict | str | None = field(
        default_factory=lambda: dict(_TE_PRECISION_CONFIG)
    )
    # Selective precision: the last 15% of 48 layers stay BF16. All layers are MoE
    # (first_k_dense_replace=0), so there is no start carve-out.
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 7

    # ── Rollout / algorithm ─────────────────────────────────────────────────
    num_rollout: int = 20
    rollout_batch_size: int = 32
    # Long enough that math traces terminate instead of truncating (4096 clipped
    # most responses and starved the reward signal).
    rollout_max_response_len: int = 12288
    rollout_temperature: float = 0.8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    use_dynamic_global_batch_size: bool = True
    # Trainer-side client concurrency to the gateway: high enough that a rollout
    # wave drives the scaled-out pool rather than one engine.
    sglang_server_concurrency: int = 128
    # R3 (arxiv 2510.11370): replay the rollout engine's expert routing in the
    # train forward.
    use_rollout_routing_replay: bool = True
    rollout_shuffle: bool = True
    balance_data: bool = True
    rm_type: str | None = "deepscaler"
    # miles still writes a final megatron save on the last rollout, and a ~120 GB
    # torch_dist save blows the trainer's ephemeral disk (the Volume write cache
    # lives there). None means no save path exists at all; set a real interval
    # plus ephemeral_disk for a long run.
    save_interval: int | None = None

    # ── Trainer parallelism (world = TP4 × DP2 = 8, EP over the node) ───────
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 8192
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    no_check_for_nan_in_loss_and_grad: bool = True
    # TE's cuDNN fused-attention backward fails on this recipe's dynamic sequence
    # shapes (CUDNN_STATUS_BAD_PARAM). Attention stays BF16, so FlashAttention
    # does not change the NVFP4 expert-layer recipe.
    attention_backend: str = "flash"

    # ── Optimizer (CPU offload keeps GPU state tiny for ~3B active) ──────────
    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Algorithm: GRPO + truncated importance sampling ─────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    use_tis: bool = True
    eval_interval: int | None = None

    update_weights_interval: int = 1
    environment: dict = field(
        default_factory=lambda: {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NVSHMEM_DISABLE_NCCL": "1",
            "NCCL_TIMEOUT_MS": "360000000",
            **_NVTE_QUANT_ENV,
        }
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Stitch_Serve(StitchServeConfig):
    """Flash pool of single-B200 SGLang replicas serving the NVFP4 baseline.

    NVFP4 packs ~30B into ~17 GB, so one B200 per replica with room for a CPU
    weight cache — which is what makes a delta apply cheap: the sparse patch
    lands in pinned host memory and the engine reloads from there rather than
    re-reading a checkpoint. Precision comes from the served checkpoint's own
    quant config; there is deliberately no ``--quantization`` flag.
    """

    sglang: SglangRecipe = field(
        default_factory=lambda: SglangRecipe(
            gpu="B200",
            tp=1,
            context_length=16384,
            mem_fraction_static=0.8,
            chunked_prefill_size=4096,
            extra_server_args={
                "--weight-loader-prefetch-checkpoints": "",
                "--weight-loader-prefetch-num-threads": "8",
                # Delta applies land against a pinned CPU copy of the weights.
                "--enable-cpu-weight-cache": "",
                # Qwen3 MoE is GQA (no MLA), and the routed FlashInfer TRTLLM MoE
                # runner is what emits per-token routed experts for R3 replay.
                "--attention-backend": "trtllm_mha",
                "--moe-runner-backend": "flashinfer_trtllm_routed",
                "--kv-cache-dtype": "bfloat16",
                "--skip-server-warmup": "",
                "--enable-return-routed-experts": "",
            },
        )
    )
    delta_update_mode: str = "cpu"
    # Per-container autoscaler target, well below the trainer's client
    # concurrency: a rollout wave (32 × 8 = 256) must register as queue pressure
    # so Flash scales OUT to the cap instead of one engine absorbing the wave.
    concurrency: int = 24
    # The pool must be UP before the trainer sends its first rollout; the cap
    # bounds the footprint at trainer 8 + pool 3 = 11 concurrent B200.
    min_containers: int = 1
    max_containers: int | None = 3
    # CPU persist is ~2× the 23.5 GiB canonical checkpoint; the request also
    # covers staging plus the serving baseline.
    memory: tuple[int, int] | None = (128 * 1024, 512 * 1024)
    env: dict[str, str] = field(
        default_factory=lambda: {
            **_FLASHINFER_QUANT_ENV,
            "SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1",
        }
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Stitch_Recipe(StitchRecipe):
    """Qwen3-30B-A3B NVFP4 disaggregated GRPO: 1×8 B200 trainer + Flash pool."""

    train: StitchTrainConfig = field(default_factory=Qwen3_30B_A3B_Stitch_Train)
    serve: StitchServeConfig = field(default_factory=Qwen3_30B_A3B_Stitch_Serve)
