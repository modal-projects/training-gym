"""DeepSeek-V4.1-Flash GRPO recipe, ported from upstream's ``run_deepseek_v41.py``."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import DeepSeek_V4_1_Flash, ModelConfig
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

# radixark/miles#3179 (DeepSeek-V4.1 RL support) and sgl-project/sglang#38798
# (the matching engine support) are both unmerged, and the only image built from
# them, `radixark/miles:deepseek-v41`, is arm64/GB300-only — Modal's builder
# resolves amd64 and rejects it. So this recipe layers the two upstream refs onto
# the newest amd64 nightly instead: sglang is an editable install and miles' JIT
# kernels compile at runtime, so checking the refs out is enough. Replace all
# three with a single published tag once the PRs land in a nightly.
_MILES_PR = "pull/3179/head"  # b7e51b026e6d5cb681d145794be1af666e037520
_SGLANG_PR = "pull/38798/head"  # 1aa0e962b206102b7c439a4a0c4981cfec6e87bc


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_Recipe(MilesRecipe):
    """DeepSeek-V4.1-Flash GRPO recipe for 8 nodes with 8 H200 GPUs each."""

    model_config_class: ClassVar[type[ModelConfig]] = DeepSeek_V4_1_Flash

    docker_image: str = "radixark/miles:dev-202609100049"
    miles_git_ref: str | None = _MILES_PR
    sglang_git_ref: str | None = _SGLANG_PR
    gpu_type: str = "H200"
    # The fp32 optimizer state is offloaded to host RAM: ~8.5B params per GPU at
    # 12 bytes each is ~100 GiB per rank.
    memory: tuple[int, int] = (1024, int(2 * 1024 * 1024))

    # V4.1's DSA indexer, CSA compression and Engram memory are not
    # representable as a ModelArchitecture, so the launcher renders upstream's
    # scripts/models/deepseek-v4.1.py through model_args_utils.py and passes
    # ${MODEL_ARGS[@]} verbatim — including its --spec for the V4.1 layer spec.
    miles_model_name: str = "deepseek-v4.1"
    # Selects miles' megatron→HF weight mapping (miles/backends/megatron_utils/
    # megatron_to_hf/deepseekv41.py).
    model_name: str = "deepseekv41"
    # Selects the miles (not sglang) V4.1 training implementation.
    dsv4_impl: str = "miles"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            # Pin the conversion to the pipeline layout below; the converter
            # otherwise inflates PP to the rank count.
            "CONVERT_KEEP_PP1": "1",
            # The HF release is fp8 dense / packed-fp4 experts with e8m0 block
            # scales and no bf16 export; mbridge expects bf16, so the conversion
            # wrapper dequantizes weights as it reads them.
            "CONVERT_DEQUANT_HF_WEIGHTS": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
            # Upstream's V4.1 engine settings. FP4 experts are Blackwell-only, so
            # they stay off on H200; the rest guard against slow/absent kernels
            # and a load that outlives the default health-check window.
            "SGLANG_DSV4_FP4_EXPERTS": "0",
            "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
            "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
            "SGLANG_OPT_FP8_WO_A_GEMM": "0",
            "SGLANG_OPT_FUSE_WQA_WKV": "0",
            "SGLANG_DISABLE_MULTIMEM_AG": "1",
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTHONFAULTHANDLER": "1",
            # Paired with --deterministic-mode below.
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    # "raw" turns on the HF → torch_dist conversion that upstream runs as
    # `run_deepseek_v41.py prepare-spmd`; miles then loads it as ref_load.
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/DeepSeek-V4.1-Flash_torch_dist"

    # Upstream's conversion layout (TP4, PP1) over two nodes instead of one: the
    # converter builds the model on GPU and the bf16 weights are ~970 GB, which
    # is ~121 GiB a rank at EP8 — fine on their 288 GiB GB300s, an OOM on an
    # H200. EP16 halves it, served by data-parallel replicas; TP stays at 4
    # since the 129280 vocab only splits 8 ways with a smaller
    # make-vocab-size-divisible-by. torch_dist reshards on load, so this layout
    # is independent of the training one below.
    conversion_tensor_model_parallel_size: int = 4
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 16
    conversion_expert_tensor_parallel_size: int = 1
    # The bf16 torch_dist checkpoint is ~1.1 TB, and a Volume buffers writes on
    # container-local disk before committing them.
    convert_ephemeral_disk_mb: int | None = 2048 * 1024

    # ── Cluster + parallelism ────────────────────────────────────────────────
    # ~560B total parameters: bf16 weights and grads alone are ~2.2 TB, so the
    # expert dimension is sharded across every rank (upstream's default
    # ep_size = actor_num_nodes * gpus_per_node / pp_size).
    actor_num_nodes: int = 8
    tensor_model_parallel_size: int = 4
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    expert_model_parallel_size: int = 64
    expert_tensor_parallel_size: int = 1

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    transformer_impl: str = "transformer_engine"
    bf16: bool = True
    qkv_format: str = "bshd"
    # V4.1's router gate and its expert-bias correction term are frozen during RL.
    moe_router_freeze_gate: bool = True
    freeze_e_score_correction_bias: bool = True
    # Bit-exact recomputation, so a rollout's log probs match the training pass.
    deterministic_mode: bool = True
    train_memory_margin_bytes: int = 3 * 1024**3
    colocate_memory_peak_device: str = "cpu"
    update_weight_buffer_size: int = 1024**3

    # ── Rollout + reward ─────────────────────────────────────────────────────
    rm_type: str = "math"
    balance_data: bool = True
    num_rollout: int = 5
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    # One train step per rollout, so the global batch is the rollout itself.
    num_steps_per_rollout: int = 1
    global_batch_size: int = 128
    # Dynamic packing conflicts with --qkv-format bshd upstream.
    use_dynamic_batch_size: bool = False
    rollout_temperature: float = 0.8
    # Upstream's gsm8k debug default is 256; math-RL prompts need room to finish.
    rollout_max_response_len: int = 2048
    max_tokens_per_gpu: int = 2048
    micro_batch_size: int = 1
    skip_eval_before_train: bool = True
    # R3: replay the rollout's routed expert ids during training so train and
    # inference routing agree.
    use_rollout_routing_replay: bool = True

    # ── Optimizer + GRPO ─────────────────────────────────────────────────────
    use_distributed_optimizer: bool = True
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── SGLang ───────────────────────────────────────────────────────────────
    # One engine per node, expert-parallel across it.
    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int = 8
    sglang_dp_size: int = 1
    sglang_attention_backend: str = "dsv4"
    sglang_moe_runner_backend: str = "auto"
    sglang_mem_fraction_static: float = 0.6
    sglang_max_running_requests: int = 128
    sglang_disable_cuda_graph: bool = True
    sglang_disable_radix_cache: bool = True

    # miles#3179 predates miles#3124: sglang >= 0.5.19 no longer auto-detects
    # ``device`` when a ServerArgs is constructed, and miles always renders it on
    # the engine command line, so an unset device becomes ``--device None`` and
    # fails its own argv round-trip check. Naming the device sidesteps that.
    extra_config: dict | None = field(default_factory=lambda: {"sglang_device": "cuda"})

    rollout_health_check_interval: int = 300
    rollout_health_check_timeout: int = 300
    # The first engine start compiles deepgemm kernels for a 560B MoE; without a
    # long grace period the health checker kills the engines mid-warmup.
    rollout_health_check_first_wait: int = 3600
