from collections.abc import Callable
from dataclasses import field
from typing import Any, ClassVar, Literal

import modal
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.metrics import MetricConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.base import (
    BaseTrainRecipe,
    # Re-exported for backwards compatibility (e.g. frameworks/miles/launcher.py
    # imports the volume paths from this module).
    CHECKPOINTS_PATH as CHECKPOINTS_PATH,
    DATA_PATH as DATA_PATH,
    HF_CACHE_PATH as HF_CACHE_PATH,
    JSON_CONFIG_FIELDS as JSON_CONFIG_FIELDS,
)
from modal_training_gym.train_recipes.gpu_allocation import (
    resolve_gpu_allocation,
    validate_num_experts_divisible_by_expert_parallel_size,
)

_MILES_SKIP = {
    "environment",
    "async_mode",
    "miles_model_script",
    "miles_model_name",
    "source_hf_checkpoint",
    "megatron_conversion_hf_checkpoint",
    "docker_image",
    "gpu_type",
    "memory",
    "cpu",
    "cloud",
    "region",
    "name",
    "app_tags",
    "image_overlay",
    "image_run_commands",
    "image_env",
    "local_miles",
    "patch_files",
    "substep_timing",
    "metrics",
    # Callables shipped by value into the containers; the launcher writes the
    # resolved import path into `extra_config` (rm/generate) or back onto the
    # field itself (the *_path flags remapped in `_fields`).
    "custom_rm_function",
    "custom_generate_function",
    "custom_reward_post_process_function",
    "custom_rollout_log_function",
    "custom_eval_rollout_log_function",
    "rollout_function",
    "custom_megatron_before_log_prob_hook",
    "custom_megatron_before_train_step_hook",
    "train_function_kwargs",
    # Conversion-only parallelism and scratch: launcher-side, never forwarded to
    # the miles CLI.
    "conversion_tensor_model_parallel_size",
    "conversion_pipeline_model_parallel_size",
    "conversion_expert_model_parallel_size",
    "conversion_expert_tensor_parallel_size",
    "convert_ephemeral_disk_mb",
    "capture_trace",
    "trace_sample_limit",
}

YAML_CONFIG_FIELDS = ("eval_config", "extra_config", "sglang_config")

# Callable fields whose resolved import path Miles reads from a `--<name>-path`
# CLI flag rather than from the YAML custom-config.
_HOOK_PATH_FLAGS = {
    "rollout_function": "rollout_function_path",
    "custom_reward_post_process_function": "custom_reward_post_process_path",
    "custom_rollout_log_function": "custom_rollout_log_function_path",
    "custom_eval_rollout_log_function": "custom_eval_rollout_log_function_path",
    "custom_megatron_before_log_prob_hook": "custom_megatron_before_log_prob_hook_path",
    "custom_megatron_before_train_step_hook": "custom_megatron_before_train_step_hook_path",
}

_HOOK_PATH_CONFIG_KEYS = {
    "custom_rollout_log_function": "training_gym_custom_rollout_log_function_path",
    "custom_eval_rollout_log_function": "training_gym_custom_eval_rollout_log_function_path",
    "custom_megatron_before_log_prob_hook": "training_gym_custom_megatron_before_log_prob_hook_path",
    "custom_megatron_before_train_step_hook": "training_gym_custom_megatron_before_train_step_hook_path",
}

_HOOK_WRAPPER_PATHS = {
    "custom_rollout_log_function": "modal_training_gym.frameworks.miles.phase_reporting.log_rollout_data",
    "custom_eval_rollout_log_function": "modal_training_gym.frameworks.miles.phase_reporting.log_eval_rollout_data",
    "custom_megatron_before_log_prob_hook": "modal_training_gym.frameworks.miles.phase_reporting.before_log_prob_hook",
    "custom_megatron_before_train_step_hook": "modal_training_gym.frameworks.miles.phase_reporting.before_train_step_hook",
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class MilesRecipe(BaseTrainRecipe):
    """Recipe for configuring Miles RL training on Modal.

    Fields named in ``_MILES_SKIP`` are launcher instructions — image build,
    cluster topology, W&B, checkpoint conversion, callable shipping — and are
    never emitted as CLI flags; shipped callables reach Miles as resolved
    import paths instead. Every other field becomes
    ``--<field-name-with-dashes> <value>`` via ``BaseTrainRecipe.cli_args``:
    ``None``/``False``/``""`` omit the flag, ``True`` emits it bare, lists
    become space-separated values, ``YAML_CONFIG_FIELDS`` dicts are written to
    YAML on the container and passed as paths, and ``JSON_CONFIG_FIELDS`` dicts
    are passed as inline JSON. ``sglang_*`` fields configure the rollout
    engines — Miles registers every sglang ``ServerArgs`` option under a
    ``--sglang-`` prefix.

    ## App Identity

    name : str
        Modal app title; empty lets the launcher derive one from the class.
    app_tags : dict
        Extra tags merged into the Modal app metadata for the dashboard.

    ## Modal Launcher

    docker_image : str
        Modal named Image reference for the Miles image every container runs,
        published from the upstream registry image by
        ``scripts/publish_framework_images.py``.
    environment : dict
        Env vars for the training containers (Megatron ``PYTHONPATH``, NCCL tuning).
    async_mode : bool
        Run Miles' ``train_async.py``: rollout and training overlap (off-policy).
    metrics : MetricConfig | None
        Metric tracker settings; expands to Miles' W&B-compatible flags.
    image_overlay : Callable[[modal.Image], modal.Image] | None
        Customizes the Modal image, e.g. ``lambda img: img.pip_install("pkg")``.
    local_miles : str | None
        Local Miles checkout mounted over the image's copy; no rebuild needed.
    memory : int | tuple[int, int] | None
        Modal Function memory request/limit in MiB.
    cpu : float | tuple[float, float] | None
        Modal Function CPU request/limit in cores per container.
    cloud : str | None
        Modal cloud provider to pin the cluster to.
    region : str | None
        Modal region to pin the cluster to.
    miles_model_script : str
        Script in the Miles repo sourced for ``MODEL_ARGS`` instead of arch flags.
    miles_model_name : str
        Name accepted by Miles' ``model_args_utils.py``.
    source_hf_checkpoint : str | None
        Source checkpoint when it differs from the model's own.
    megatron_conversion_hf_checkpoint : str | None
        HF weights used for the HF→Megatron conversion instead of the model's own.
    patch_files : list[str]
        Local patch scripts applied to Miles/Megatron sources at image build.
    image_run_commands : list[str]
        Extra shell commands run while building the image.
    image_env : dict[str, str]
        Extra env vars baked into the image.
    train_function_kwargs : dict[str, Any]
        Extra Modal Function kwargs for the train function; supports
        ``ephemeral_disk`` (MiB), ``secrets`` and ``experimental_options``.
    capture_trace : bool
        Attach miles' per-sample execution trace (generate/reward/tool-call
        timeline) to recorded rollouts for the dashboard.
    trace_sample_limit : int
        With ``capture_trace``, number of samples per rollout that get a
        trace attached (sampling keeps the added data volume small).

    ## Cluster and Parallelism

    gpu_type : str
        Modal GPU type for every node, e.g. ``"H100"`` or ``"H200"``.
    colocate : bool
        Trainer and rollout engines share GPUs; ``False`` gives each its own.
    actor_num_nodes : int
        Nodes for the Megatron actor (trainer).
    actor_num_gpus_per_node : int
        GPUs per actor node.
    rollout_num_gpus : int | None
        Rollout-engine GPUs when disaggregated; ``None`` lets the resolver size it.
    rollout_num_gpus_per_engine : int
        GPUs per sglang engine — its tensor-parallel size.
    train_backend : str
        Training backend, e.g. ``"megatron"``.
    tensor_model_parallel_size : int
        Megatron tensor-parallel size for the actor.
    pipeline_model_parallel_size : int
        Megatron pipeline-parallel size for the actor.
    context_parallel_size : int | None
        Megatron context-parallel size; multiplies the effective context length.
    expert_model_parallel_size : int | None
        Expert-parallel size for MoE; must divide the model's ``num_experts``.
    expert_tensor_parallel_size : int | None
        Tensor-parallel size within each expert.
    decoder_last_pipeline_num_layers : int | None
        Layers placed on the last pipeline stage, to rebalance an uneven split.
    sequence_parallel : bool
        Megatron sequence parallelism (requires TP > 1).
    use_critic : bool
        Train a separate critic model (PPO-style; GRPO runs without one).
    critic_num_nodes : int | None
        Nodes for the critic when ``use_critic`` is set.
    critic_num_gpus_per_node : int | None
        GPUs per critic node.

    ## Rollout and Sampling

    num_rollout : int
        Total rollout steps (= training steps) for the run.
    rollout_batch_size : int
        Prompts per rollout step, each expanded into a group of responses.
    rollout_max_response_len : int
        Max generated tokens per sample.
    rollout_temperature : float
        Sampling temperature for rollout generation.
    rollout_shuffle : bool
        Shuffle the prompt dataset between epochs.
    rollout_top_p : float
        Nucleus-sampling top-p for rollout generation.
    rollout_stop_token_ids : list[int] | None
        Extra token ids that terminate generation.
    use_miles_router : bool
        Route rollout requests through Miles' router, not engines directly.
    rollout_top_k : int | None
        Top-k for rollout generation; ``None`` leaves Miles' own default.
    use_rollout_routing_replay : bool
        Reuse the rollout's MoE expert routing in training.

    ## Checkpointing

    hf_checkpoint : str
        Checkpoint trained from; normally set from the attached ``ModelConfig``.
    save : str
        Checkpoint output directory (the mounted ``/checkpoints`` volume).
    save_interval : int
        Save a checkpoint every N rollout steps.
    load : str
        Directory to resume from; empty starts from the converted HF weights.
    no_save_optim : bool
        Omit optim state from checkpoints (smaller, but no exact resume).
    megatron_to_hf_mode : str
        Export mode for saved Megatron checkpoints; empty disables the export.

    ## Fault Tolerance and Health Checks

    use_fault_tolerance : bool
        Enable Miles' fault tolerance to recover from worker failures.
    rollout_health_check_interval : int
        Seconds between rollout engine ``/health_generate`` checks.
    rollout_health_check_timeout : int
        Seconds to wait for ``/health_generate`` before killing the engine.
    rollout_health_check_first_wait : int
        Grace period before checks start; raise a lot for deepgemm compilation.

    ## Weight Sync

    update_weight_buffer_size : int | None
        Byte size of the buffer broadcasting updated weights to the engines.

    ## RL Algorithm

    advantage_estimator : str
        Advantage estimator, e.g. ``"grpo"``.
    n_samples_per_prompt : int
        Responses sampled per prompt (the GRPO group size).
    eps_clip : float
        PPO clip lower bound.
    eps_clip_high : float
        PPO clip upper bound (asymmetric DAPO-style clipping).
    use_kl_loss : bool
        Add a per-token KL loss term against the reference model.
    kl_loss_type : str
        KL formulation, e.g. ``"low_var_kl"``.
    kl_loss_coef : float
        Coefficient of the KL loss term.
    kl_coef : float
        KL penalty coefficient applied in the reward.
    entropy_coef : float
        Entropy bonus coefficient.
    calculate_per_token_loss : bool
        Average the loss over tokens instead of over samples.
    ref_load : str
        Checkpoint the reference model is read from (for KL terms).
    use_tis : bool
        Truncated importance sampling, correcting rollout/trainer mismatch.

    ## Dynamic Sampling

    over_sampling_batch_size : int | None
        Extra prompts sampled so filter-rejected groups can be replaced (DAPO).
    dynamic_sampling_filter_path : str | None
        Import path of the predicate deciding which sample groups to keep.
    balance_data : bool
        Rebalance kept samples across data-parallel ranks.

    ## Training and Optimizer

    global_batch_size : int
        Training samples per optim step.
    lr : float
        Learning rate.
    lr_decay_style : str
        Schedule, e.g. ``"constant"`` or ``"cosine"``.
    weight_decay : float
        Weight decay.
    adam_beta1 : float
        Adam beta1.
    adam_beta2 : float
        Adam beta2.
    optimizer : str
        Optimizer name, e.g. ``"adam"``.
    use_distributed_optimizer : bool
        Shard optim state across data-parallel ranks (Megatron distributed opt).
    optimizer_cpu_offload : bool
        Keep optimizer state on CPU, trading step time for GPU memory.
    overlap_cpu_optimizer_d2h_h2d : bool
        Overlap the offloaded optimizer's device↔host copies with compute.
    use_precision_aware_optimizer : bool
        Megatron's precision-aware optimizer (lower-precision optim state).

    ## LoRA

    lora_rank : int | None
        LoRA rank; ``None`` trains full weights.
    lora_alpha : int | None
        LoRA scaling factor.
    lora_dropout : float | None
        Dropout applied to LoRA layers.
    target_modules : str | None
        Comma-separated module names LoRA adapters attach to.
    experts_shared_outer_loras : bool
        Share one outer LoRA across MoE experts instead of one per expert.
    lora_base_cpu_backup : bool
        Keep a CPU copy of the frozen base weights, freeing GPU memory.
    no_gradient_accumulation_fusion : bool
        Disable fused gradient accumulation (required by some LoRA paths).
    sglang_lora_backend : str | None
        sglang LoRA kernel backend, e.g. ``"triton"``.
    sglang_lora_use_virtual_experts : bool
        Serve MoE LoRA adapters as virtual experts in sglang.

    ## Memory and Precision

    attention_dropout : float
        Attention dropout probability.
    hidden_dropout : float
        Hidden-layer dropout probability.
    attention_softmax_in_fp32 : bool
        Compute attention softmax in fp32.
    accumulate_allreduce_grads_in_fp32 : bool
        Accumulate and all-reduce gradients in fp32.
    attention_backend : str | None
        Megatron attention kernel backend, e.g. ``"flash"``.
    no_check_for_nan_in_loss_and_grad : bool
        Skip the NaN check on loss and gradients (saves a sync per step).
    recompute_granularity : str | None
        Activation recomputation granularity (``"full"`` or ``"selective"``).
    recompute_method : str | None
        Recomputation method (``"uniform"`` or ``"block"``).
    recompute_num_layers : int | None
        Layers per recomputation chunk.
    qkv_format : str
        QKV layout for the Megatron backend (``"thd"`` or ``"bshd"``).

    ## Dynamic Batching

    use_dynamic_batch_size : bool
        Pack samples up to ``max_tokens_per_gpu`` instead of a fixed micro batch.
    micro_batch_size : int | None
        Fixed micro-batch size when dynamic batching is off; ``None`` leaves
        Miles' own default.
    max_tokens_per_gpu : int
        Token budget per GPU per micro-batch when dynamic batching is on.

    ## Eval

    eval_interval : int | None
        Run eval every N rollout steps; ``None`` disables eval.
    n_samples_per_eval_prompt : int
        Responses sampled per eval prompt.
    eval_max_response_len : int
        Max generated tokens per eval sample.
    eval_top_p : float
        Nucleus-sampling top-p for eval generation.
    eval_config : dict | str | None
        Dict written to YAML for ``--eval-config``: eval defaults + dataset list.
    skip_eval_before_train : bool
        Skip the eval pass before the first train step.

    ## Reward Model

    rm_type : str | None
        Miles built-in reward function (e.g. ``"deepscaler"``), else ``None``.

    ## Custom Functions and Hooks

    custom_rm_function : Callable | None
        Reward callable shipped by value as Miles' ``custom_rm_path``.
    custom_generate_function : Callable | None
        Replaces Miles' generate step; shipped by value, registered by path.
    custom_reward_post_process_function : Callable | None
        Applied to rewards after generation. Prefer this over a raw dotted path: a
        ``__main__`` function has no importable module name, so Miles' own
        ``importlib.import_module`` fails inside the Ray actor.
    rollout_function : Callable | str | None
        Replaces Miles' entire rollout loop (``--rollout-function-path``).
    custom_rollout_log_function : Callable | str | None
        Called with each rollout's data for logging; the gym wraps it so
        phase reporting and dashboard capture still run.
    custom_eval_rollout_log_function : Callable | str | None
        Same as above, for eval rollouts.
    custom_megatron_before_log_prob_hook : Callable | str | None
        Hook run in the Megatron trainer before log-prob computation.
    custom_megatron_before_train_step_hook : Callable | str | None
        Hook run in the Megatron trainer before each train step.

    ## Config Overrides

    extra_config : dict | None
        Primary escape hatch: dict written to YAML at ``--custom-config-path``; its keys
        become Miles args and override same-named fields.
    sglang_config : dict | str | None
        YAML at ``--sglang-config`` for structured engine config, not flat flags.
    apply_chat_template_kwargs : str | dict
        Kwargs for the tokenizer's ``apply_chat_template``, as inline JSON.
    train_env_vars : dict | str | None
        Env vars for the training processes, passed as inline JSON.
    multimodal_keys : dict | str | None
        Dataset columns holding multimodal inputs (inline JSON); auto-filled.

    ## SGLang Rollout Engine

    sglang_mem_fraction_static : float
        Fraction of GPU memory sglang reserves for weights + KV cache.
    sglang_enable_dp_attention : bool
        Enable data-parallel attention across engine ranks.
    sglang_dp_size : int | None
        Data-parallel size for the engines.
    sglang_ep_size : int | None
        Expert-parallel size for MoE models.
    sglang_enable_dp_lm_head : bool
        Data-parallel LM head (pairs with DP attention).
    sglang_disable_custom_all_reduce : bool
        Fall back to NCCL all-reduce instead of sglang's custom kernel.
    sglang_cuda_graph_bs : list[int] | None
        Batch sizes to capture CUDA graphs for.
    sglang_attention_backend : str | None
        sglang attention kernel backend, e.g. ``"triton"``. ``None`` leaves
        sglang's own selection (FlashAttention) in place.
    sglang_disable_cuda_graph : bool
        Run the engines in eager mode instead of capturing CUDA graphs.
    sglang_disable_overlap_schedule : bool
        Disable sglang's overlapped scheduler.
    sglang_disable_radix_cache : bool
        Disable prefix (radix) caching across requests.
    no_offload_train : bool
        Keep the training weights and optimizer resident instead of offloading
        them between rollout and train phases (colocated runs).
    no_offload_rollout : bool
        Keep the rollout engines resident instead of offloading them.
    sglang_moe_runner_backend : str | None
        MoE GEMM runner for the engines, e.g. ``"triton"``. ``None`` leaves
        sglang's ``auto`` selection in place.
    sglang_max_running_requests : int | None
        Cap on concurrent in-flight requests per engine.
    sglang_server_concurrency : int | None
        Cap on concurrent requests Miles sends to each engine.
    sglang_tool_call_parser : str | None
        Parser for tool-call output, e.g. ``"qwen25"``.
    sglang_reasoning_parser : str | None
        Parser for reasoning/thinking output.
    """

    # ── Launcher instructions (not Miles CLI flags) ─────────────────────────
    docker_image: str = "radixark/miles:dev-202608120325"
    gpu_type: str = "H100"
    memory: int | tuple[int, int] | None = None
    cpu: float | tuple[float, float] | None = None
    cloud: str | None = None
    region: str | None = None
    name: str = ""
    app_tags: dict = field(default_factory=dict)
    image_overlay: Callable[[modal.Image], modal.Image] | None = None
    image_run_commands: list[str] = field(default_factory=list)
    image_env: dict[str, str] = field(default_factory=dict)
    local_miles: str | None = None
    patch_files: list[str] = field(default_factory=list)
    substep_timing: Literal["auto", "off"] = "auto"

    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    miles_model_script: str = ""
    miles_model_name: str = ""
    source_hf_checkpoint: str | None = None
    megatron_conversion_hf_checkpoint: str | None = None
    metrics: MetricConfig | None = None

    # ── Cluster and parallelism ────────────────────────────────────────────
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int | None = None
    colocate: bool = True
    use_critic: bool = False
    critic_num_nodes: int | None = None
    critic_num_gpus_per_node: int | None = None

    # ── Checkpointing ───────────────────────────────────────────────────────
    hf_checkpoint: str = ""
    save: str = str(CHECKPOINTS_PATH)
    load: str = ""
    ref_load: str = ""
    megatron_to_hf_mode: str = "bridge"
    # Selects miles' megatron→HF weight mapping (e.g. "inkling"); when empty miles
    # infers it from the HF config's class name.
    model_name: str = ""
    save_interval: int = 10
    no_save_optim: bool = False

    # ── Checkpoint conversion ───────────────────────────────────────────
    # Conversion-only parallelism overrides. Launcher instructions, not CLI flags
    # (see _MILES_SKIP): torch_dist reshards on load, so the conversion layout is
    # independent of the training layout.
    conversion_tensor_model_parallel_size: int | None = None
    conversion_pipeline_model_parallel_size: int | None = None
    conversion_expert_model_parallel_size: int | None = None
    conversion_expert_tensor_parallel_size: int | None = None
    # Ephemeral disk (MiB) for the conversion container. Local staging needs room for
    # the whole torch_dist checkpoint plus the Volume's write buffer for the shard in
    # flight; the default container disk is not enough for a 276B model.
    convert_ephemeral_disk_mb: int | None = None

    # ── Fault tolerance and health checks ───────────────────────────────────
    # Miles' own argparse default; slime defaults this on instead.
    use_fault_tolerance: bool = False
    # Miles' own argparse defaults (slime uses 30/30/300).
    rollout_health_check_interval: int = 30
    rollout_health_check_timeout: int = 30
    rollout_health_check_first_wait: int = 0

    # ── Weight sync ─────────────────────────────────────────────────────────
    update_weight_buffer_size: int | None = None

    # ── Rollout and sampling ────────────────────────────────────────────────
    num_rollout: int = 1
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_shuffle: bool = True
    rollout_top_p: float = 1.0
    rollout_stop_token_ids: list[int] | None = None
    rollout_num_gpus_per_engine: int = 1
    use_miles_router: bool = False
    rollout_top_k: int | None = None
    use_rollout_routing_replay: bool = False

    # ── Parallelism ─────────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int | None = None
    expert_model_parallel_size: int | None = None
    expert_tensor_parallel_size: int | None = None
    decoder_last_pipeline_num_layers: int | None = None
    sequence_parallel: bool = False
    train_backend: str = "megatron"

    # ── Training and optimizer ──────────────────────────────────────────────
    global_batch_size: int = 16
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer: str = "adam"
    use_distributed_optimizer: bool = False
    optimizer_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    use_precision_aware_optimizer: bool = False

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    target_modules: str | None = None
    experts_shared_outer_loras: bool = False
    lora_base_cpu_backup: bool = False
    no_gradient_accumulation_fusion: bool = False
    sglang_lora_backend: str | None = None
    sglang_lora_use_virtual_experts: bool = False
    use_tis: bool = False

    # ── RL algorithm ────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.0
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    use_kl_loss: bool = False
    calculate_per_token_loss: bool = False
    rm_type: str | None = None

    # ── Dynamic sampling (DAPO) ────────────────────────────────────────────
    over_sampling_batch_size: int | None = None
    dynamic_sampling_filter_path: str | None = None
    balance_data: bool = False

    # ── Memory and precision ────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_backend: str | None = None
    no_check_for_nan_in_loss_and_grad: bool = False
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int | None = None
    qkv_format: str = "thd"

    # ── Dynamic batching ────────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    micro_batch_size: int | None = None
    max_tokens_per_gpu: int = 9216

    # ── Eval ────────────────────────────────────────────────────────────────
    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 16384
    eval_top_p: float = 1.0
    eval_config: dict | str | None = None
    skip_eval_before_train: bool = False

    # ── SGLang rollout engine ──────────────────────────────────────────────
    sglang_mem_fraction_static: float = 0.75
    sglang_enable_dp_attention: bool = False
    sglang_dp_size: int | None = None
    sglang_ep_size: int | None = None
    sglang_enable_dp_lm_head: bool = False
    sglang_disable_custom_all_reduce: bool = False
    sglang_cuda_graph_bs: list[int] | None = None
    sglang_attention_backend: str | None = None
    sglang_disable_cuda_graph: bool = False
    sglang_disable_overlap_schedule: bool = False
    sglang_disable_radix_cache: bool = False
    no_offload_train: bool = False
    no_offload_rollout: bool = False
    sglang_moe_runner_backend: str | None = None
    sglang_max_running_requests: int | None = None
    sglang_server_concurrency: int | None = None
    sglang_tool_call_parser: str | None = None
    sglang_reasoning_parser: str | None = None

    # ── Config overrides ────────────────────────────────────────────────────
    extra_config: dict | None = None
    sglang_config: dict | str | None = None
    apply_chat_template_kwargs: str | dict = ""
    train_env_vars: dict | str | None = None
    multimodal_keys: dict | str | None = None

    # ── Custom functions and hooks ──────────────────────────────────────────
    custom_rm_function: Callable | None = None
    custom_generate_function: Callable | None = None
    custom_reward_post_process_function: Callable | None = None
    rollout_function: Callable | str | None = None
    custom_rollout_log_function: Callable | str | None = None
    custom_eval_rollout_log_function: Callable | str | None = None
    custom_megatron_before_log_prob_hook: Callable | str | None = None
    custom_megatron_before_train_step_hook: Callable | str | None = None

    # ── Per-sample execution tracing (dashboard timeline) ───────────────────
    # When True, the rollout recorder attaches miles' per-sample trace (the
    # generate/reward/tool-call timeline) to the first `trace_sample_limit`
    # samples of each rollout. Off by default — traces inflate payloads, so
    # sampling keeps the added volume well under 1%. Not a miles CLI flag.
    train_function_kwargs: dict[str, Any] = field(default_factory=dict)
    capture_trace: bool = False
    trace_sample_limit: int = 16

    # ── Validators ───────────────────────────────────────────────────────────

    _SKIP_FIELDS: ClassVar[frozenset[str]] = frozenset(_MILES_SKIP)

    @model_validator(mode="after")
    def _resolve_callable_paths(self) -> "MilesRecipe":
        cfg = dict(self.extra_config) if isinstance(self.extra_config, dict) else {}
        for field_name, config_key in _HOOK_PATH_CONFIG_KEYS.items():
            native_key = config_key.removeprefix("training_gym_")
            native_value = cfg.pop(native_key, None)
            value = getattr(self, field_name)
            if cfg.get(config_key):
                continue
            if value is None:
                if isinstance(native_value, str) and native_value.strip():
                    cfg[config_key] = native_value
                continue
            if isinstance(value, str):
                cfg[config_key] = value
            else:
                cfg[config_key] = self._callable_path(value)
        if cfg != (self.extra_config or {}):
            object.__setattr__(self, "extra_config", cfg)
        return self

    @model_validator(mode="after")
    def _validate_gpu_allocation(self) -> "MilesRecipe":
        resolve_gpu_allocation(self)
        return self

    # ── Container → miles flag converters ────────────────────────────────────

    @classmethod
    def _dataset_to_fields(cls, ds: "DatasetConfig") -> dict[str, Any]:
        fields = super()._dataset_to_fields(ds)
        if getattr(ds, "multimodal_keys", None):
            fields["multimodal_keys"] = ds.multimodal_keys
        return fields

    @staticmethod
    def _model_to_fields(m: ModelConfig) -> dict[str, Any]:
        fields: dict[str, Any] = {"hf_checkpoint": m.model_path or m.model_name}
        arch = getattr(m, "architecture", None)
        if arch is None:
            return fields
        fields.update(
            {
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "ffn_hidden_size": arch.ffn_hidden_size,
                "num_attention_heads": arch.num_attention_heads,
                "group_query_attention": arch.group_query_attention,
                "num_query_groups": arch.num_query_groups,
                "kv_channels": arch.kv_channels,
                "vocab_size": arch.vocab_size,
                "normalization": arch.normalization,
                "norm_epsilon": arch.norm_epsilon,
                "swiglu": arch.swiglu,
                "disable_bias_linear": arch.disable_bias_linear,
                "qk_layernorm": arch.qk_layernorm,
                "untie_embeddings_and_output_weights": arch.untie_embeddings_and_output_weights,
                "no_masked_softmax_fusion": arch.no_masked_softmax_fusion,
                "multi_latent_attention": arch.multi_latent_attention,
                "use_rotary_position_embeddings": arch.use_rotary_position_embeddings,
                "rotary_base": arch.rotary_base,
                "rotary_scaling_factor": arch.rotary_scaling_factor,
                "mscale": arch.mscale,
                "mscale_all_dim": arch.mscale_all_dim,
                "no_rope_fusion": arch.no_rope_fusion,
            }
        )
        optional = {
            "kv_lora_rank": arch.kv_lora_rank,
            "qk_head_dim": arch.qk_head_dim,
            "qk_pos_emb_head_dim": arch.qk_pos_emb_head_dim,
            "v_head_dim": arch.v_head_dim,
            "num_experts": arch.num_experts,
            "moe_layer_freq": arch.moe_layer_freq,
            "moe_ffn_hidden_size": arch.moe_ffn_hidden_size,
            "moe_shared_expert_intermediate_size": arch.moe_shared_expert_intermediate_size,
            "moe_router_topk": arch.moe_router_topk,
            "moe_router_pre_softmax": arch.moe_router_pre_softmax,
            "moe_router_score_function": arch.moe_router_score_function,
            "moe_router_enable_expert_bias": arch.moe_router_enable_expert_bias,
            "moe_router_load_balancing_type": arch.moe_router_load_balancing_type,
            "moe_token_dispatcher_type": arch.moe_token_dispatcher_type,
            "moe_router_bias_update_rate": arch.moe_router_bias_update_rate,
            "moe_router_group_topk": arch.moe_router_group_topk,
            "moe_router_num_groups": arch.moe_router_num_groups,
            "moe_router_topk_scaling_factor": arch.moe_router_topk_scaling_factor,
            "moe_token_drop_policy": arch.moe_token_drop_policy,
            "moe_router_dtype": arch.moe_router_dtype,
            "moe_aux_loss_coeff": arch.moe_aux_loss_coeff,
            "spec": arch.megatron_spec,
            "rotary_percent": arch.rotary_percent
            if arch.rotary_percent != 1.0
            else None,
        }
        fields.update({k: v for k, v in optional.items() if v not in (None, "", 0)})
        for key in ("moe_aux_loss_coeff", "moe_router_bias_update_rate"):
            if optional[key] is not None:
                fields[key] = optional[key]
        for key in (
            "moe_grouped_gemm",
            "moe_shared_expert_gate",
            "moe_permute_fusion",
            "moe_router_pre_softmax",
            "moe_router_enable_expert_bias",
            "apply_layernorm_1p",
            "use_gated_attention",
            "attention_output_gate",
        ):
            if getattr(arch, key):
                fields[key] = True
        return fields

    def validate_model_parallelism(self, model: ModelConfig) -> None:
        validate_num_experts_divisible_by_expert_parallel_size(self, model)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(
        self,
        dataset: DatasetConfig | None = None,
        model: ModelConfig | None = None,
    ) -> dict[str, Any]:
        fields = self._field_values()
        if model is not None:
            self.validate_model_parallelism(model)
            for k, v in self._model_to_fields(model).items():
                if k == "hf_checkpoint":
                    if fields.get(k):
                        continue
                elif self.miles_model_script:
                    # The model script already sources the arch args.
                    continue
                elif self.miles_model_name:
                    continue
                fields[k] = v
        if dataset is not None:
            fields.update(self._dataset_to_fields(dataset))
        if self.metrics is not None:
            fields.update(self._metrics_to_fields(self.metrics))
        out = self._emit_fields(fields)
        for src, dst in _HOOK_PATH_FLAGS.items():
            if src in _HOOK_WRAPPER_PATHS:
                out[dst] = _HOOK_WRAPPER_PATHS[src]
                continue
            if path := self._path_or_callable_path(fields.get(src)):
                out[dst] = path
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def get_base_recipe(cls, model_config: ModelConfig) -> "MilesRecipe | None":
        from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
            Gemma4_26B_A4B_Recipe,
        )
        from modal_training_gym.train_recipes.miles_recipe.inkling import (
            Inkling_Small_LoRA_Recipe,
            Inkling_Small_Recipe,
        )
        from modal_training_gym.train_recipes.miles_recipe.moonlight_16b_a3b import (
            Moonlight_16B_A3B_Recipe,
        )
        from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
            Qwen3_5_4b_Miles_Recipe,
        )

        if model_config.model_name == "Qwen/Qwen3.5-4B":
            return Qwen3_5_4b_Miles_Recipe()
        if model_config.model_name == "moonshotai/Moonlight-16B-A3B-Instruct":
            return Moonlight_16B_A3B_Recipe()
        if model_config.model_name == "google/gemma-4-26B-A4B-it":
            return Gemma4_26B_A4B_Recipe()
        if model_config.model_name == "thinkingmachines/Inkling-Small":
            if issubclass(cls, Inkling_Small_LoRA_Recipe):
                return Inkling_Small_LoRA_Recipe()
            return Inkling_Small_Recipe()
        return None

    def download_model(self) -> None:
        from modal_training_gym.frameworks.miles.modal_helpers.utils import (
            resolve_checkpoint_ref,
        )

        ref = self.source_hf_checkpoint or self.hf_checkpoint
        if ref:
            resolve_checkpoint_ref(ref, local_files_only=False)

    def post_process_model(self) -> None:
        pass

    def post_process_data(self) -> None:
        pass
