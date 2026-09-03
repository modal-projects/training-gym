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
    """Miles training and Modal resource settings.

    Args:

        recipe_type:
            Internal discriminator fixed to Miles.
        name:
            Modal app title. The launcher derives it from the class when empty.
        app_tags:
            Extra tags merged into the Modal app metadata for the dashboard.

        docker_image:
            Registry reference for the Miles image every container runs.
        environment:
            Training-container environment variables such as Megatron
            ``PYTHONPATH`` and NCCL settings.
        async_mode:
            Run Miles' ``train_async.py`` so rollout generation and training overlap.
        metrics:
            Metric tracker settings; expands to Miles' W&B-compatible flags.
        image_overlay:
            Function that modifies the Modal image.
        local_miles:
            Local Miles checkout mounted over the image copy without rebuilding it.
        memory:
            Modal Function memory request/limit in MiB.
        cpu:
            Modal Function CPU request/limit in cores per container.
        cloud:
            Modal cloud provider to pin the cluster to. Defaults to ``"oci"``:
            miles multi-node runs hang on AWS EFA hosts (see the field), so
            they stay on Mellanox/IB fleets until that is fixed.
        region:
            Modal region to pin the cluster to.
        miles_model_script:
            Script in the Miles repository sourced for ``MODEL_ARGS`` instead of
            model-architecture flags.
        miles_model_name:
            Name accepted by Miles' ``model_args_utils.py``.
        source_hf_checkpoint:
            Source checkpoint when it differs from the model's own.
        megatron_conversion_hf_checkpoint:
            HF weights used for the HF→Megatron conversion instead of the model's own.
        patch_files:
            Local patch scripts applied to Miles/Megatron sources at image build.
        image_run_commands:
            Extra shell commands run while building the image.
        image_env:
            Extra env vars baked into the image.
        train_function_kwargs:
            Additional Modal Function keyword arguments for the training function.
        capture_trace:
            Attach sampled per-request execution traces to recorded rollouts.
        trace_sample_limit:
            Maximum traced samples per rollout when ``capture_trace`` is enabled.

        gpu_type:
            Modal GPU type for every node.
        colocate:
            Trainer and rollout engines share GPUs; ``False`` gives each its own.
        actor_num_nodes:
            Megatron actor nodes.
        actor_num_gpus_per_node:
            GPUs per actor node.
        rollout_num_gpus:
            Rollout-engine GPUs when disaggregated; ``None`` lets the resolver size it.
        rollout_num_gpus_per_engine:
            GPUs and tensor-parallel size per SGLang engine.
        train_backend:
            Training backend.
        tensor_model_parallel_size:
            Megatron tensor-parallel size for the actor.
        pipeline_model_parallel_size:
            Megatron pipeline-parallel size for the actor.
        context_parallel_size:
            Megatron context-parallel size; multiplies the effective context length.
        expert_model_parallel_size:
            Expert-parallel size for MoE; must divide the model's ``num_experts``.
        expert_tensor_parallel_size:
            Tensor-parallel size within each expert.
        decoder_last_pipeline_num_layers:
            Layers placed on the last pipeline stage, to rebalance an uneven split.
        sequence_parallel:
            Megatron sequence parallelism. Requires tensor parallelism greater than one.
        use_critic:
            Train a separate critic model for PPO. GRPO does not use one.
        critic_num_nodes:
            Nodes for the critic when ``use_critic`` is set.
        critic_num_gpus_per_node:
            GPUs per critic node.

        num_rollout:
            Training and rollout steps for the run.
        start_rollout_id:
            Rollout step to start counting from. ``None`` continues from the
            iteration stored in ``load``; ``TrainConfig(checkpoint=...)`` sets
            ``0`` so ``num_rollout`` counts the steps this run performs.
        rollout_batch_size:
            Prompts per rollout step, each expanded into a group of responses.
        rollout_max_response_len:
            Max generated tokens per sample.
        rollout_temperature:
            Sampling temperature for rollout generation.
        rollout_shuffle:
            Shuffle the prompt dataset between epochs.
        rollout_top_p:
            Nucleus-sampling top-p for rollout generation.
        rollout_stop_token_ids:
            Extra token ids that terminate generation.
        use_miles_router:
            Route rollout requests through Miles' router instead of directly to
            engines.
        rollout_top_k:
            Top-k for rollout generation; ``None`` leaves Miles' own default.
        use_rollout_routing_replay:
            Reuse the rollout's MoE expert routing in training.

        hf_checkpoint:
            Checkpoint trained from; normally set from the attached ``ModelConfig``.
        save:
            Checkpoint output directory on the mounted ``/checkpoints`` volume.
        save_interval:
            Save a checkpoint every N rollout steps.
        load:
            Directory to resume from; empty starts from the converted HF weights.
        no_save_optim:
            Omit optimizer state from checkpoints. The resulting checkpoints cannot
            resume the optimizer exactly.
        megatron_to_hf_mode:
            Export mode for saved Megatron checkpoints; empty disables the export.

        use_fault_tolerance:
            Enable Miles' fault tolerance to recover from worker failures.
        rollout_health_check_interval:
            Seconds between rollout engine ``/health_generate`` checks.
        rollout_health_check_timeout:
            Seconds to wait for ``/health_generate`` before killing the engine.
        rollout_health_check_first_wait:
            Initial health-check delay in seconds before checking
            ``/health_generate``. DeepGEMM compilation may require a longer delay.

        update_weight_buffer_size:
            Byte size of the buffer broadcasting updated weights to the engines.

        advantage_estimator:
            Advantage estimator.
        n_samples_per_prompt:
            Responses sampled per prompt as one GRPO group.
        eps_clip:
            PPO clip lower bound.
        eps_clip_high:
            Upper PPO clip bound for asymmetric DAPO clipping.
        use_kl_loss:
            Add a per-token KL loss term against the reference model.
        kl_loss_type:
            KL formulation.
        kl_loss_coef:
            Coefficient of the KL loss term.
        kl_coef:
            KL penalty coefficient applied in the reward.
        entropy_coef:
            Entropy bonus coefficient.
        calculate_per_token_loss:
            Average the loss over tokens instead of over samples.
        ref_load:
            Checkpoint read by the reference model for KL terms.
        use_tis:
            Correct rollout and trainer mismatch with truncated importance sampling.

        over_sampling_batch_size:
            Extra DAPO prompts sampled to replace filtered groups.
        dynamic_sampling_filter_path:
            Import path of the predicate deciding which sample groups to keep.
        balance_data:
            Rebalance kept samples across data-parallel ranks.

        global_batch_size:
            Training samples per optim step.
        lr:
            Learning rate.
        lr_decay_style:
            Learning-rate schedule.
        weight_decay:
            Weight decay.
        adam_beta1:
            Adam beta1.
        adam_beta2:
            Adam beta2.
        optimizer:
            Optimizer name.
        use_distributed_optimizer:
            Shard optimizer state across data-parallel ranks with Megatron's
            distributed optimizer.
        optimizer_cpu_offload:
            Keep optimizer state on CPU to reduce GPU memory use at the cost of
            slower steps.
        overlap_cpu_optimizer_d2h_h2d:
            Overlap the offloaded optimizer's device↔host copies with compute.
        use_precision_aware_optimizer:
            Use Megatron's precision-aware optimizer with lower-precision state.

        lora_rank:
            LoRA rank; ``None`` trains full weights.
        lora_alpha:
            LoRA scaling factor.
        lora_dropout:
            Dropout applied to LoRA layers.
        target_modules:
            Comma-separated module names LoRA adapters attach to.
        experts_shared_outer_loras:
            Share one outer LoRA across MoE experts instead of one per expert.
        lora_base_cpu_backup:
            Keep frozen base weights on CPU to free GPU memory.
        no_gradient_accumulation_fusion:
            Disable fused gradient accumulation for incompatible LoRA paths.
        sglang_lora_backend:
            SGLang LoRA kernel backend.
        sglang_lora_use_virtual_experts:
            Serve MoE LoRA adapters as virtual experts in sglang.

        attention_dropout:
            Attention dropout probability.
        hidden_dropout:
            Hidden-layer dropout probability.
        attention_softmax_in_fp32:
            Compute attention softmax in fp32.
        accumulate_allreduce_grads_in_fp32:
            Accumulate and all-reduce gradients in fp32.
        attention_backend:
            Megatron attention kernel backend.
        no_check_for_nan_in_loss_and_grad:
            Skip the NaN check on loss and gradients to avoid a synchronization per
            step.
        recompute_granularity:
            Activation recomputation granularity: ``"full"`` or ``"selective"``.
        recompute_method:
            Recomputation method: ``"uniform"`` or ``"block"``.
        recompute_num_layers:
            Layers per recomputation chunk.
        qkv_format:
            QKV layout for the Megatron backend: ``"thd"`` or ``"bshd"``.

        use_dynamic_batch_size:
            Pack samples up to ``max_tokens_per_gpu`` instead of a fixed micro batch.
        micro_batch_size:
            Fixed micro-batch size when dynamic batching is off; ``None`` leaves
            Miles' own default.
        max_tokens_per_gpu:
            Token budget per GPU per micro-batch when dynamic batching is on.

        eval_interval:
            Run eval every N rollout steps; ``None`` disables eval.
        n_samples_per_eval_prompt:
            Responses sampled per eval prompt.
        eval_max_response_len:
            Max generated tokens per eval sample.
        eval_top_p:
            Nucleus-sampling top-p for eval generation.
        eval_config:
            Evaluation defaults and datasets written to ``--eval-config`` as YAML.
        skip_eval_before_train:
            Skip the eval pass before the first train step.

        rm_type:
            Built-in reward function name. Leave unset for a custom reward.

        custom_rm_function:
            Reward callable shipped by value as Miles' ``custom_rm_path``.
        custom_generate_function:
            Custom Miles generation step shipped by value.
        custom_reward_post_process_function:
            Function applied to rewards after generation and shipped by value.
        rollout_function:
            Custom rollout loop passed through ``--rollout-function-path``.
        custom_rollout_log_function:
            Function called with each rollout's data after dashboard and phase
            reporting.
        custom_eval_rollout_log_function:
            Function called with each evaluation rollout's data.
        custom_megatron_before_log_prob_hook:
            Hook run in the Megatron trainer before log-prob computation.
        custom_megatron_before_train_step_hook:
            Hook run in the Megatron trainer before each train step.

        extra_config:
            Custom configuration written to YAML at ``--custom-config-path``. Keys
            become Miles arguments and override same-named fields.
        sglang_config:
            SGLang engine settings written to ``--sglang-config`` as YAML.
        apply_chat_template_kwargs:
            Keyword arguments for tokenizer ``apply_chat_template``, passed as JSON.
        train_env_vars:
            Env vars for the training processes, passed as inline JSON.
        multimodal_keys:
            Multimodal dataset columns passed as JSON.

        sglang_mem_fraction_static:
            Fraction of GPU memory sglang reserves for weights + KV cache.
        sglang_enable_dp_attention:
            Enable data-parallel attention across engine ranks.
        sglang_dp_size:
            Data-parallel size for the engines.
        sglang_ep_size:
            Expert-parallel size for MoE models.
        sglang_enable_dp_lm_head:
            Data-parallel LM head paired with DP attention.
        sglang_disable_custom_all_reduce:
            Fall back to NCCL all-reduce instead of sglang's custom kernel.
        sglang_cuda_graph_bs:
            Batch sizes to capture CUDA graphs for.
        sglang_attention_backend:
            SGLang attention kernel backend. The server selects one when unset.
        sglang_disable_cuda_graph:
            Run the engines in eager mode instead of capturing CUDA graphs.
        sglang_disable_overlap_schedule:
            Disable sglang's overlapped scheduler.
        sglang_disable_radix_cache:
            Disable prefix (radix) caching across requests.
        no_offload_train:
            Keep training weights and optimizer resident between rollout and training
            phases for colocated runs.
        no_offload_rollout:
            Keep the rollout engines resident instead of offloading them.
        sglang_moe_runner_backend:
            SGLang MoE GEMM runner. The server selects one when unset.
        sglang_max_running_requests:
            Cap on concurrent in-flight requests per engine.
        sglang_server_concurrency:
            Cap on concurrent requests Miles sends to each engine.
        sglang_tool_call_parser:
            Tool-call output parser.
        sglang_reasoning_parser:
            Parser for reasoning/thinking output.
    """

    # ── Launcher instructions (not Miles CLI flags) ─────────────────────────
    docker_image: str = "radixark/miles:dev-202608120325"
    gpu_type: str = "H100"
    memory: int | tuple[int, int] | None = None
    cpu: float | tuple[float, float] | None = None
    # Every miles recipe here is multi-node RDMA. On AWS EFA hosts the first
    # pipeline-parallel shape exchange can hang for the full collective timeout
    # (Nemotron-3-Ultra, p5en.48xlarge); the same code trains every step on
    # Mellanox/IB hosts. Pinned off AWS until that is root-caused. A recipe or
    # launch can still pass cloud= to override.
    cloud: str | None = "oci"
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
    start_rollout_id: int | None = None
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 2
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
        from modal_training_gym.train_recipes.miles_recipe.nemotron3_ultra_550b_a55b import (
            Nemotron3_Ultra_550B_A55B_Recipe,
        )
        from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
            Qwen3_5_4B_Miles_Recipe,
        )

        if model_config.model_name == "Qwen/Qwen3.5-4B":
            return Qwen3_5_4B_Miles_Recipe()
        if model_config.model_name == "moonshotai/Moonlight-16B-A3B-Instruct":
            return Moonlight_16B_A3B_Recipe()
        if model_config.model_name == "google/gemma-4-26B-A4B-it":
            return Gemma4_26B_A4B_Recipe()
        if model_config.model_name == "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16":
            return Nemotron3_Ultra_550B_A55B_Recipe()
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
