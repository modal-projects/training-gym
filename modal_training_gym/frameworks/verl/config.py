"""Configuration for verl (Volcengine Verl) RLVR/GRPO training on Modal.

`VerlConfig` holds the training knobs; `cli_args()` emits the Hydra-style
overrides (`section.subsection.key=value`) that `verl.trainer.main_ppo`
expects. Containers (`dataset`, `model`, `wandb`) are interpreted explicitly
because verl's flag names are deep dotted paths that don't match the common
config attribute names 1:1.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modal_training_gym.common import GPUType

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Model
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

DATA_PATH = Path("/data")
MODELS_PATH = Path("/models")

# Derived sub-paths under MODELS_PATH.
HF_MODELS_SUBDIR = "hf_models"
MCORE_MODELS_SUBDIR = "mcore_models"
TRAINING_CHECKPOINT_SUBDIR = "training_checkpoints"

# ── Types ─────────────────────────────────────────────────────────────────────

class VerlModalConfig:
    """Modal infrastructure for verl — GPU family + image registry override."""

    gpu: GPUType = "H100"
    # verl ships pre-built images pinned to a vLLM version.
    image: str = "verlai/verl:vllm011.latest"
    # Git ref of verl to `git clone` + install. Pin to a commit for reproducibility.
    verl_git_url: str = "https://github.com/volcengine/verl"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class VerlConfig:
    """verl (RLVR / GRPO) training configuration.

    Subclass and override class attributes. `cli_args()` assembles Hydra-style
    `key=value` overrides appended to `verl.trainer.main_ppo`.

    Attach `dataset` / `model` / `wandb` as containers; they're read
    explicitly in `cli_args()` — this framework doesn't blindly merge
    container `.to_fields()` because verl's flag vocabulary is mostly deep
    dotted paths that don't line up with the common config attr names.
    """

    # ── Containers ───────────────────────────────────────────────────────────
    dataset: "DatasetConfig | None" = None
    model: "Model | None" = None
    wandb: "WandbConfig | None" = None

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 4
    gpus_per_node: int = 8

    # ── Algorithm ────────────────────────────────────────────────────────────
    adv_estimator: str = "grpo"
    use_kl_in_reward: bool = False

    # ── Data ─────────────────────────────────────────────────────────────────
    max_prompt_length: int = 768
    max_response_length: int = 2048
    train_batch_size: int = 128
    filter_overlong_prompts: bool = True
    truncation: str = "error"
    return_raw_chat: bool = True
    prompt_key: str = "prompt"

    # ── Actor ────────────────────────────────────────────────────────────────
    actor_strategy: str = "megatron"
    actor_lr: float = 1e-6
    ppo_mini_batch_size: int = 128
    ppo_max_token_len_per_gpu: int = 2816 * 4
    actor_use_dynamic_bsz: bool = True
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    entropy_coeff: float = 0.0
    loss_agg_mode: str = "token-mean"
    use_fused_kernels: bool = True
    # Save only the model weights in actor checkpoints (no optimizer state).
    actor_save_contents: list[str] = ["model"]

    # ── Actor parallelism (Megatron) ─────────────────────────────────────────
    actor_tp_size: int = 8
    actor_pp_size: int = 1
    actor_cp_size: int = 1
    actor_ep_size: int = 1
    actor_etp_size: int = 1  # expert tensor parallel
    actor_dtype: str = "bfloat16"
    actor_param_offload: bool = False
    actor_grad_offload: bool = False
    actor_optimizer_offload: bool = False

    # ── Rollout (vLLM) ───────────────────────────────────────────────────────
    rollout_name: str = "vllm"
    rollout_mode: str = "async"
    n_rollouts_per_prompt: int = 4  # actor_rollout_ref.rollout.n
    rollout_tp_size: int = 4
    rollout_dp_size: int = 1
    rollout_ep_size: int = 1
    rollout_dtype: str = "bfloat16"
    rollout_gpu_memory_util: float = 0.25
    rollout_log_prob_max_token_len: int = 2048 * 18
    rollout_log_prob_use_dynamic_bsz: bool = True
    rollout_enable_chunked_prefill: bool = True
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_top_k: int = -1
    val_temperature: float = 1.0
    val_top_p: float = 0.7
    val_top_k: int = -1
    val_n: int = 1
    val_do_sample: bool = True

    # ── Reference model (Megatron) ───────────────────────────────────────────
    ref_tp_size: int = 8
    ref_pp_size: int = 1
    ref_cp_size: int = 1
    ref_ep_size: int = 1
    ref_etp_size: int = 1
    ref_param_offload: bool = True
    ref_dtype: str = "bfloat16"
    ref_log_prob_micro_batch_size_per_gpu: int = 64
    ref_log_prob_max_token_len: int = 2048 * 18
    ref_log_prob_use_dynamic_bsz: bool = True

    # ── Trainer ──────────────────────────────────────────────────────────────
    critic_warmup: int = 0
    test_freq: int = 5
    save_freq: int = -1
    resume_mode: str = "auto"

    # ── Reward function ──────────────────────────────────────────────────────
    # Resolution order (highest priority first):
    #   1. `reward_function_source`: Python source code written to the rewards
    #      volume by `app.upload_reward`; file can be executed by verl as a
    #      standalone module (also plain-importable from Python / notebooks).
    #   2. `reward_function_path`: an absolute path inside the image.
    #   3. The shipped `modal_training_gym.frameworks.verl.gsm8k_reward` module.
    reward_function_source: str = ""
    reward_function_path: str = ""
    reward_function_name: str = "compute_reward"

    # ── Extra free-form overrides appended to the verl command. ──────────────
    extra_overrides: list[str] = []

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging.
    app_tags: dict = {}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Path helpers ─────────────────────────────────────────────────────────

    def hf_model_dir(self) -> str:
        assert self.model is not None and self.model.hf_checkpoint is not None
        return str(MODELS_PATH / HF_MODELS_SUBDIR / self.model.hf_checkpoint)

    def mcore_model_dir(self) -> str:
        assert self.model is not None and self.model.hf_checkpoint is not None
        return str(MODELS_PATH / MCORE_MODELS_SUBDIR / self.model.hf_checkpoint)

    def training_checkpoint_dir(self) -> str:
        assert self.model is not None and self.model.hf_checkpoint is not None
        return str(MODELS_PATH / TRAINING_CHECKPOINT_SUBDIR / self.model.hf_checkpoint)

    # ── CLI assembly ─────────────────────────────────────────────────────────

    def cli_args(self, *, reward_function_path: str) -> list[str]:
        """Build the list of Hydra overrides appended to `verl.trainer.main_ppo`.

        The launcher prepends the fixed
        `python -u -m verl.trainer.main_ppo --config-path=config
        --config-name=ppo_megatron_trainer.yaml` portion.
        """
        assert self.model is not None, "VerlConfig.model must be set"
        assert self.dataset is not None, "VerlConfig.dataset must be set"

        train_files = self.dataset.prompt_data
        val_files = (
            self.dataset.eval_prompt_data
            if isinstance(self.dataset.eval_prompt_data, str)
            else self.dataset.prompt_data
        )
        # Common: eval_prompt_data may be ["name", "path"] (SLIME shape) or
        # a plain string; fall back to train files if absent.
        if isinstance(self.dataset.eval_prompt_data, list) and self.dataset.eval_prompt_data:
            val_files = self.dataset.eval_prompt_data[-1]

        project_name = self.wandb.project if self.wandb else "verl"
        experiment_name = (
            self.wandb.exp_name or self.wandb.group if self.wandb else "verl_run"
        )
        trainer_loggers = '["console","wandb"]' if self.wandb else '["console"]'

        overrides: list[str] = [
            # ─ Algorithm / data ──────────────────────────────────────────────
            f"algorithm.adv_estimator={self.adv_estimator}",
            f"algorithm.use_kl_in_reward={self.use_kl_in_reward}",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.prompt_key={self.prompt_key}",
            f"data.return_raw_chat={self.return_raw_chat}",
            f"data.max_prompt_length={self.max_prompt_length}",
            f"data.max_response_length={self.max_response_length}",
            f"data.filter_overlong_prompts={self.filter_overlong_prompts}",
            f"data.truncation={self.truncation}",
            f"data.train_batch_size={self.train_batch_size}",
            # ─ Base model (HF view) ──────────────────────────────────────────
            f"actor_rollout_ref.model.path={self.hf_model_dir()}",
            f"actor_rollout_ref.model.use_fused_kernels={self.use_fused_kernels}",
            # ─ Resource layout ───────────────────────────────────────────────
            f"trainer.nnodes={self.n_nodes}",
            f"trainer.n_gpus_per_node={self.gpus_per_node}",
            # ─ Actor ─────────────────────────────────────────────────────────
            f"actor_rollout_ref.actor.strategy={self.actor_strategy}",
            f"actor_rollout_ref.actor.optim.lr={self.actor_lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={self.ppo_max_token_len_per_gpu}",
            f"actor_rollout_ref.actor.use_dynamic_bsz={self.actor_use_dynamic_bsz}",
            f"actor_rollout_ref.actor.use_kl_loss={self.use_kl_loss}",
            f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
            f"actor_rollout_ref.actor.kl_loss_type={self.kl_loss_type}",
            f"actor_rollout_ref.actor.entropy_coeff={self.entropy_coeff}",
            f"actor_rollout_ref.actor.loss_agg_mode={self.loss_agg_mode}",
            (
                "actor_rollout_ref.actor.checkpoint.save_contents="
                + "[" + ",".join(f'"{x}"' for x in self.actor_save_contents) + "]"
            ),
            # ─ Actor Megatron parallelism + checkpointing ────────────────────
            f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={self.actor_pp_size}",
            f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={self.actor_tp_size}",
            f"actor_rollout_ref.actor.megatron.context_parallel_size={self.actor_cp_size}",
            f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={self.actor_ep_size}",
            f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={self.actor_etp_size}",
            f"actor_rollout_ref.actor.megatron.dtype={self.actor_dtype}",
            f"actor_rollout_ref.actor.megatron.param_offload={self.actor_param_offload}",
            f"actor_rollout_ref.actor.megatron.grad_offload={self.actor_grad_offload}",
            f"actor_rollout_ref.actor.megatron.optimizer_offload={self.actor_optimizer_offload}",
            "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True",
            f"actor_rollout_ref.actor.megatron.dist_checkpointing_path={self.mcore_model_dir()}",
            # ─ Rollout (vLLM) ────────────────────────────────────────────────
            f"actor_rollout_ref.rollout.name={self.rollout_name}",
            f"actor_rollout_ref.rollout.mode={self.rollout_mode}",
            f"actor_rollout_ref.rollout.n={self.n_rollouts_per_prompt}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.rollout_tp_size}",
            f"actor_rollout_ref.rollout.data_parallel_size={self.rollout_dp_size}",
            f"actor_rollout_ref.rollout.expert_parallel_size={self.rollout_ep_size}",
            f"actor_rollout_ref.rollout.dtype={self.rollout_dtype}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.rollout_gpu_memory_util}",
            f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={self.rollout_log_prob_use_dynamic_bsz}",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={self.rollout_log_prob_max_token_len}",
            f"actor_rollout_ref.rollout.enable_chunked_prefill={self.rollout_enable_chunked_prefill}",
            f"actor_rollout_ref.rollout.temperature={self.rollout_temperature}",
            f"actor_rollout_ref.rollout.top_p={self.rollout_top_p}",
            f"actor_rollout_ref.rollout.top_k={self.rollout_top_k}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={self.val_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.top_p={self.val_top_p}",
            f"actor_rollout_ref.rollout.val_kwargs.top_k={self.val_top_k}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={self.val_do_sample}",
            f"actor_rollout_ref.rollout.val_kwargs.n={self.val_n}",
            # ─ Reference model (Megatron) ────────────────────────────────────
            f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={self.ref_pp_size}",
            f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={self.ref_tp_size}",
            f"actor_rollout_ref.ref.megatron.context_parallel_size={self.ref_cp_size}",
            f"actor_rollout_ref.ref.megatron.expert_model_parallel_size={self.ref_ep_size}",
            f"actor_rollout_ref.ref.megatron.expert_tensor_parallel_size={self.ref_etp_size}",
            f"actor_rollout_ref.ref.megatron.param_offload={self.ref_param_offload}",
            "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True",
            f"actor_rollout_ref.ref.megatron.dist_checkpointing_path={self.mcore_model_dir()}",
            f"actor_rollout_ref.ref.megatron.dtype={self.ref_dtype}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={self.ref_log_prob_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={self.ref_log_prob_use_dynamic_bsz}",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={self.ref_log_prob_max_token_len}",
            # ─ Trainer ───────────────────────────────────────────────────────
            f"trainer.critic_warmup={self.critic_warmup}",
            f"trainer.logger={trainer_loggers}",
            f"trainer.project_name={project_name}",
            f"trainer.experiment_name={experiment_name}",
            f"trainer.test_freq={self.test_freq}",
            f"trainer.default_local_dir={self.training_checkpoint_dir()}",
            f"trainer.save_freq={self.save_freq}",
            f"trainer.resume_mode={self.resume_mode}",
            # ─ Reward ────────────────────────────────────────────────────────
            f"custom_reward_function.path={reward_function_path}",
            f"custom_reward_function.name={self.reward_function_name}",
        ]
        overrides.extend(self.extra_overrides)
        return overrides
