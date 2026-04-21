"""Configuration for verl (Volcengine Verl) RLVR/GRPO training on Modal.

`VerlConfig` holds the training knobs; `cli_args()` emits the Hydra-style
overrides (`section.subsection.key=value`) that `verl.trainer.main_ppo`
expects. Containers (`dataset`, `model`, `wandb`) are interpreted explicitly
because verl's flag names are deep dotted paths that don't match the common
config attribute names 1:1.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from modal_training_gym.common import GPUType
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

DATA_PATH = Path("/data")
MODELS_PATH = Path("/models")

# Derived sub-paths under MODELS_PATH.
HF_MODELS_SUBDIR = "hf_models"
MCORE_MODELS_SUBDIR = "mcore_models"
TRAINING_CHECKPOINT_SUBDIR = "training_checkpoints"

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class VerlFrameworkConfig:
    """verl (RLVR / GRPO) configuration, including Modal infrastructure.

    Subclass and override class attributes. `cli_args()` assembles Hydra-style
    `key=value` overrides appended to `verl.trainer.main_ppo`.

    Attach `dataset` / `model` / `wandb` as containers; they're read
    explicitly in `cli_args()` — this framework doesn't blindly merge
    container `.to_fields()` because verl's flag vocabulary is mostly deep
    dotted paths that don't line up with the common config attr names.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "H100"
    # verl ships pre-built images pinned to a vLLM version.
    image: str = "verlai/verl:vllm011.latest"
    # Git ref of verl to `git clone` + install. Pin to a commit for reproducibility.
    verl_git_url: str = "https://github.com/volcengine/verl"

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
    actor_save_contents: list[str] = field(default_factory=lambda: ["model"])

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
    extra_overrides: list[str] = field(default_factory=list)

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging.
    app_tags: dict = field(default_factory=dict)

class VerlConfig:
    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: VerlFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: VerlFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or VerlFrameworkConfig()

    def hf_model_dir(self) -> str:
        assert self.model is not None and self.model.model_name is not None
        return str(MODELS_PATH / HF_MODELS_SUBDIR / self.model.model_name)

    def mcore_model_dir(self) -> str:
        assert self.model is not None and self.model.model_name is not None
        return str(MODELS_PATH / MCORE_MODELS_SUBDIR / self.model.model_name)

    def training_checkpoint_dir(self) -> str:
        assert self.model is not None and self.model.model_name is not None
        return str(MODELS_PATH / TRAINING_CHECKPOINT_SUBDIR / self.model.model_name)

    def cli_args(self, *, reward_function_path: str) -> list[str]:
        assert self.model is not None, "VerlConfig.model must be set"
        assert self.dataset is not None, "VerlConfig.dataset must be set"
        cfg = self.framework_config

        train_files = self.dataset.prompt_data
        val_files = (
            self.dataset.eval_prompt_data
            if isinstance(self.dataset.eval_prompt_data, str)
            else self.dataset.prompt_data
        )
        if isinstance(self.dataset.eval_prompt_data, list) and self.dataset.eval_prompt_data:
            val_files = self.dataset.eval_prompt_data[-1]

        project_name = self.wandb.project if self.wandb else "verl"
        experiment_name = (
            self.wandb.exp_name or self.wandb.group if self.wandb else "verl_run"
        )
        trainer_loggers = '["console","wandb"]' if self.wandb else '["console"]'

        overrides: list[str] = [
            f"algorithm.adv_estimator={cfg.adv_estimator}",
            f"algorithm.use_kl_in_reward={cfg.use_kl_in_reward}",
            f"data.train_files={train_files}",
            f"data.val_files={val_files}",
            f"data.prompt_key={cfg.prompt_key}",
            f"data.return_raw_chat={cfg.return_raw_chat}",
            f"data.max_prompt_length={cfg.max_prompt_length}",
            f"data.max_response_length={cfg.max_response_length}",
            f"data.filter_overlong_prompts={cfg.filter_overlong_prompts}",
            f"data.truncation={cfg.truncation}",
            f"data.train_batch_size={cfg.train_batch_size}",
            f"actor_rollout_ref.model.path={self.hf_model_dir()}",
            f"actor_rollout_ref.model.use_fused_kernels={cfg.use_fused_kernels}",
            f"trainer.nnodes={cfg.n_nodes}",
            f"trainer.n_gpus_per_node={cfg.gpus_per_node}",
            f"actor_rollout_ref.actor.strategy={cfg.actor_strategy}",
            f"actor_rollout_ref.actor.optim.lr={cfg.actor_lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg.ppo_max_token_len_per_gpu}",
            f"actor_rollout_ref.actor.use_dynamic_bsz={cfg.actor_use_dynamic_bsz}",
            f"actor_rollout_ref.actor.use_kl_loss={cfg.use_kl_loss}",
            f"actor_rollout_ref.actor.kl_loss_coef={cfg.kl_loss_coef}",
            f"actor_rollout_ref.actor.kl_loss_type={cfg.kl_loss_type}",
            f"actor_rollout_ref.actor.entropy_coeff={cfg.entropy_coeff}",
            f"actor_rollout_ref.actor.loss_agg_mode={cfg.loss_agg_mode}",
            (
                "actor_rollout_ref.actor.checkpoint.save_contents="
                + "[" + ",".join(f'"{x}"' for x in cfg.actor_save_contents) + "]"
            ),
            f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={cfg.actor_pp_size}",
            f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={cfg.actor_tp_size}",
            f"actor_rollout_ref.actor.megatron.context_parallel_size={cfg.actor_cp_size}",
            f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={cfg.actor_ep_size}",
            f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={cfg.actor_etp_size}",
            f"actor_rollout_ref.actor.megatron.dtype={cfg.actor_dtype}",
            f"actor_rollout_ref.actor.megatron.param_offload={cfg.actor_param_offload}",
            f"actor_rollout_ref.actor.megatron.grad_offload={cfg.actor_grad_offload}",
            f"actor_rollout_ref.actor.megatron.optimizer_offload={cfg.actor_optimizer_offload}",
            "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True",
            f"actor_rollout_ref.actor.megatron.dist_checkpointing_path={self.mcore_model_dir()}",
            f"actor_rollout_ref.rollout.name={cfg.rollout_name}",
            f"actor_rollout_ref.rollout.mode={cfg.rollout_mode}",
            f"actor_rollout_ref.rollout.n={cfg.n_rollouts_per_prompt}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.rollout_tp_size}",
            f"actor_rollout_ref.rollout.data_parallel_size={cfg.rollout_dp_size}",
            f"actor_rollout_ref.rollout.expert_parallel_size={cfg.rollout_ep_size}",
            f"actor_rollout_ref.rollout.dtype={cfg.rollout_dtype}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.rollout_gpu_memory_util}",
            f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={cfg.rollout_log_prob_use_dynamic_bsz}",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg.rollout_log_prob_max_token_len}",
            f"actor_rollout_ref.rollout.enable_chunked_prefill={cfg.rollout_enable_chunked_prefill}",
            f"actor_rollout_ref.rollout.temperature={cfg.rollout_temperature}",
            f"actor_rollout_ref.rollout.top_p={cfg.rollout_top_p}",
            f"actor_rollout_ref.rollout.top_k={cfg.rollout_top_k}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={cfg.val_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.top_p={cfg.val_top_p}",
            f"actor_rollout_ref.rollout.val_kwargs.top_k={cfg.val_top_k}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={cfg.val_do_sample}",
            f"actor_rollout_ref.rollout.val_kwargs.n={cfg.val_n}",
            f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={cfg.ref_pp_size}",
            f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={cfg.ref_tp_size}",
            f"actor_rollout_ref.ref.megatron.context_parallel_size={cfg.ref_cp_size}",
            f"actor_rollout_ref.ref.megatron.expert_model_parallel_size={cfg.ref_ep_size}",
            f"actor_rollout_ref.ref.megatron.expert_tensor_parallel_size={cfg.ref_etp_size}",
            f"actor_rollout_ref.ref.megatron.param_offload={cfg.ref_param_offload}",
            "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True",
            f"actor_rollout_ref.ref.megatron.dist_checkpointing_path={self.mcore_model_dir()}",
            f"actor_rollout_ref.ref.megatron.dtype={cfg.ref_dtype}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={cfg.ref_log_prob_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={cfg.ref_log_prob_use_dynamic_bsz}",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={cfg.ref_log_prob_max_token_len}",
            f"trainer.critic_warmup={cfg.critic_warmup}",
            f"trainer.logger={trainer_loggers}",
            f"trainer.project_name={project_name}",
            f"trainer.experiment_name={experiment_name}",
            f"trainer.test_freq={cfg.test_freq}",
            f"trainer.default_local_dir={self.training_checkpoint_dir()}",
            f"trainer.save_freq={cfg.save_freq}",
            f"trainer.resume_mode={cfg.resume_mode}",
            f"custom_reward_function.path={reward_function_path}",
            f"custom_reward_function.name={cfg.reward_function_name}",
        ]
        overrides.extend(cfg.extra_overrides)
        return overrides

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_verl_app

        return build_verl_app(verl=self, name=name)


VerlTrainingConfig = VerlFrameworkConfig
