import math
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import Any

from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.checkpoint import CheckpointConfig
from modal_training_gym.common.models import (
    ModelArchitecture,
    ModelConfig,
)
from modal_training_gym.common.wandb import WandbConfig

import modal

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

_SLIME_SKIP = {
    "recipe_type",
    "environment",
    "async_mode",
    "wandb",
    "name",
    "app_tags",
    "image_run_commands",
    "local_python_sources",
    "local_slime",
    "checkpoint",
    "custom_rm_function",
}

YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class SlimeRecipe(BaseTrainRecipe):
    recipe_type: RecipeType = RecipeType.SLIME

    # ── App identity ─────────────────────────────────────────────────────────
    name: str = ""
    app_tags: dict = field(default_factory=dict)

    # ── Launcher instructions (not slime CLI flags) ─────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    wandb: WandbConfig | None = None
    image_run_commands: Callable[[modal.Image], modal.Image] | None = None
    local_python_sources: list[str] = field(default_factory=list)
    local_slime: str | None = None

    # ── Cluster and parallelism ────────────────────────────────────────────
    gpu_type: str = "H100"
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    colocate: bool = True
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus: int | None = None
    rollout_num_gpus_per_engine: int = 1
    use_critic: bool = False
    critic_num_nodes: int | None = None
    critic_num_gpus_per_node: int | None = None

    # ── RL algorithm ────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    n_samples_per_prompt: int = 2
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = False
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0
    ref_load: str = ""

    # ── Rollout ─────────────────────────────────────────────────────────────
    num_rollout: int = 1
    rollout_batch_size: int = 8
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.7
    rollout_shuffle: bool = True

    # ── Training ────────────────────────────────────────────────────────────
    global_batch_size: int = 16
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer: str = "adam"

    # ── Memory and precision ────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1

    # ── Dynamic batching ────────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216

    # ── Eval ────────────────────────────────────────────────────────────────
    eval_interval: int = 0
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 16384
    eval_top_p: float = 1.0
    eval_config: dict | None = None

    # ── Checkpointing ───────────────────────────────────────────────────────
    save: str = "/checkpoints"
    save_interval: int = 1000
    megatron_to_hf_mode: str = "bridge"
    use_fault_tolerance: bool = True

    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(iteration_prefix="iter_")
    )

    # ── Reward model ─────────────────────────────────────────────────────────
    rm_type: str | None = None
    custom_rm_path: str = ""
    custom_rm_function: Callable | None = None

    # ── SGLang / config overrides ───────────────────────────────────────────
    sglang_config: dict | None = None
    custom_config_path: dict | None = None
    apply_chat_template_kwargs: str = ""

    # ── Post-init ────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.custom_rm_function is not None and not self.custom_rm_path:
            fn = self.custom_rm_function
            mod = getattr(fn, "__module__", None) or ""
            name = getattr(fn, "__qualname__", None) or fn.__name__
            if mod == "__main__":
                import inspect

                src_file = inspect.getfile(fn)
                mod = Path(src_file).stem
            object.__setattr__(self, "custom_rm_path", f"{mod}.{name}")

    # ── Container → slime flag converters ────────────────────────────────────

    @staticmethod
    def _dataset_to_fields(ds: "DatasetConfig") -> dict[str, Any]:
        return {
            "prompt_data": ds.prompt_data,
            "eval_prompt_data": ds.eval_prompt_data,
            "input_key": ds.input_key,
            "label_key": ds.label_key,
            "apply_chat_template": ds.apply_chat_template,
        }

    @staticmethod
    def _validate_custom_model_architecture(
        m: "ModelConfig",
    ) -> "ModelArchitecture":
        if m.architecture is None:
            raise ValueError(
                "SlimeRecipe requires a ModelArchitecture on the attached "
                "ModelConfig. Set `architecture = ModelArchitecture(...)` "
                "on your subclass."
            )
        return m.architecture

    @staticmethod
    def _model_to_fields(m: "ModelConfig") -> dict[str, Any]:
        arch = SlimeRecipe._validate_custom_model_architecture(m)
        return {
            "hf_checkpoint": m.model_path or m.model_name,
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
            "use_rotary_position_embeddings": arch.use_rotary_position_embeddings,
            "rotary_base": arch.rotary_base,
        }

    @staticmethod
    def _wandb_to_fields(w: "WandbConfig") -> dict[str, Any]:
        return {
            "use_wandb": True,
            "wandb_project": w.project,
            "wandb_group": w.group,
            "wandb_key": w.key,
            "disable_wandb_random_suffix": w.disable_random_suffix,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        import dataclasses as _dc

        fields: dict[str, Any] = {}
        for f in _dc.fields(self):
            fields[f.name] = getattr(self, f.name)
        if dataset is not None:
            fields.update(self._dataset_to_fields(dataset))
        if model is not None:
            fields.update(self._model_to_fields(model))
        if self.wandb is not None:
            fields.update(self._wandb_to_fields(self.wandb))
        return {k: v for k, v in fields.items() if k not in _SLIME_SKIP}

    # ── Public API ────────────────────────────────────────────────────────────

    def cli_args(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> list[str]:
        out: list[str] = []
        for key, val in self._fields(dataset=dataset, model=model).items():
            if val is None or val is False or val == "":
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    @property
    def total_nodes(self) -> int:
        f = self._fields()
        gpus_per_node = f.get("actor_num_gpus_per_node", 8)
        actor_nodes = f.get("actor_num_nodes", 1)
        colocate = f.get("colocate", False)
        use_critic = f.get("use_critic", False)
        critic_nodes = f.get("critic_num_nodes") or actor_nodes
        critic_gpus = f.get("critic_num_gpus_per_node") or gpus_per_node
        rollout_gpus = f.get("rollout_num_gpus")

        training_gpus = actor_nodes * gpus_per_node
        if use_critic:
            training_gpus += critic_nodes * critic_gpus

        if colocate:
            total_gpus = training_gpus
        else:
            rollout_gpus = rollout_gpus or (actor_nodes * gpus_per_node)
            total_gpus = training_gpus + rollout_gpus

        if total_gpus % gpus_per_node != 0:
            raise ValueError(
                f"total_gpus={total_gpus} is not a multiple of gpus_per_node={gpus_per_node}. "
                f"Adjust actor_num_nodes, rollout_num_gpus, or actor_num_gpus_per_node."
            )

        return math.ceil(total_gpus / gpus_per_node)

    @classmethod
    def get_base_recipe(cls, model_config: ModelConfig) -> "SlimeRecipe":
        from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import (
            Qwen3_4b_Recipe,
        )

        if model_config.model_name == "Qwen/Qwen3-4B":
            return Qwen3_4b_Recipe()
        return SlimeRecipe()
