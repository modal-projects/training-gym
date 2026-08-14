"""Gemma-4-26B-A4B GRPO recipe on miles (1x8xH200), text-only or vision-language."""

import dataclasses as _dc
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    from modal_training_gym.common.models import ModelConfig

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

# Build-time shims for upstream gaps; see each patch's docstring.
_PATCHES = (
    "patch_router_startup_timeout",
    "patch_gemma4_vl_rollout_text",
)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


_EPHEMERAL_DISK_MIB = 1_048_576


# Defaults for modality="vision", for fields the caller left unset: smaller
# rollouts, since images are expensive.
_VISION_MODE: dict[str, Any] = {
    "num_rollout": 15,
    "rollout_batch_size": 8,
    "n_samples_per_prompt": 8,
    "global_batch_size": 64,
    "rollout_max_response_len": 256,
    "rollout_temperature": 1.0,
    "rollout_top_p": 0.95,
    "rollout_top_k": 64,
    "rm_type": None,
    "sglang_max_running_requests": 8,
    "save_interval": 10,
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Gemma4_26B_A4B_Recipe(MilesRecipe):
    """Gemma-4-26B-A4B MoE GRPO on 1×8×H200 with TP4/PP1/EP8, colocated.

    One checkpoint, two modes, chosen by ``modality``. The fields below are the
    text defaults; ``modality="vision"`` replaces those named in ``_VISION_MODE``
    with smaller-rollout values. Either way an argument you pass wins, because
    the vision values are applied as defaults before your arguments, not over
    them::

        TrainConfig(
            model=Gemma4_26B_A4B(),
            dataset=MultimodalDataset(modality="image", n_rows=120),
            recipe=Gemma4_26B_A4B_Recipe(modality="vision"),
        ).train()

    A vision run needs ``apply_chat_template=True`` so the prompt reaches the
    processor as a string, a leading ``<image>`` in each prompt so the processor
    inserts a placeholder for it — without one the image never reaches the model
    and it answers "I cannot see the image" at a constant reward — plus its own
    reward: the text path's ``gemma_math`` scores maths, not images, so
    ``modality="vision"`` clears ``rm_type`` and requires you to supply one.

    Based on upstream ``scripts/run_gemma_4_26b_a4b.py``, with the deviations
    noted inline.
    """

    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | {"modality"}

    modality: Literal["text", "vision"] = "text"

    gpu_type: str = "H200"
    colocate: bool = True
    image_run_commands: list[str] = field(default_factory=_image_patches)

    hf_checkpoint: str = "google/gemma-4-26B-A4B-it"
    ref_load: str = "google/gemma-4-26B-A4B-it"
    megatron_to_hf_mode: str = "bridge"
    miles_model_script: str = "scripts/models/gemma-4-26b-a4b-it.sh"
    # Model overflows container disk, so reserve 1 TiB.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8

    train_backend: str = "megatron"
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1

    # Off, unlike upstream: Gemma-4's decoder layer returns a tuple, which
    # Megatron's checkpointed forward rejects ("save_for_backward can only save
    # variables").
    recompute_granularity: str | None = None
    recompute_method: str | None = None
    recompute_num_layers: int | None = None
    # bshd rules out dynamic batching and miles asserts on the pair (upstream passes
    # both and trips it), so use an explicit micro batch; max_tokens_per_gpu is inert.
    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1
    max_tokens_per_gpu: int = 1024

    rm_type: str | None = "gemma_math"
    rollout_shuffle: bool = True
    balance_data: bool = True
    num_rollout: int = 3
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 256
    rollout_temperature: float = 1.0
    # None so text mode omits the flag and miles keeps its default; vision mode sets it.
    rollout_top_p: float | None = None
    rollout_top_k: int | None = None
    # generation_config.json's eos_token_id: <eos>, <turn|>, <|tool_response>.
    rollout_stop_token_ids: list[int] | None = field(
        default_factory=lambda: [1, 106, 50]
    )
    global_batch_size: int = 256
    save_interval: int = 20

    rollout_num_gpus_per_engine: int = 4
    # 0.25, not upstream's 0.55: both engines stay resident, and at 0.55 the
    # optimizer step OOMs on a 139.8 GiB H200.
    sglang_mem_fraction_static: float = 0.25
    # Gemma-4's global head_dim=512 exceeds FlashAttention's 256 cap.
    sglang_attention_backend: str = "triton"
    sglang_moe_runner_backend: str = "triton"
    sglang_disable_custom_all_reduce: bool = True
    sglang_disable_cuda_graph: bool = True
    sglang_disable_overlap_schedule: bool = True
    sglang_disable_radix_cache: bool = True
    sglang_max_running_requests: int | None = None
    # Resident, as upstream has it: offloading instead hits an illegal memory
    # access in SGLang's memory-saver path during the training step.
    no_offload_train: bool = True
    no_offload_rollout: bool = True
    # Off, unlike upstream: sglang's routed-experts capturer reads
    # num_experts_per_tok, which Gemma-4 calls top_k_experts, so every scheduler
    # dies with AttributeError.
    use_rollout_routing_replay: bool = False

    advantage_estimator: str = "grpo"
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    attention_backend: str = "unfused"
    qkv_format: str = "bshd"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    no_gradient_accumulation_fusion: bool = True
    no_check_for_nan_in_loss_and_grad: bool = True

    @model_validator(mode="before")
    @classmethod
    def _apply_vision_defaults(cls, data: Any) -> Any:
        """Fill unsupplied ``_VISION_MODE`` fields when ``modality="vision"``.

        Applied to the incoming arguments rather than to the built recipe, so a
        value the caller supplied is never overwritten and there is nothing to
        disambiguate afterwards. Rebuilding a resolved recipe from all of its
        fields is therefore a no-op.
        """
        if isinstance(data, ArgsKwargs):
            args, kwargs = data.args or (), dict(data.kwargs or {})
        elif isinstance(data, Mapping):
            args, kwargs = (), dict(data)
        else:
            return data

        names = [f.name for f in _dc.fields(cls) if f.init is not False]
        positional = set(names[: len(args)])
        if "modality" in positional:
            modality = args[names.index("modality")]
        else:
            modality = kwargs.get("modality", "text")
        if modality != "vision":
            return data

        for name, value in _VISION_MODE.items():
            if name not in positional and name not in kwargs:
                kwargs[name] = value
        return ArgsKwargs(args, kwargs) if isinstance(data, ArgsKwargs) else kwargs

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Gemma4_26B_A4B_Recipe":
        """Keep the build-time patches at the head of ``image_run_commands``.

        The field is replaced wholesale, so a caller adding their own command
        would otherwise drop the patches — and losing the VL one shows up as a
        blind model rather than an error.
        """
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "Gemma4_26B_A4B_Recipe":
        """Keep the disk reservation when a caller supplies their own kwargs.

        Same reasoning as ``_keep_image_patches``: passing ``{"secrets": [...]}``
        would otherwise drop the reservation and the run would die part-way through
        the checkpoint download. A caller who names ``ephemeral_disk`` still wins.
        """
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self

    def _brings_own_reward(self) -> bool:
        if self.custom_rm_function is not None or self.rollout_function is not None:
            return True
        if self.rm_type:
            return True
        extra = self.extra_config
        if isinstance(extra, str) and extra:
            return True
        return isinstance(extra, dict) and bool(extra.get("custom_rm_path"))

    @model_validator(mode="after")
    def _require_vision_reward(self) -> "Gemma4_26B_A4B_Recipe":
        """A vision run scores nothing unless the caller brings a reward.

        ``_VISION_MODE`` clears the text path's ``gemma_math``, so without this
        the run would reach a rollout before anything noticed.
        """
        if self.modality == "vision" and not self._brings_own_reward():
            raise TrainingGymConfigError(
                f"{type(self).__name__}(modality='vision') needs its own reward: "
                "the text default rm_type='gemma_math' scores maths, not images, "
                "so vision mode clears it. Pass custom_rm_function=..., or "
                "rm_type=... to choose a built-in deliberately."
            )
        return self

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        if self.pipeline_model_parallel_size != 1:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs pipeline_model_parallel_size=1: the "
                "Megatron bridge loads the vision tower and the tied input/output "
                "embedding onto a single pipeline stage, so a split only fails once "
                f"Megatron builds the model. Got {self.pipeline_model_parallel_size}; "
                "scale with tensor_model_parallel_size or expert_model_parallel_size."
            )
