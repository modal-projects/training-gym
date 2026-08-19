"""GLM-5.2 projector-only miles recipes.

Trains an embedding projector against a frozen GLM-5.2 — no LoRA, no optimizer
state for 744B parameters, no weight sync, no engines — which is what brings a
744B-A40B model's adapter training down to a shape worth running.

``GLM_5_2_Projector_Recipe``
    The 744B checkpoint, 8×8×H200, TP4/EP32 with the base frozen.
``GLM_5_2_5Layer_Projector_Recipe``
    The same plumbing against upstream's 5-layer pruned checkpoint on
    1×8×H200, for proving freezing, the forward merge and the projector
    checkpoint at a single node's cost.

Full-weight or LoRA GLM-5.2 GRPO is not covered here: that is upstream's
``scripts/run_glm5_2_744b_a40b.py`` shape at 32 nodes per training group.
"""

from dataclasses import field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import GLM_5_2, GLM_5_2_5Layer
from modal_training_gym.frameworks.miles.projector_config import (
    ARGS_KEY,
    PROVIDER_PATH,
    ROLLOUT_PATH,
    SAVE_HOOK_PATH,
    ProjectorSpec,
)
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    from modal_training_gym.common.models import ModelConfig

# 1 TiB: the 744B checkpoint plus its torch_dist conversion overflows the
# container's default disk.
_EPHEMERAL_DISK_MIB = 1_048_576


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_2_Projector_Recipe(MilesRecipe):
    """Projector-only supervised training against a frozen GLM-5.2 (8×8×H200).

    The trainable set is the projector alone::

        TrainConfig(
            model=GLM_5_2(),
            dataset=my_embedding_dataset,  # embeddings + positions columns
            recipe=GLM_5_2_Projector_Recipe(
                projector=ProjectorSpec(input_dim=1536),
            ),
        ).train()

    ``miles_model_name`` points at miles' own
    ``scripts/models/glm5.2-744B-A40B.py``, so the DSA attention spec, the
    cross-layer index and the MoE routing come from upstream rather than being
    restated here — which is also why
    :class:`~modal_training_gym.common.models.GLM_5_2` carries no
    ``ModelArchitecture``.

    How each part of "train only the projector" is arranged:

    *Freezing.* ``custom_model_provider_path`` builds the model the model script
    asks for, sets ``requires_grad=False`` on every base parameter, then
    attaches the projector — all before Megatron builds the optimizer, which
    skips parameters that do not require grad. Optimizer state, gradient
    buffers and checkpoints are therefore proportional to the projector, not to
    744B parameters, which is what removes the node count LoRA was being
    considered for.

    *Weight sync.* There is none: ``debug_train_only`` runs the training path
    with no engines. Miles' non-LoRA sync pushes the full base model to SGLang
    (its ``skip_base_sync`` is gated on ``is_lora``), so projector-only sync is
    not something the pinned image can express, and a supervised run has
    nothing to generate. An RL variant needs either a projector-aware sync path
    in miles or a full sync paid once per rollout.

    *Data.* The dataset's embeddings ride in the ``metadata`` column
    (:class:`~modal_training_gym.common.dataset.EmbeddingProjectorDataset`),
    which miles carries onto each ``Sample``; the recipe's rollout function
    turns them into per-sample tensors that miles concatenates over the packed
    batch and splats into ``forward``.

    *Checkpointing.* The before-train-step hook writes the projector's own state
    dict (:mod:`modal_training_gym.frameworks.miles.embedding_projector`). The
    frozen base needs no checkpoint: it stays byte-identical to the HF
    checkpoint the run started from, so the projector file plus that model name
    describe the trained artifact completely. ``save_interval`` is left far
    above ``num_rollout`` so Megatron never writes terabytes of unchanged
    weights.

    ``pipeline_model_parallel_size`` is 1: with the base frozen only the first
    pipeline stage holds trainable parameters, so later stages would build an
    optimizer over an empty parameter set. Scale with tensor, context or expert
    parallelism instead.
    """

    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | {"projector"}
    model_config_class: ClassVar[type["ModelConfig"]] = GLM_5_2

    projector: ProjectorSpec = field(default_factory=ProjectorSpec)

    gpu_type: str = "H200"
    colocate: bool = True
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _EPHEMERAL_DISK_MIB}
    )

    hf_checkpoint: str = "zai-org/GLM-5.2"
    miles_model_name: str = "glm5.2-744B-A40B"
    megatron_to_hf_mode: str = "bridge"

    custom_model_provider_path: str | None = PROVIDER_PATH
    custom_megatron_before_train_step_hook: str | None = SAVE_HOOK_PATH

    # Supervised: the dataset carries the targets, so there is nothing to
    # generate, score, or turn into advantages. The rollout function is miles'
    # own SFT one plus the step that lifts each row's embeddings into the tensor
    # dict the model's forward receives.
    loss_type: str | None = "sft_loss"
    rollout_function: str | None = ROLLOUT_PATH
    disable_compute_advantages_and_returns: bool = True
    debug_train_only: bool = True
    calculate_per_token_loss: bool = True
    num_epoch: int | None = 1
    n_samples_per_prompt: int = 1
    ref_load: str = ""

    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8

    train_backend: str = "megatron"
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 32
    expert_tensor_parallel_size: int = 1

    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 8192

    num_rollout: int = 10
    rollout_batch_size: int = 8
    global_batch_size: int = 8
    # Only the projector is written, by the hook.
    save_interval: int = 1_000_000

    # An adapter trained from random initialization, unlike the frozen base it
    # feeds, so far above the 1e-6 a full-weight GLM-5.2 run uses.
    optimizer: str = "adam"
    lr: float = 1e-4
    lr_decay_style: str = "cosine"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    attention_backend: str | None = "flash"
    update_weight_buffer_size: int | None = 2 * 1024**3

    @model_validator(mode="after")
    def _serialize_projector(self) -> "GLM_5_2_Projector_Recipe":
        """Carry the projector spec to the containers through ``extra_config``.

        Miles sets every ``extra_config`` key on ``args``, so the provider and
        the checkpoint hook read the spec back without new CLI flags.
        """
        cfg = dict(self.extra_config) if isinstance(self.extra_config, dict) else {}
        if cfg.get(ARGS_KEY) != self.projector.to_args_dict():
            cfg[ARGS_KEY] = self.projector.to_args_dict()
            object.__setattr__(self, "extra_config", cfg)
        return self

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "GLM_5_2_Projector_Recipe":
        """Keep the reservation when a caller supplies their own train kwargs.

        Replacing ``train_function_kwargs`` wholesale would otherwise drop it,
        and the run would die part-way through the checkpoint download. A caller
        naming ``ephemeral_disk`` still wins.
        """
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _EPHEMERAL_DISK_MIB, **kwargs},
            )
        return self

    @model_validator(mode="after")
    def _reject_lora(self) -> "GLM_5_2_Projector_Recipe":
        """Reject LoRA: two adapter mechanisms cannot both own the base.

        miles' LoRA path drives its own freezing, weight sync and checkpoint
        format, none of which the projector plumbing goes through.
        """
        if self.lora_rank:
            raise TrainingGymConfigError(
                f"{type(self).__name__} trains a projector over a frozen base, so "
                f"lora_rank must be unset (got {self.lora_rank}). A LoRA or "
                "full-weight GLM-5.2 run is upstream's "
                "scripts/run_glm5_2_744b_a40b.py shape, not this recipe."
            )
        return self

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        if self.pipeline_model_parallel_size != 1:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs pipeline_model_parallel_size=1: the "
                "projector lives on the first pipeline stage and the base is frozen, "
                "so every later stage would build an optimizer over an empty "
                f"parameter set. Got {self.pipeline_model_parallel_size}; scale with "
                "tensor_model_parallel_size, context_parallel_size or "
                "expert_model_parallel_size."
            )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_2_5Layer_Projector_Recipe(GLM_5_2_Projector_Recipe):
    """The projector path on the 5-layer pruned GLM-5.2, 1×8×H200.

    Upstream's own single-node smoke shape (the ``num_nodes == 1`` branch of
    ``scripts/run_glm5_2_744b_a40b.py``): 5 decoder layers, TP4/EP8, tokens per
    GPU cut to 2048. Exercises the same freezing, forward merge and projector
    checkpoint as the 744B recipe for one node's cost. The base is pruned, so
    its outputs mean nothing — this proves plumbing, not quality.
    """

    model_config_class: ClassVar[type["ModelConfig"]] = GLM_5_2_5Layer

    hf_checkpoint: str = "Pinaster/GLM-5.2_5layer"
    miles_model_name: str = "glm5.2-744B-A40B_5layer"

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    expert_model_parallel_size: int = 8
    max_tokens_per_gpu: int = 2048
