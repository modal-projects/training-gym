"""Projector-only training on Qwen3.6-35B-A3B with slime.

``Qwen3_6_35b_Projector_Recipe`` trains an embedding projector — a small MLP
mapping externally computed per-token embeddings (a protein or DNA encoder's
outputs, say) into the decoder's hidden space — against a fully frozen
Qwen3.6-35B-A3B. No LoRA, no optimizer state for 35B parameters, no weight
sync, no rollout engines: one node, and the trained artifact is a file of tens
of megabytes.

The RL and full-weight shapes of this model stay where they are, in
:class:`~modal_training_gym.train_recipes.slime_recipe.Qwen3_6_35b_Recipe`,
which this recipe inherits its cluster shape, MoE settings and sglang
configuration from.
"""

from collections.abc import Callable
from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig, Qwen3_6_35B
from modal_training_gym.frameworks.slime.projector_config import (
    ARGS_KEY,
    PROVIDER_PATH,
    ROLLOUT_PATH,
    SAVE_HOOK_PATH,
    TRAINABLE_PARAM_PATTERNS,
    ProjectorSpec,
)
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import Qwen3_6_35b_Recipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_35b_Projector_Recipe(Qwen3_6_35b_Recipe):
    """Projector-only supervised training against a frozen Qwen3.6-35B-A3B.

    The trainable set is the projector alone::

        TrainConfig(
            model=Qwen3_6_35B(),
            dataset=my_embedding_dataset,  # embeddings + positions columns
            recipe=Qwen3_6_35b_Projector_Recipe(
                projector=ProjectorSpec(input_dim=1536),
            ),
        ).train()

    How each part of "train only the projector" is arranged:

    *Freezing.* ``custom_model_provider_path`` builds the model slime's model
    script asks for — so the MoE routing, gated attention and qk-layernorm come
    from upstream verbatim — sets ``requires_grad=False`` on every base
    parameter, then attaches the projector, all before Megatron builds the
    optimizer, which skips parameters that do not require grad. Optimizer
    state, gradient buffers and DDP buckets are therefore proportional to the
    projector rather than to 35B parameters, which is what takes this down to a
    single node. ``only_train_params_name_list`` states the same intent in
    slime's own terms.

    *Weight sync.* There is none: ``debug_train_only`` runs the training path
    with no sglang engines, which a supervised run wants anyway since it
    generates nothing. An RL variant needs more than this recipe: slime's sync
    walks every named parameter of the model, so the projector would have to be
    excluded from it, and the projector's contribution to a rollout — embeddings
    written into the input embedding stream — has no representation in an sglang
    request today.

    *Data.* The dataset's per-sample embeddings and their token positions ride
    in the sample's metadata
    (:class:`~modal_training_gym.common.dataset.EmbeddingProjectorDataset`);
    the recipe's rollout function turns them into ``multimodal_train_inputs``,
    the per-sample tensor dict slime concatenates over a packed microbatch and
    splats into ``model.forward``, where the merge picks them up.

    *Checkpointing.* Every ``projector.save_interval`` optimizer steps, and
    always on the run's last one, the projector's own state dict is written
    (:mod:`modal_training_gym.frameworks.slime.embedding_projector`). The
    frozen base needs none: it stays byte-identical to the checkpoint the run
    started from, so the projector file plus ``hf_checkpoint`` describe the
    trained artifact completely — and the projector is kept out of Megatron's
    own state dicts, so the base checkpoint still loads even though the model
    now has parameters that checkpoint never had. ``save_interval`` is left far
    above ``num_rollout`` so Megatron never writes tens of gigabytes of
    unchanged weights.

    Parallelism is TP2/PP1/CP1/EP4 on one 8×H100 node. Pipeline parallelism is
    1 because with the base frozen only the first stage holds trainable
    parameters, so later stages would build an optimizer over an empty
    parameter set; context parallelism is 1 because the merge rebases the
    data's token positions onto a tensor/sequence shard and not onto Megatron's
    context-parallel chunking. Scale with tensor or expert parallelism.
    """

    _SKIP_FIELDS: ClassVar[frozenset[str]] = Qwen3_6_35b_Recipe._SKIP_FIELDS | {
        "projector"
    }
    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_6_35B

    projector: ProjectorSpec = field(default_factory=ProjectorSpec)

    custom_model_provider_path: str | None = PROVIDER_PATH
    only_train_params_name_list: list[str] | None = field(
        default_factory=lambda: list(TRAINABLE_PARAM_PATTERNS)
    )
    custom_megatron_before_train_step_hook: Callable | str | None = SAVE_HOOK_PATH

    # Supervised: the dataset carries the targets, so there is nothing to
    # generate, score, or turn into advantages. The rollout function is slime's
    # own SFT one plus the step that lifts each sample's embeddings into the
    # tensor dict the model's forward receives.
    loss_type: str = "sft_loss"
    loss_mask_type: str = "qwen3_5"
    rollout_function: Callable | str | None = ROLLOUT_PATH
    disable_compute_advantages_and_returns: bool = True
    debug_train_only: bool = True
    calculate_per_token_loss: bool = True
    n_samples_per_prompt: int = 1
    # No reference model: a KL term against the frozen base is a KL against the
    # model itself, and it would cost a second 35B copy to compute.
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    kl_coef: float = 0.0
    # The base is still read from here — slime falls back to ``ref_load`` when
    # ``load`` holds no checkpoint — but at PP1 rather than the RL recipe's PP2,
    # so it gets its own conversion directory instead of invalidating that one.
    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp2pp1"

    # No evaluation pass: the rollout function has nothing to generate with, so
    # it raises when slime calls it with ``evaluation=True``.
    eval_interval: int | None = None

    # ── Parallelism (see the class docstring) ─────────────────────────────
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1

    num_rollout: int = 10
    rollout_batch_size: int = 8
    global_batch_size: int = 8
    # Only the projector is written, by the hook.
    save_interval: int = 1_000_000
    no_save_optim: bool = True

    # An adapter trained from random initialization, unlike the frozen base it
    # feeds, so far above the 1e-6 a full-weight run of this model uses.
    lr: float = 1e-4
    lr_decay_style: str = "cosine"

    @model_validator(mode="after")
    def _serialize_projector(self) -> "Qwen3_6_35b_Projector_Recipe":
        """Carry the projector spec to the containers through ``extra_config``.

        slime sets every ``extra_config`` key on ``args``, so the provider, the
        rollout function and the checkpoint hook read the spec back without new
        slime CLI flags.
        """
        cfg = dict(self.extra_config) if isinstance(self.extra_config, dict) else {}
        if cfg.get(ARGS_KEY) != self.projector.to_args_dict():
            cfg[ARGS_KEY] = self.projector.to_args_dict()
            object.__setattr__(self, "extra_config", cfg)
        return self

    @model_validator(mode="after")
    def _reject_distributed_optimizer(self) -> "Qwen3_6_35b_Projector_Recipe":
        """Reject the distributed optimizer: it shards the gradient we sum.

        The projector's tensor-parallel gradient sum reads each parameter's
        ``main_grad`` whole. Under ``use_distributed_optimizer`` that buffer is
        a reduce-scattered shard of a bucket, and summing shards across the
        tensor-parallel group is not the replicated weight's whole-sequence
        gradient — it would train on a silently wrong gradient rather than
        fail. There is nothing to gain either: with the base frozen, this run's
        optimizer state is the projector's alone, tens of megabytes.
        """
        if self.use_distributed_optimizer:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs use_distributed_optimizer=False: "
                "the replicated projector's gradients are summed across the "
                "tensor-parallel group from whole main_grad buffers, which the "
                "distributed optimizer replaces with reduce-scattered shards. "
                "Its only benefit is sharding optimizer state, and the base is "
                "frozen, so this run's optimizer state is the projector's alone."
            )
        return self

    @model_validator(mode="after")
    def _reject_rl_objective(self) -> "Qwen3_6_35b_Projector_Recipe":
        """Reject the RL settings this recipe has no rollouts to feed.

        ``debug_train_only`` starts no sglang engines, so nothing generates
        responses to score: a policy loss or a KL against a reference model
        would be computed over the dataset's own tokens, which is not the
        objective either flag means. RL over a projector needs a
        projector-aware weight sync first (see the class docstring).
        """
        if self.loss_type != "sft_loss":
            raise TrainingGymConfigError(
                f"{type(self).__name__} is supervised: loss_type must be "
                f'"sft_loss" (got {self.loss_type!r}). It starts no rollout '
                "engines, so a policy loss would have no generated responses "
                "to score."
            )
        if self.use_kl_loss or self.kl_loss_coef or self.kl_coef:
            raise TrainingGymConfigError(
                f"{type(self).__name__} trains no base weights, so a KL term "
                "against the reference model is a KL against the frozen base "
                "itself — identically zero, at the cost of a second 35B copy. "
                f"Got use_kl_loss={self.use_kl_loss}, "
                f"kl_loss_coef={self.kl_loss_coef}, kl_coef={self.kl_coef}."
            )
        return self

    def validate_model_parallelism(self, model: ModelConfig) -> None:
        super().validate_model_parallelism(model)
        if self.pipeline_model_parallel_size != 1:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs pipeline_model_parallel_size=1: the "
                "projector lives on the first pipeline stage and the base is frozen, "
                "so every later stage would build an optimizer over an empty "
                f"parameter set. Got {self.pipeline_model_parallel_size}; scale with "
                "tensor_model_parallel_size or expert_model_parallel_size."
            )
        if self.context_parallel_size != 1:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs context_parallel_size=1: the "
                "projected embeddings are written into the input embeddings at "
                "the dataset's token positions, which are rebased onto a "
                "tensor/sequence shard but not onto Megatron's context-parallel "
                f"chunking, so CP={self.context_parallel_size} would land them "
                "on the wrong tokens. Scale with tensor_model_parallel_size or "
                "expert_model_parallel_size."
            )
        if self.tensor_model_parallel_size > 1 and not self.sequence_parallel:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs sequence_parallel=True when "
                f"tensor_model_parallel_size > 1 (got "
                f"TP={self.tensor_model_parallel_size}). The projector is "
                "replicated across the tensor-parallel group and its gradients "
                "are summed over it, which is the whole-sequence gradient only "
                "while each rank merges a disjoint sequence shard; without "
                "sequence parallelism every rank merges every position, so the "
                "sum would scale the projector's gradient by TP."
            )
