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

import warnings
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

# Routed experts in miles' ``scripts/models/glm5.2-744B-A40B.py`` (``--num-experts
# 256``, top-8, 1 shared), which the 5-layer checkpoint keeps. The gym's
# expert-parallel preflight reads this off a model's ``ModelArchitecture``, and
# GLM-5.2 has none — the DSA spec and the indexer are not representable there,
# so the model script owns the arch args. Kept here instead, where the model
# script is pinned, so changing EP is still checked before the cluster starts.
_NUM_ROUTED_EXPERTS = 256


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

    *Checkpointing.* Every ``projector.save_interval`` optimizer steps, and
    always on the run's last one, the projector's own state dict is written
    (:mod:`modal_training_gym.frameworks.miles.embedding_projector`). The
    frozen base needs no checkpoint: it stays byte-identical to the HF
    checkpoint the run started from, so the projector file plus that model name
    describe the trained artifact completely. ``save_interval`` is left far
    above ``num_rollout`` so Megatron never writes terabytes of unchanged
    weights.

    ``pipeline_model_parallel_size`` is 1: with the base frozen only the first
    pipeline stage holds trainable parameters, so later stages would build an
    optimizer over an empty parameter set. ``context_parallel_size`` is 1 too,
    because the merge rebases the data's token positions onto a tensor/sequence
    shard and not onto Megatron's context-parallel chunking. Scale with tensor
    or expert parallelism.
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
    # No evaluation pass: the rollout function has nothing to generate with, so
    # it raises when miles calls it with ``evaluation=True``. Leaving this to
    # miles' own gate on ``eval_interval`` would make a user-launched run's
    # survival depend on that gate, while the dataset still emits
    # ``--eval-prompt-data``.
    skip_eval_before_train: bool = True
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

    @model_validator(mode="after")
    def _reject_megatron_load(self) -> "GLM_5_2_Projector_Recipe":
        """Reject Megatron checkpoint loading: resume goes through the projector.

        The projector is registered as a submodule of the actor (Megatron's DDP
        and optimizer only see parameters in the module tree), so it shows up in
        the model's ``state_dict`` under ``embedding_projector.*``. A Megatron
        checkpoint written before this recipe existed has no such keys, and the
        base itself never changes during a projector-only run, so ``load`` /
        ``ref_load`` have nothing to offer here and a strict key check would
        fail. Resume by pointing ``projector.load`` at a
        ``projector_iter_*.pt``; the base comes from ``hf_checkpoint``.

        Safe because of the bridge-mode branch in the pinned image's
        ``miles/utils/arguments.py``: with no usable ``--load`` directory it sets
        ``args.load = args.ref_load or args.hf_checkpoint``, and
        ``megatron_utils/checkpoint.py`` routes a non-Megatron directory into
        ``_load_checkpoint_hf``. ``ref_load`` is therefore an alternative
        spelling of the same load here, not the only one — bridge recipes that
        set it (``Gemma4_26B_A4B_Recipe``) point it at the same HF reference as
        ``hf_checkpoint``. ``_require_pretrained_base`` keeps that fallback
        populated.
        """
        if self.load or self.ref_load:
            raise TrainingGymConfigError(
                f"{type(self).__name__} resumes through projector.load, not "
                f"Megatron's load/ref_load (got load={self.load!r}, "
                f"ref_load={self.ref_load!r}). The frozen base is loaded from "
                f"hf_checkpoint={self.hf_checkpoint!r} and only the projector "
                "is ever written."
            )
        return self

    @model_validator(mode="after")
    def _require_pretrained_base(self) -> "GLM_5_2_Projector_Recipe":
        """Require the one field that makes the frozen base a pretrained base.

        With ``load`` and ``ref_load`` both rejected, ``hf_checkpoint`` is the
        only source the bridge-mode fallback
        (``args.load = args.ref_load or args.hf_checkpoint``) has left. Were it
        empty, miles would log one ``--load '' is empty; starting from
        model_provider-initialized weights`` warning and train the projector
        against a *randomly initialized* base — a run that costs a full GPU
        budget and produces an adapter for a model that does not exist. Same for
        a non-bridge ``megatron_to_hf_mode``, whose branch falls back to
        ``ref_load`` alone.
        """
        if not self.hf_checkpoint:
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs hf_checkpoint set: it is the only "
                "source the frozen base can be loaded from once load/ref_load "
                "are rejected, and miles would otherwise train the projector "
                "against randomly initialized base weights."
            )
        if self.megatron_to_hf_mode != "bridge":
            raise TrainingGymConfigError(
                f"{type(self).__name__} requires megatron_to_hf_mode='bridge' "
                f"(got {self.megatron_to_hf_mode!r}): only the bridge branch "
                "falls back to hf_checkpoint for the base weights, and this "
                "recipe rejects the ref_load the other branch needs."
            )
        return self

    @model_validator(mode="after")
    def _reject_evaluation(self) -> "GLM_5_2_Projector_Recipe":
        """Reject an eval schedule: the rollout function cannot evaluate.

        ``projector_sft_rollout`` raises on ``evaluation=True`` — a supervised
        projector run has no generation path to evaluate with — so an
        ``eval_interval`` would kill the run partway through instead of at
        launch.
        """
        if self.eval_interval is not None:
            raise TrainingGymConfigError(
                f"{type(self).__name__} cannot evaluate (got "
                f"eval_interval={self.eval_interval}): its rollout function is "
                "supervised and does not generate, so miles' eval pass has "
                "nothing to run. Leave eval_interval unset."
            )
        return self

    @model_validator(mode="after")
    def _reject_distributed_optimizer(self) -> "GLM_5_2_Projector_Recipe":
        """Reject the distributed optimizer: it shards the gradient we sum.

        The projector's tensor-parallel gradient sum reads each parameter's
        ``main_grad`` whole. Under ``use_distributed_optimizer`` that buffer is
        a reduce-scattered shard of a bucket, and summing shards across the
        tensor-parallel group is not the replicated weight's whole-sequence
        gradient — it would train on a silently wrong gradient rather than
        fail. There is nothing to gain either: the optimizer state a
        projector-only run carries is the projector's, tens of megabytes, so
        sharding it saves nothing against a frozen 744B base.
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

    def _recheck_mutable_launch_invariants(self) -> None:
        """Re-run the guards that assignment after construction can defeat.

        Pydantic dataclasses do not re-validate on assignment, and callers do
        assign: ``scripts/validate_model_configs.py`` sets ``eval_interval`` and
        ``save_interval`` on the recipe the backend built. Re-checking here (on
        the preflight ``_fields`` runs while emitting flags) means an override
        that would kill the run mid-flight fails at launch instead.

        ``save_interval`` is only warned about: it re-enables Megatron's
        full-model save, which writes the *unchanged* frozen base — terabytes of
        it at the 744B shape — but a run that pays for it still trains
        correctly, and the projector's own cadence is ``projector.save_interval``.
        """
        self._reject_evaluation()
        self._reject_lora()
        self._reject_megatron_load()
        self._require_pretrained_base()
        self._reject_distributed_optimizer()
        if self.save_interval is not None and self.save_interval <= self.num_rollout:
            warnings.warn(
                f"{type(self).__name__} has save_interval={self.save_interval} "
                f"with num_rollout={self.num_rollout}: Megatron will write the "
                "frozen base (unchanged, and terabytes of it at the 744B shape) "
                "during the run. Only the projector needs saving; its cadence "
                "is projector.save_interval.",
                stacklevel=2,
            )

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        super().validate_model_parallelism(model)
        self._recheck_mutable_launch_invariants()
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
        if (
            self.expert_model_parallel_size
            and _NUM_ROUTED_EXPERTS % self.expert_model_parallel_size
        ):
            raise TrainingGymConfigError(
                f"{type(self).__name__} needs expert_model_parallel_size to "
                f"divide the model script's {_NUM_ROUTED_EXPERTS} routed "
                f"experts, got EP={self.expert_model_parallel_size}."
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
