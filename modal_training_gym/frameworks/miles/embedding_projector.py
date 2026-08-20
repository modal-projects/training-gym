"""Projector-only training on miles: a trainable adapter over a frozen base model.

Runs inside the miles containers, not on the launching machine. Three entry
points, all reached by dotted path:

``projector_model_provider``
    miles' ``--custom-model-provider-path``. Builds the model the model script
    asks for (so GLM-5.2's DSA spec, MoE routing and indexer come from
    upstream verbatim), freezes every base parameter, and attaches an
    :class:`EmbeddingProjector` on the first pipeline stage. The projector maps
    externally computed per-token embeddings — protein/DNA encoder outputs, for
    instance — into the decoder's hidden space and writes them into the input
    embeddings at the positions the data gives, following the merge that
    ``miles_plugins.models.inkling.mm_towers`` does for its vision/audio towers.
``save_projector_checkpoint``
    miles' before-train-step hook. Installs a wrapper around the optimizer's
    ``step`` that writes the projector (a few hundred MB at most) instead of the
    base model, which at 744B is terabytes of sharded weights that never change
    during a projector-only run.
``load_projector_checkpoint``
    Called by the provider when ``load`` is set, and usable standalone for
    inspection or export.

The trainable-parameter set is what makes this cheap: freezing the base before
Megatron builds the optimizer keeps optimizer state, gradient buffers and
checkpoints proportional to the projector, not to the model — LoRA-free, and
without the full-parameter memory bill.

Configuration arrives as one dict under the ``training_gym_projector`` miles
arg (the recipe writes it into ``extra_config``, whose keys miles sets on
``args``), so no new miles CLI flags are needed.
"""

import logging
import os

import torch  # pyright: ignore[reportMissingImports]  # torch is installed only in training images

from modal_training_gym.common.embedding_projector import (
    EmbeddingProjector as EmbeddingProjector,
    all_reduce_projector_grads,
    build_projector,
    check_projector_weights as check_projector_weights,
    freeze_base_model as freeze_base_model,
    get_projector as get_projector,
    init_projector as init_projector,
    is_projector_writer,
    load_projector_checkpoint as load_projector_checkpoint,
    log_projector_replica,
    projector_graph_tap,
    scatter_projected,
    write_projector_checkpoint as write_projector_checkpoint,
)
from modal_training_gym.frameworks.miles.projector_config import (
    ProjectorSpec,
    from_miles_args,
    should_save_projector,
)

logger = logging.getLogger(__name__)


def _build_base_model(args, pre_process: bool, post_process: bool, vp_stage):
    """Build the model miles would have built without this provider.

    The build happens with ``custom_model_provider_path`` still cleared, not
    just the lookup: whether miles resolves the choice when handing back the
    provider or when the provider runs is its business, and re-entering this
    function would recurse until the stack overflows.
    """
    from miles.backends.megatron_utils.model_provider import (  # pyright: ignore[reportMissingImports]
        get_model_provider_func,
    )

    saved = args.custom_model_provider_path
    args.custom_model_provider_path = None
    try:
        provider = get_model_provider_func(args)
        return provider(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
    finally:
        args.custom_model_provider_path = saved


class _ProjectorMerge:
    """Moves the batch's embeddings from ``model.forward`` into the embedding layer.

    miles splats ``multimodal_train_inputs`` straight into ``model(**kwargs)``
    (``miles/backends/megatron_utils/model.py``), but ``GPTModel.forward`` takes
    no such arguments, so they are stripped there and merged where the input
    embeddings actually exist: a forward hook on ``model.embedding``, which is
    the only place whose output layout and sequence-parallel sharding are already
    settled. Recomputing the embedding here instead would duplicate the vocab
    embedding's reduce-scatter and position-embedding branches.
    """

    def __init__(
        self, model, projector: EmbeddingProjector, cfg: ProjectorSpec
    ) -> None:
        self._model = model
        self._projector = projector
        self._embeddings_key = cfg.embeddings_key
        self._positions_key = cfg.positions_key
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self._weights_checked = False
        self._original_forward = model.forward
        model.forward = self._forward
        if model.pre_process:
            model.embedding.register_forward_hook(self._embedding_hook)

    def _forward(self, *fargs, **fkwargs):
        embeddings = fkwargs.pop(self._embeddings_key, None)
        positions = fkwargs.pop(self._positions_key, None)
        if embeddings is None or not self._model.pre_process:
            return self._original_forward(*fargs, **fkwargs)
        if positions is None:
            raise ValueError(
                f"'{self._embeddings_key}' was passed without "
                f"'{self._positions_key}'; the dataset must emit the token "
                "positions the embeddings occupy."
            )
        if positions.numel() != embeddings.shape[0]:
            raise ValueError(
                f"{positions.numel()} projector position(s) for "
                f"{embeddings.shape[0]} embedding row(s)"
            )
        self._pending = (embeddings, positions)
        try:
            return self._original_forward(*fargs, **fkwargs)
        finally:
            self._pending = None

    def _embedding_hook(self, module, inputs, output):
        if self._pending is None:
            # No embeddings in this microbatch, so nothing to merge — but the
            # base is frozen, so returning the embeddings untouched would leave
            # this rank's loss without a ``grad_fn`` and it would skip a
            # backward its peers wait on. Today's rollout rejects a sample
            # without embeddings, so this is the shape of a future rollout that
            # mixes in text-only samples rather than a path taken now.
            return output + projector_graph_tap(self._projector, output.dtype)
        embeddings, positions = self._pending
        if not self._weights_checked:
            # Once per process: the state this catches is set before the first
            # forward and never changes back, and the check reads every
            # parameter.
            self._weights_checked = True
            logger.info(
                "projector weights at first forward: %s",
                check_projector_weights(self._projector),
            )
        projected = self._projector(
            embeddings.to(
                device=output.device,
                # The rollout emits fp32; the projector carries the model's
                # params_dtype, bf16 for these recipes.
                dtype=next(self._projector.parameters()).dtype,
            )
        )
        return scatter_projected(
            self._model.config.sequence_parallel,
            output,
            projected,
            positions.to(output.device),
            self._projector,
        )


def projector_model_provider(
    pre_process: bool = True, post_process: bool = True, vp_stage=None
):
    """miles ``--custom-model-provider-path``: frozen base + trainable projector."""
    from megatron.training import get_args  # pyright: ignore[reportMissingImports]

    args = get_args()
    cfg = from_miles_args(args)
    model = _build_base_model(args, pre_process, post_process, vp_stage)
    frozen = freeze_base_model(model)

    if pre_process:
        projector = build_projector(
            cfg, model.config.hidden_size, model.config.params_dtype
        )
        model.__dict__["_training_gym_projector"] = projector
        # Registered so Megatron's DDP and the optimizer see the parameters.
        model.add_module("embedding_projector", projector)
        if cfg.load:
            # Stashed on the projector because the saver is built later, from a
            # different hook, and only ever gets the model chunk: a resumed run
            # continues the numbering rather than overwriting the checkpoints
            # the previous one left in the same directory.
            projector._training_gym_loaded_iteration = load_projector_checkpoint(
                cfg.load, projector
            )
        model.__dict__["_training_gym_projector_merge"] = _ProjectorMerge(
            model, projector, cfg
        )
        trainable = sum(p.numel() for p in projector.parameters())
        logger.info(
            "projector-only training: froze %d base parameter tensors, "
            "%d trainable projector parameters",
            frozen,
            trainable,
        )
    else:
        # Later pipeline stages get no projector, but miles still splats the
        # batch's embedding tensors into every stage's forward, so they have to
        # be stripped before they reach ``GPTModel.forward``.
        model.__dict__["_training_gym_projector_merge"] = _ProjectorMerge(
            model, EmbeddingProjector(1, 1, 1, 1), cfg
        )

    return model


def projector_sft_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    """miles ``--rollout-function-path``: supervised samples carrying embeddings.

    miles' own ``sft_rollout.generate_rollout`` builds tokens and the loss mask
    from each conversation, which is all a text SFT run needs. This does the
    same and additionally lifts the dataset's embeddings out of
    ``Sample.metadata`` into ``multimodal_train_inputs``, the per-sample tensor
    dict miles concatenates across a packed batch — adding each sample's offset
    to keys ending in ``_positions`` — and splats into ``model.forward``, where
    the projector picks them up.
    """
    from miles.utils.mask_utils import (  # pyright: ignore[reportMissingImports]
        MultiTurnLossMaskGenerator,
    )
    from miles.utils.processing_utils import (  # pyright: ignore[reportMissingImports]
        load_tokenizer,
    )

    if evaluation:
        raise ValueError("projector_sft_rollout does not generate, so cannot evaluate")
    cfg = from_miles_args(args)
    tokenizer = load_tokenizer(
        args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        trust_remote_code=True,
    )
    mask_generator = MultiTurnLossMaskGenerator(
        tokenizer, tokenizer_type=args.loss_mask_type
    )

    samples = data_buffer.get_samples(args.rollout_batch_size)
    for sample in samples:
        (sample,) = sample
        messages = sample.prompt
        token_ids, loss_mask = mask_generator.get_loss_mask(
            messages, tools=sample.metadata.get("tools")
        )
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        embeddings = sample.metadata.get(cfg.embeddings_key)
        positions = sample.metadata.get(cfg.positions_key)
        if embeddings is None or positions is None:
            raise ValueError(
                f"sample {sample.index} carries no '{cfg.embeddings_key}'/"
                f"'{cfg.positions_key}' metadata; a projector-only run has "
                "nothing to train without embeddings."
            )
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32)
        if embeddings_t.shape[-1] != cfg.input_dim:
            raise ValueError(
                f"sample {sample.index} has {embeddings_t.shape[-1]}-wide "
                f"embeddings but the projector expects {cfg.input_dim}"
            )
        max_position = max(positions, default=-1)
        if max_position >= len(token_ids):
            raise ValueError(
                f"sample {sample.index} places an embedding at token "
                f"{max_position} of a {len(token_ids)}-token sequence"
            )
        sample.multimodal_train_inputs = {
            cfg.embeddings_key: embeddings_t,
            cfg.positions_key: torch.tensor(positions, dtype=torch.long),
        }
    return samples


class _ProjectorSaver:
    """Writes the projector after optimizer steps, including the run's last one.

    The before-train-step hook is the only hook miles offers on the training path
    (``custom_megatron_post_save_hook`` fires off Megatron's own full-model save,
    which a projector-only run never triggers). Saving *in* that hook would only
    ever persist state one step stale and would miss the final step entirely, so
    the hook is used just to get hold of the optimizer, and the write hangs off
    ``optimizer.step``: counted in optimizer steps, so ``save_interval`` means
    what it says. ``projector_latest.pt`` is refreshed on every applied step, so
    the artifact survives a run that performs a different number of steps than
    ``train_iters`` predicted; ``save_interval`` and that prediction decide only
    which steps keep a numbered file too.

    Hanging a collective (the projector gradient all-reduce) off ``optimizer.step``
    is safe because the hook is per-rank: the pinned image calls
    ``custom_before_train_step_hook`` unconditionally inside
    ``megatron_utils/model.py``'s train step, with no rank guard, so every
    training rank installs this wrapper. A rank that has no projector to reduce
    would break that symmetry, so ``__init__`` refuses to install instead.
    """

    def __init__(self, args, model, optimizer) -> None:
        self._cfg = from_miles_args(args)
        self._save_dir = self._cfg.save_dir or os.path.join(
            str(args.save or "/checkpoints"), "projector"
        )
        chunks = model if isinstance(model, list) else [model]
        self._projectors = [p for c in chunks if (p := get_projector(c)) is not None]
        if not self._projectors:
            # Every rank's chunk holds a projector replica (the provider builds
            # one, and PP>1 is rejected so there is only ever one stage). If one
            # rank found none, its peers would enter the all-reduce below alone:
            # fail here rather than hang there, or train replicas that silently
            # diverge and write a checkpoint from one of them.
            raise RuntimeError(
                "projector-only training installed no saver: none of the "
                f"{len(chunks)} model chunk(s) handed to miles' "
                "before-train-step hook holds an EmbeddingProjector. The "
                "custom model provider is what attaches it — check that "
                "custom_model_provider_path points at "
                "projector_model_provider."
            )
        for chunk in chunks:
            direct = chunk.__dict__.get("_training_gym_projector")
            logger.info(
                "projector lookup: chunk type=%s bare __dict__ lookup=%s resolved=%s",
                type(chunk).__name__,
                "found" if isinstance(direct, EmbeddingProjector) else "None",
                "found" if get_projector(chunk) is not None else "NOT FOUND",
            )
        self._sequence_parallel = bool(getattr(args, "sequence_parallel", False))
        self._total_steps = int(getattr(args, "train_iters", 0) or 0)
        self._steps = 0
        self._attempts = 0
        # Steps this run inherited from the checkpoint it resumed, so file names
        # and the recorded ``iteration`` stay absolute. The counters above stay
        # relative to this run: ``save_interval`` is this run's cadence, and
        # ``train_iters`` counts this run's steps alone.
        self._step_offset = int(
            getattr(self._projectors[0], "_training_gym_loaded_iteration", 0) or 0
        )
        self._original_step = optimizer.step
        optimizer.step = self._step

    def _step(self, *fargs, **fkwargs):
        for projector in self._projectors:
            all_reduce_projector_grads(projector, self._sequence_parallel)
        out = self._original_step(*fargs, **fkwargs)
        self._log_replica_state()
        self._attempts += 1
        # Megatron returns (success, grad_norm, num_zeros); a step that did not
        # apply (gradient overflow) left the projector unchanged, so it does not
        # advance the interval — but it still consumed one of the run's steps,
        # which is what decides whether this was the last chance to write.
        if not (isinstance(out, tuple) and out and out[0] is False):
            self._steps += 1
            # Every applied step refreshes projector_latest.pt, so the run leaves
            # what it trained however many steps it turns out to take;
            # ``train_iters`` only decides which steps also keep a numbered file.
            self._save(
                numbered=should_save_projector(
                    self._steps,
                    self._attempts,
                    self._total_steps,
                    self._cfg.save_interval,
                )
            )
        return out

    def _log_replica_state(self) -> None:
        for projector in self._projectors:
            log_projector_replica(projector, self._step_offset + self._steps + 1)

    def _save(self, numbered: bool = True) -> None:
        if not is_projector_writer():
            return
        write_projector_checkpoint(
            self._cfg,
            self._save_dir,
            self._projectors[0],
            self._step_offset + self._steps,
            numbered=numbered,
        )


def save_projector_checkpoint(
    args, rollout_id: int, step_id: int, model, optimizer=None, opt_param_scheduler=None
) -> None:
    """miles before-train-step hook: arrange for projector-only checkpoints.

    Idempotent — the hook fires before every train step and the saver is
    installed on the first one, once per optimizer.
    """
    if optimizer is None or getattr(optimizer, "_training_gym_projector_saver", None):
        return
    optimizer._training_gym_projector_saver = _ProjectorSaver(args, model, optimizer)
