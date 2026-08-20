"""Projector-only training on slime: a trainable adapter over a frozen base model.

Runs inside the slime containers, not on the launching machine. Three entry
points, all reached by dotted path:

``projector_model_provider``
    slime's ``--custom-model-provider-path``. Builds the model the model script
    asks for (so Qwen3.6-35B-A3B's MoE routing, gated attention and qk-layernorm
    come from upstream verbatim), freezes every base parameter, and attaches an
    :class:`~modal_training_gym.common.embedding_projector.EmbeddingProjector` on
    the first pipeline stage. The projector maps externally computed per-token
    embeddings — protein/DNA encoder outputs, for instance — into the decoder's
    hidden space and writes them into the input embeddings at the positions the
    data gives.
``save_projector_checkpoint``
    slime's ``--custom-megatron-before-train-step-hook-path``. Installs a wrapper
    around the optimizer's ``step`` that writes the projector (a few hundred MB
    at most) instead of the base model, which at 35B is tens of gigabytes of
    sharded weights that never change during a projector-only run.
``projector_sft_rollout``
    slime's ``--rollout-function-path``. slime's own
    ``sft_rollout.generate_rollout`` plus the per-sample embedding tensors.

The trainable-parameter set is what makes this cheap: freezing the base before
Megatron builds the optimizer keeps optimizer state, gradient buffers and
checkpoints proportional to the projector, not to the model — LoRA-free, and
without the full-parameter memory bill.

Configuration arrives as one dict under the ``training_gym_projector`` arg (the
recipe writes it into ``extra_config``, whose keys slime applies to ``args``
after parsing), so no new slime CLI flags are needed.

What differs from the miles side (``frameworks/miles/embedding_projector.py``)
is position bookkeeping: miles offsets any ``multimodal_train_inputs`` key ending
in ``_positions`` when it packs a microbatch, while slime concatenates every key
verbatim (``slime/backends/megatron_utils/data.py``). So the rollout emits
sample-local positions plus a per-sample row count, and the merge rebases them
onto the packed sequence using the ``cu_seqlens`` slime already computed for
``PackedSeqParams``.
"""

import logging
import os

import torch  # pyright: ignore[reportMissingImports]  # torch is installed only in training images

from modal_training_gym.common.embedding_projector import (
    EmbeddingProjector as EmbeddingProjector,
    all_reduce_projector_grads,
    build_projector,
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
from modal_training_gym.frameworks.slime.projector_config import (
    ROW_COUNTS_KEY,
    ProjectorSpec,
    from_slime_args,
    should_save_projector,
)

logger = logging.getLogger(__name__)


def _build_base_model(args, pre_process: bool, post_process: bool, kwargs: dict):
    """Build the model slime would have built without this provider.

    The build happens with ``custom_model_provider_path`` still cleared, not just
    the lookup: re-entering this function would recurse until the stack
    overflows. Only the keyword arguments the underlying provider actually
    declares are forwarded, the way slime's own freeze wrapper does it, because
    the signature differs across Megatron versions (``vp_stage``, ``config``,
    ``pg_collection``).
    """
    import inspect

    from slime.backends.megatron_utils.model_provider import (  # pyright: ignore[reportMissingImports]
        get_model_provider_func,
    )

    saved = args.custom_model_provider_path
    args.custom_model_provider_path = None
    try:
        provider = get_model_provider_func(args)
        provider_kwargs = {"pre_process": pre_process, "post_process": post_process}
        params = inspect.signature(provider).parameters
        for key in ("vp_stage", "config", "pg_collection"):
            if key in params:
                provider_kwargs[key] = kwargs.get(key, None)
        return provider(**provider_kwargs)
    finally:
        args.custom_model_provider_path = saved


def rebase_positions(
    positions: torch.Tensor,
    row_counts: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Turn sample-local positions into positions in the packed sequence.

    ``positions`` and ``row_counts`` are the microbatch's per-sample tensors
    concatenated in sample order, and ``cu_seqlens`` is the prefix sum of the
    packed sample lengths slime handed to ``PackedSeqParams`` — so sample ``i``
    starts at ``cu_seqlens[i]`` and its rows are the ``row_counts[i]`` next
    entries of ``positions``.

    slime appends no padding entry to ``cu_seqlens`` on the non-``allgather_cp``
    path, so one boundary per sample plus the end is exactly what a well-formed
    microbatch has; a mismatch means the row-count bookkeeping and the packing
    disagree and the merge would silently write to the wrong tokens.
    """
    num_samples = int(row_counts.numel())
    if int(cu_seqlens.numel()) - 1 != num_samples:
        raise ValueError(
            f"{num_samples} projector sample(s) in this microbatch but "
            f"cu_seqlens describes {int(cu_seqlens.numel()) - 1}; the packed "
            "layout and the per-sample row counts disagree"
        )
    counts = row_counts.to(device=positions.device, dtype=torch.long)
    if int(counts.sum()) != int(positions.numel()):
        raise ValueError(
            f"row counts sum to {int(counts.sum())} but {int(positions.numel())} "
            "position(s) were passed"
        )
    starts = cu_seqlens[:-1].to(device=positions.device, dtype=torch.long)
    return positions + torch.repeat_interleave(starts, counts)


class _ProjectorMerge:
    """Moves the batch's embeddings from ``model.forward`` into the embedding layer.

    slime splats ``multimodal_train_inputs`` straight into ``model(**kwargs)``
    (``slime/backends/megatron_utils/model.py``), but ``GPTModel.forward`` takes
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
        self._original_forward = model.forward
        model.forward = self._forward
        if model.pre_process:
            model.embedding.register_forward_hook(self._embedding_hook)

    def _forward(self, *fargs, **fkwargs):
        embeddings = fkwargs.pop(self._embeddings_key, None)
        positions = fkwargs.pop(self._positions_key, None)
        row_counts = fkwargs.pop(ROW_COUNTS_KEY, None)
        if embeddings is None or not self._model.pre_process:
            return self._original_forward(*fargs, **fkwargs)
        if positions is None or row_counts is None:
            raise ValueError(
                f"'{self._embeddings_key}' was passed without "
                f"'{self._positions_key}'/'{ROW_COUNTS_KEY}'; the rollout must "
                "emit the token positions the embeddings occupy and how many "
                "rows each sample contributed."
            )
        if positions.numel() != embeddings.shape[0]:
            raise ValueError(
                f"{positions.numel()} projector position(s) for "
                f"{embeddings.shape[0]} embedding row(s)"
            )
        packed = fkwargs.get("packed_seq_params")
        if packed is None:
            raise ValueError(
                "projector training expects slime's packed batches, but this "
                "microbatch carries no packed_seq_params to rebase positions on"
            )
        self._pending = (
            embeddings,
            rebase_positions(positions, row_counts, packed.cu_seqlens_q),
        )
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


def _hide_projector_from_megatron_checkpoints(model) -> None:
    """Keep ``embedding_projector.*`` out of Megatron's checkpoint state dicts.

    The projector has to be a registered submodule for DDP and the optimizer to
    see its parameters, which also puts it in the model's sharded state dict —
    and slime loads the frozen base *after* the provider runs
    (``initialize_model_and_optimizer``), from a checkpoint that predates the
    projector and therefore has no such keys. Megatron resolves a sharded state
    dict against the checkpoint's metadata, so those keys would fail the load
    of an otherwise perfectly good base checkpoint.

    Dropping them from ``sharded_state_dict`` makes both directions ignore the
    projector: the base loads as if this were a plain run, and Megatron's own
    save (if a caller ever enables one) writes base weights only. The
    projector's own state travels through
    :func:`save_projector_checkpoint` and ``ProjectorSpec.load`` instead, which
    is a single small file rather than a resharded slice of the model.
    """
    prefix = "embedding_projector."
    original = model.sharded_state_dict

    def sharded_state_dict(*args, **kwargs):
        state = original(*args, **kwargs)
        for key in [k for k in state if k.startswith(prefix)]:
            del state[key]
        return state

    model.sharded_state_dict = sharded_state_dict


def projector_model_provider(
    pre_process: bool = True, post_process: bool = True, vp_stage=None, **kwargs
):
    """slime ``--custom-model-provider-path``: frozen base + trainable projector.

    ``vp_stage`` is declared explicitly because slime only forwards it to a
    custom provider that names it (it inspects the signature), and the base
    provider needs it whenever virtual pipeline parallelism is on.
    """
    from megatron.core import mpu  # pyright: ignore[reportMissingImports]
    from megatron.training import get_args  # pyright: ignore[reportMissingImports]

    args = get_args()
    cfg = from_slime_args(args)
    if mpu.get_context_parallel_world_size() > 1:
        # Context parallelism shards the packed sequence a second way, with its
        # own two-chunk interleave (``slice_with_cp``) and a ``cu_seqlens`` scaled
        # back up to global lengths, so a position no longer maps to a row of
        # this rank's embedding output. Rejected here rather than merging into
        # the wrong tokens.
        raise ValueError(
            "projector-only training does not support context parallelism; "
            "set context_parallel_size=1 on the recipe"
        )
    model = _build_base_model(
        args, pre_process, post_process, {"vp_stage": vp_stage, **kwargs}
    )
    frozen = freeze_base_model(model)

    if pre_process:
        projector = build_projector(
            cfg, model.config.hidden_size, model.config.params_dtype
        )
        model.__dict__["_training_gym_projector"] = projector
        # Registered so Megatron's DDP and the optimizer see the parameters.
        model.add_module("embedding_projector", projector)
        if cfg.load:
            load_projector_checkpoint(cfg.load, projector)
        model.__dict__["_training_gym_projector_merge"] = _ProjectorMerge(
            model, projector, cfg
        )
        _hide_projector_from_megatron_checkpoints(model)
        trainable = sum(p.numel() for p in projector.parameters())
        logger.info(
            "projector-only training: froze %d base parameter tensors, "
            "%d trainable projector parameters",
            frozen,
            trainable,
        )
    else:
        # Later pipeline stages get no projector, but slime still splats the
        # batch's embedding tensors into every stage's forward, so they have to
        # be stripped before they reach ``GPTModel.forward``.
        model.__dict__["_training_gym_projector_merge"] = _ProjectorMerge(
            model, EmbeddingProjector(1, 1, 1, 1), cfg
        )

    return model


def projector_sft_rollout(args, rollout_id: int, data_buffer, evaluation: bool = False):
    """slime ``--rollout-function-path``: supervised samples carrying embeddings.

    slime's own ``sft_rollout.generate_rollout`` builds tokens and the loss mask
    from each conversation, which is all a text SFT run needs. This does the same
    and additionally lifts the dataset's embeddings out of ``Sample.metadata``
    into ``multimodal_train_inputs``, the per-sample tensor dict slime
    concatenates across a packed microbatch and splats into ``model.forward``,
    where the projector picks them up.

    Positions stay sample-local here and are rebased in the merge: slime does no
    offsetting of its own, and this is the last place that would know a sample's
    length before the packing decides its offset.
    """
    from slime.utils.mask_utils import (  # pyright: ignore[reportMissingImports]
        MultiTurnLossMaskGenerator,
    )
    from slime.utils.processing_utils import (  # pyright: ignore[reportMissingImports]
        load_tokenizer,
    )

    if evaluation:
        raise ValueError("projector_sft_rollout does not generate, so cannot evaluate")
    cfg = from_slime_args(args)
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
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
            ROW_COUNTS_KEY: torch.tensor([len(positions)], dtype=torch.long),
        }
    return samples


class _ProjectorSaver:
    """Writes the projector after optimizer steps, including the run's last one.

    Saving inside the before-train-step hook would only ever persist state one
    step stale and would miss the final step entirely, so the hook is used just
    to get hold of the optimizer, and the write hangs off ``optimizer.step``:
    counted in optimizer steps, so ``save_interval`` means what it says, and the
    last step of the run always lands.

    Each step first sums the projector's gradients across the tensor-parallel
    group: under sequence parallelism every rank merges a disjoint slice of the
    packed sequence, so each holds only part of the replicated projector's
    gradient (see
    :func:`~modal_training_gym.common.embedding_projector.all_reduce_projector_grads`).
    """

    def __init__(self, args, model, optimizer) -> None:
        self._cfg = from_slime_args(args)
        self._sequence_parallel = bool(getattr(args, "sequence_parallel", False))
        self._save_dir = self._cfg.save_dir or os.path.join(
            str(args.save or "/checkpoints"), "projector"
        )
        chunks = model if isinstance(model, list) else [model]
        self._projectors = [p for c in chunks if (p := get_projector(c)) is not None]
        self._total_steps = int(getattr(args, "train_iters", 0) or 0)
        self._steps = 0
        self._attempts = 0
        self._original_step = optimizer.step
        optimizer.step = self._step

    def _step(self, *fargs, **fkwargs):
        for projector in self._projectors:
            all_reduce_projector_grads(projector, self._sequence_parallel)
        out = self._original_step(*fargs, **fkwargs)
        for projector in self._projectors:
            log_projector_replica(projector, self._steps + 1)
        self._attempts += 1
        # Megatron returns (success, grad_norm, num_zeros); a step that did not
        # apply (gradient overflow) left the projector unchanged, so it does not
        # advance the interval — but it still consumed one of the run's steps,
        # which is what decides whether this was the last chance to write.
        if not (isinstance(out, tuple) and out and out[0] is False):
            self._steps += 1
        if should_save_projector(
            self._steps, self._attempts, self._total_steps, self._cfg.save_interval
        ):
            self._save()
        return out

    def _save(self) -> None:
        if not self._projectors:
            logger.warning(
                "projector checkpoint skipped at step %d: no projector found on "
                "any model chunk handed to the before-train-step hook",
                self._steps,
            )
            return
        if not is_projector_writer():
            return
        write_projector_checkpoint(
            self._cfg, self._save_dir, self._projectors[0], self._steps
        )


def save_projector_checkpoint(
    args, rollout_id: int, step_id: int, model, optimizer=None, opt_param_scheduler=None
) -> None:
    """slime before-train-step hook: arrange for projector-only checkpoints.

    Idempotent — the hook fires before every train step and the saver is
    installed on the first one, once per optimizer.
    """
    if optimizer is None or getattr(optimizer, "_training_gym_projector_saver", None):
        return
    optimizer._training_gym_projector_saver = _ProjectorSaver(args, model, optimizer)
