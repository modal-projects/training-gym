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
import math
import os

import torch  # pyright: ignore[reportMissingImports]  # torch is installed only in training images
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from modal_training_gym.frameworks.miles.projector_config import (
    ProjectorSpec,
    from_miles_args,
    should_save_projector,
)

logger = logging.getLogger(__name__)

_LATEST = "projector_latest.pt"


class EmbeddingProjector(nn.Module):
    """MLP mapping external embeddings into the decoder's hidden space.

    Replicated on every rank rather than tensor-parallel: at a few hundred MB
    it is not worth sharding, and replication keeps the checkpoint a single
    plain state dict that loads anywhere.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"projector num_layers must be >= 1, got {num_layers}")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(embeddings))


def init_projector(
    projector: EmbeddingProjector, seed: int, output_scale: float = 1.0
) -> None:
    """Initialize from ``seed`` alone, so every replica agrees.

    ``output_scale`` goes into the final ``LayerNorm``'s weight, which sets the
    scale of everything the projector writes into the embedding stream. It has to
    be small: a ``LayerNorm`` output is unit-std by construction, while a decoder's
    token embeddings are ~1e-2 std, so a weight of 1.0 injects rows ~50-100x out
    of distribution — which is what produced NaN gradients on exactly the ranks
    holding projected positions on real hardware. It stays learnable, so training
    can grow it if the data wants more.

    The projector is replicated and its gradients are all-reduced, so ranks that
    start from different weights stay different forever — silently, since the
    updates do agree. Torch's default init draws from the ambient RNG, whose
    state depends on how the base model's build consumed it (TP/EP shards, MoE
    expert counts, the DSA indexer), so it is not relied on here: a private
    generator plus PyTorch's own ``nn.Linear`` bounds gives identical weights on
    every rank without a collective, and without assuming the projector exists
    on a rank that could take part in one.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for module in projector.modules():
            if isinstance(module, nn.Linear):
                bound = 1.0 / math.sqrt(module.weight.shape[1])
                for param in (module.weight, module.bias):
                    if param is None:
                        continue
                    param.copy_(
                        torch.empty(param.shape, dtype=torch.float32).uniform_(
                            -bound, bound, generator=gen
                        )
                    )
            elif isinstance(module, nn.LayerNorm):
                module.weight.fill_(output_scale)
                module.bias.zero_()


def freeze_base_model(model: nn.Module) -> int:
    """Freeze every parameter of ``model``; returns how many were frozen.

    Called before Megatron builds the optimizer, which skips parameters
    without ``requires_grad`` — that is what keeps optimizer state and
    gradient buffers proportional to the projector.
    """
    frozen = 0
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            frozen += 1
    return frozen


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


def projector_parameters(projector: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Projector parameters in an order identical on every rank.

    The gradient all-reduce below enqueues one collective per parameter, so the
    iteration order has to agree across ranks or the collectives pair up wrongly
    and deadlock. ``named_parameters`` walks the module tree deterministically
    and every rank holds the same replica, but it is sorted anyway so the
    guarantee does not rest on that.
    """
    return sorted(projector.named_parameters(), key=lambda item: item[0])


def projector_graph_tap(projector: nn.Module, dtype) -> torch.Tensor:
    """A differentiable, exactly-zero scalar tying ``projector`` into the graph.

    Needed on a rank whose sequence shard holds none of the projected positions:
    the base is frozen, so the projector is the only thing in the forward that
    requires grad, and dropping it there would leave the rank's loss without a
    ``grad_fn`` — it would then skip the backward its peers are waiting on.

    Built from the parameters rather than from the projector's output: parameters
    are finite by construction, whereas multiplying an activation by zero is a
    ``0 * inf`` trap that poisons the whole embedding tensor with NaN.
    """
    total = sum(param.sum() for _, param in projector_parameters(projector))
    return (total * 0.0).to(dtype)


def all_reduce_projector_grads(projector: nn.Module, sequence_parallel: bool) -> int:
    """Sum the replicated projector's gradients across the tensor-parallel group.

    Under sequence parallelism each tensor-parallel rank holds a disjoint slice
    of the packed sequence, so each computes the gradient of the same replicated
    weight with respect to a different subset of the projected positions. The
    whole-sequence gradient is their **sum**, not their average; the
    data-parallel average is DDP's job and is left alone. Without sequence
    parallelism every tensor-parallel rank merges every position and so already
    holds the whole gradient — summing there would scale it by the group size,
    which is why the flag is honored rather than assumed.

    Every rank enters this unconditionally, once per optimizer step, with a
    tensor of the same shape *and dtype* for every parameter — materializing a
    zero gradient when one is missing rather than skipping the collective, and
    reducing in fp32 so the collective's element size cannot depend on which
    buffer a rank happened to have (Megatron's ``main_grad`` is fp32 under
    ``accumulate_allreduce_grads_in_fp32``, a materialized ``param.grad``
    carries ``params_dtype``, and a collective whose ranks disagree is
    undefined). Riding Megatron's
    ``sequence_parallel`` attribute instead (which
    ``_allreduce_non_tensor_model_parallel_grads`` honors) is what deadlocked:
    that path is entered per-rank depending on what that rank's gradients look
    like, and a replicated module whose gradients are data-dependent cannot
    safely take part.

    ``main_grad`` is read whole, so this assumes the non-distributed optimizer —
    with ``use_distributed_optimizer`` it is a reduce-scattered shard of a
    bucket and summing shards across the group means nothing. The recipe rejects
    that flag rather than leaving the assumption implicit here.
    """
    import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

    if not sequence_parallel or ps.get_tensor_model_parallel_world_size() <= 1:
        return 0
    group = ps.get_tensor_model_parallel_group()
    reduced = 0
    for _, param in projector_parameters(projector):
        grad = getattr(param, "main_grad", None)
        if grad is None:
            if param.grad is None:
                # No gradient on this rank: still enter the collective, with a
                # zero of the right shape, and leave the reduced result where
                # the optimizer will read it.
                param.grad = torch.zeros_like(param)
            grad = param.grad
        buffer = grad.float()
        torch.distributed.all_reduce(
            buffer, op=torch.distributed.ReduceOp.SUM, group=group
        )
        grad.copy_(buffer)
        reduced += 1
    return reduced


def _scatter_projected(
    sequence_parallel, embeddings_out, embeds, positions, projector=None
):
    """Write ``embeds`` into ``embeddings_out`` at ``positions`` (sequence-first).

    ``LanguageModelEmbedding.forward`` returns ``[sequence, batch, hidden]``, and
    under sequence parallelism the sequence dimension is already this rank's
    shard (either reduce-scattered inside the vocab embedding or scattered right
    after it), so positions are rebased onto the shard and the ones belonging to
    other ranks dropped. miles packs a microbatch into one sequence, so the batch
    dimension is 1. Context parallelism would shard the sequence a second way,
    with its own chunking, which is why the recipe rejects ``CP > 1``.
    """
    import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

    def unmerged():
        """``embeddings_out``, with the projector still in the graph at zero weight."""
        if projector is None:
            return embeddings_out
        return embeddings_out + projector_graph_tap(projector, embeddings_out.dtype)

    if positions.numel() == 0:
        return unmerged()
    if embeddings_out.shape[1] != 1:
        raise ValueError(
            "projector training expects miles' packed layout, one sequence per "
            f"microbatch, but the embedding output has batch {embeddings_out.shape[1]}"
        )
    s_local = embeddings_out.shape[0]
    if sequence_parallel and ps.get_tensor_model_parallel_world_size() > 1:
        local = positions - ps.get_tensor_model_parallel_rank() * s_local
        keep = (local >= 0) & (local < s_local)
        local, embeds = local[keep], embeds[keep]
    else:
        local = positions
    logger.info(
        "projector merge: %d of %d position(s) on this rank's %d-token shard",
        int(local.numel()),
        int(positions.numel()),
        s_local,
    )
    if not local.numel():
        return unmerged()
    merged = embeddings_out.clone()
    kept = embeds.to(merged.dtype)
    with torch.no_grad():
        # The scales have to be comparable: a projector row far above the base's
        # embedding scale is out of the decoder's distribution and overflows the
        # forward, which shows up as NaN gradients on exactly the merging ranks.
        logger.info(
            "projector merge scale: base rms=%.6g absmax=%.6g, projected "
            "rms=%.6g absmax=%.6g",
            float(embeddings_out.detach().float().pow(2).mean().sqrt()),
            float(embeddings_out.detach().float().abs().max()),
            float(kept.detach().float().pow(2).mean().sqrt()),
            float(kept.detach().float().abs().max()),
        )
    merged[local, 0] = kept
    return merged


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
        projected = self._projector(
            embeddings.to(
                device=output.device,
                # The rollout emits fp32; the projector carries the model's
                # params_dtype, bf16 for these recipes.
                dtype=next(self._projector.parameters()).dtype,
            )
        )
        return _scatter_projected(
            self._model.config.sequence_parallel,
            output,
            projected,
            positions.to(output.device),
            self._projector,
        )


def get_projector(model) -> EmbeddingProjector | None:
    """The projector attached to ``model``, if this rank holds one.

    Tolerates wrapping: by the time miles' hooks run, a model chunk is a
    ``DistributedDataParallel(Float16Module(GPTModel))``, and ``__dict__``
    indexing does not follow the wrappers' ``__getattr__`` forwarding. The
    projector is registered with ``add_module("embedding_projector", ...)``, so
    unwrapping ``.module`` and finally scanning the module tree always reaches
    it.
    """
    obj, depth = model, 0
    while obj is not None and depth <= 8:
        projector = obj.__dict__.get("_training_gym_projector")
        if isinstance(projector, EmbeddingProjector):
            return projector
        obj = getattr(obj, "module", None)
        depth += 1
    modules = getattr(model, "modules", None)
    if callable(modules):
        for sub in modules():
            if isinstance(sub, EmbeddingProjector):
                return sub
    return None


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
        projector = EmbeddingProjector(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim or model.config.hidden_size,
            num_layers=cfg.num_layers,
        )
        init_projector(projector, cfg.init_seed, cfg.output_scale)
        projector.to(
            device=torch.cuda.current_device(), dtype=model.config.params_dtype
        )
        projector.requires_grad_(True)
        model.__dict__["_training_gym_projector"] = projector
        # Registered so Megatron's DDP and the optimizer see the parameters.
        model.add_module("embedding_projector", projector)
        if cfg.load:
            load_projector_checkpoint(cfg.load, projector)
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


def load_projector_checkpoint(path: str, projector: EmbeddingProjector) -> int:
    """Load a projector checkpoint (file or directory) into ``projector``.

    Returns the iteration it was saved at. A directory resolves to the run's
    ``projector_latest.pt``.
    """
    resolved = os.path.join(path, _LATEST) if os.path.isdir(path) else path
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    projector.load_state_dict(state["state_dict"])
    iteration = int(state.get("iteration", 0))
    logger.info("loaded projector checkpoint %s (iteration %d)", resolved, iteration)
    return iteration


def _is_projector_writer() -> bool:
    """True on exactly one rank per projector replica group.

    The projector is replicated, so every rank holds the same weights and one
    writer is enough; the context-parallel rank is checked explicitly because
    Megatron's ``get_data_parallel_rank`` defaults to ``with_context_parallel=False``
    and would leave every ``cp_rank`` writing the same file.
    """
    import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

    return (
        ps.get_tensor_model_parallel_rank() == 0
        and ps.get_data_parallel_rank() == 0
        and ps.get_context_parallel_rank() == 0
        and ps.get_expert_model_parallel_rank() == 0
    )


def write_projector_checkpoint(cfg: ProjectorSpec, save_dir: str, projector, step: int):
    """Write ``projector``'s state dict plus enough config to rebuild it.

    Optimizer state is not saved — the projector's Adam moments are cheap to
    rebuild, and reading Megatron's distributed optimizer state for a replicated
    module would need the base model's sharding machinery. The frozen base is not
    saved either: it stays byte-identical to the run's ``hf_checkpoint``.
    """
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "iteration": step,
        "state_dict": {
            k: v.detach().to("cpu") for k, v in projector.state_dict().items()
        },
        "config": {
            "input_dim": cfg.input_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            # The resolved width, not ``cfg.output_dim``, which is None whenever
            # the projector's output was sized from the model's hidden size — a
            # checkpoint has to describe its own shape.
            "output_dim": _projector_output_dim(projector),
        },
    }
    path = os.path.join(save_dir, f"projector_iter_{step:07d}.pt")
    torch.save(state, path)
    torch.save(state, os.path.join(save_dir, _LATEST))
    logger.info("saved projector checkpoint %s", path)


def _projector_output_dim(projector) -> int:
    """The projector's actual output width, read off its last linear layer."""
    linears = [m for m in projector.modules() if isinstance(m, nn.Linear)]
    if linears:
        return int(linears[-1].out_features)
    return int(projector.norm.normalized_shape[0])


class _ProjectorSaver:
    """Writes the projector after optimizer steps, including the run's last one.

    The before-train-step hook is the only hook miles offers on the training path
    (``custom_megatron_post_save_hook`` fires off Megatron's own full-model save,
    which a projector-only run never triggers). Saving *in* that hook would only
    ever persist state one step stale and would miss the final step entirely, so
    the hook is used just to get hold of the optimizer, and the write hangs off
    ``optimizer.step``: counted in optimizer steps, so ``save_interval`` means
    what it says, and the last step of the run always lands.
    """

    def __init__(self, args, model, optimizer) -> None:
        self._cfg = from_miles_args(args)
        self._save_dir = self._cfg.save_dir or os.path.join(
            str(args.save or "/checkpoints"), "projector"
        )
        chunks = model if isinstance(model, list) else [model]
        self._projectors = [p for c in chunks if (p := get_projector(c)) is not None]
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
        if should_save_projector(
            self._steps, self._attempts, self._total_steps, self._cfg.save_interval
        ):
            self._save()
        return out

    def _log_replica_state(self) -> None:
        """Log a per-rank fingerprint of the projector, so replicas can be compared.

        Every rank holds a replica whose gradients are all-reduced, so after a
        step the weights must be bit-identical across ranks; a divergence here
        means the all-reduce is misplaced and the ranks have silently forked.
        """
        import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

        for projector in self._projectors:
            name, param = projector_parameters(projector)[0]
            with torch.no_grad():
                checksum = float(param.detach().float().abs().sum().item())
                norm = float(param.detach().float().norm().item())
            logger.info(
                "projector replica after step %d: tp_rank=%d dp_rank=%d %s "
                "norm=%.8f abs_sum=%.8f",
                self._steps + 1,
                ps.get_tensor_model_parallel_rank(),
                ps.get_data_parallel_rank(),
                name,
                norm,
                checksum,
            )

    def _save(self) -> None:
        if not self._projectors:
            logger.warning(
                "projector checkpoint skipped at step %d: no projector found on "
                "any model chunk handed to the before-train-step hook",
                self._steps,
            )
            return
        if not _is_projector_writer():
            return
        write_projector_checkpoint(
            self._cfg, self._save_dir, self._projectors[0], self._steps
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
