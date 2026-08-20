"""Projector-only training: the torch and Megatron parts both trainers share.

Imported inside the training containers only — it needs torch and Megatron.
Miles and slime are both Megatron trainers, so the projector module, its
initialization, the merge into the input embeddings and the projector-only
checkpoint format are identical between them; what differs (config arg, hook
paths, how a packed microbatch's per-sample tensors are laid out) lives in
``frameworks/<framework>/embedding_projector.py``.

The config that reaches this module travels as a
:class:`~modal_training_gym.common.projector_config.ProjectorSpec`, whose module
stays torch-free so the launching side can import it too.
"""

import logging
import math
import os

import torch  # pyright: ignore[reportMissingImports]  # torch is installed only in training images
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from modal_training_gym.common.projector_config import (
    LATEST_CHECKPOINT,
    ProjectorSpec,
)

logger = logging.getLogger(__name__)


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

    The projector is replicated and its gradients are reduced together, so ranks
    that start from different weights stay different forever — silently, since
    the updates do agree. Torch's default init draws from the ambient RNG, whose
    state depends on how the base model's build consumed it (TP/EP shards, MoE
    expert counts), so it is not relied on here: a private generator plus
    PyTorch's own ``nn.Linear`` bounds gives identical weights on every rank
    without a collective, and without assuming the projector exists on a rank
    that could take part in one.
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


def build_projector(
    cfg: ProjectorSpec, hidden_size: int, params_dtype
) -> EmbeddingProjector:
    """The projector ``cfg`` describes, on this rank's device and ready to train."""
    projector = EmbeddingProjector(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim or hidden_size,
        num_layers=cfg.num_layers,
    )
    init_projector(projector, cfg.init_seed, cfg.output_scale)
    projector.to(device=torch.cuda.current_device(), dtype=params_dtype)
    projector.requires_grad_(True)
    return projector


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


def projector_parameters(projector: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Projector parameters in an order identical on every rank.

    A per-parameter gradient collective has to be enqueued in the same order on
    every rank or the collectives pair up wrongly and deadlock.
    ``named_parameters`` walks the module tree deterministically and every rank
    holds the same replica, but it is sorted anyway so the guarantee does not
    rest on that.
    """
    return sorted(projector.named_parameters(), key=lambda item: item[0])


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

    ``main_grad`` is read whole. The miles recipes reject
    ``use_distributed_optimizer`` for that reason; slime sets it unconditionally
    in ``_set_default_megatron_args``, so on that backend the recipe's field is
    advisory and what makes this correct is *when* it runs: the hook wraps
    ``optimizer.step``, where ``main_grad`` still views the whole bucket for the
    projector's own data-parallel group. Two things make that sum right rather
    than lucky: data-parallel groups are formed across ranks sharing a
    tensor-parallel rank, so every rank in the group owns the same shard region
    of the bucket, and the optimizer reads only that region — the unreduced
    remainder outside it is never used. A run whose replicas diverged here
    would show it immediately — the fingerprint
    :func:`log_projector_replica` writes per tensor-parallel rank is compared on
    every validated run and has stayed byte-identical.
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


def check_projector_weights(projector: nn.Module) -> str:
    """Describe the projector's weights, and refuse an all-zero one.

    Called on the first forward rather than at build time, because what this
    catches happens in between: Megatron allocates the DDP parameter buffer the
    parameters are re-pointed into inside
    ``torch_memory_saver.region(tag="param_buffer", enable_cpu_backup=False)``,
    and a paused region without a backup comes back zeroed. With the base
    frozen, that buffer holds only the projector, so offloading the train model
    zeroes exactly the weights being trained. That is not a bad initialization
    that trains slowly: an all-zero projector writes exactly-zero rows for every
    embedding the encoder produced, so the run trains on no signal at all, and
    on real hardware it surfaced as NaN gradients rather than as its cause. The
    miles recipes reject the flag that does it, and slime's engine-free
    ``debug_train_only`` mode never offloads; this makes any other route to the
    same state say so.
    """
    parts, all_zero = [], True
    for name, param in projector_parameters(projector):
        detached = param.detach().float()
        rms = float(detached.pow(2).mean().sqrt())
        all_zero = all_zero and rms == 0.0
        finite = bool(torch.isfinite(detached).all())
        parts.append(f"{name} rms={rms:.6g} absmax={float(detached.abs().max()):.6g}")
        if not finite:
            raise ValueError(
                f"projector parameter '{name}' is not finite at the first "
                "forward; the projector is initialized from a seed and cannot "
                "reach this state by training."
            )
    if all_zero:
        raise ValueError(
            "every projector parameter is exactly zero at the first forward, "
            "which the seeded initialization cannot produce. The train model "
            "was most likely offloaded: Megatron's parameter buffer (holding "
            "only the projector, since the base is frozen) lives in a "
            "torch_memory_saver region with no backup, so pausing it zeroes the "
            "projector. Launch with no_offload_train=True."
        )
    return ", ".join(parts)


_grads_logged: set[str] = set()
_merges_logged: set[str] = set()


def _log_merge_once(key: str) -> bool:
    """Whether to emit the merge diagnostic ``key`` in this process.

    :func:`scatter_projected` runs from a forward hook on ``model.embedding``, so
    once per micro-batch per rank, and again for every recomputation of the
    embedding. The scale line reduces the whole embedding tensor and pulls the
    result to Python, which drains the stream, so it is worth exactly once — like
    every other diagnostic here. What these say (which ranks hold projected
    positions, at what scale the projector writes) does not change between steps.
    """
    if key in _merges_logged:
        return False
    _merges_logged.add(key)
    return True


def log_incoming_grad(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Log the first gradient that reaches ``tensor``, once per ``name``.

    A projector-only run detects nothing but the projector's own gradients:
    Megatron's grad-norm check covers the DDP buckets, and with the base frozen
    the only bucket is the projector's. So a base whose backward hands the
    embedding stream an ``inf`` (bf16 overflow) or a NaN is reported as *the
    projector's* NaN, with nothing to distinguish it from the projector itself
    producing one. Logging what arrives at the merge separates the two, and one
    line per tensor per process is not worth gating behind a flag.
    """
    if not tensor.requires_grad or name in _grads_logged:
        return tensor
    _grads_logged.add(name)

    def _log(grad: torch.Tensor) -> None:
        detached = grad.detach().float()
        logger.info(
            "projector backward: grad into %s rms=%.6g absmax=%.6g nan=%d inf=%d",
            name,
            float(detached.pow(2).mean().sqrt()),
            float(detached.abs().max()),
            int(detached.isnan().sum()),
            int(detached.isinf().sum()),
        )

    tensor.register_hook(_log)
    return tensor


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


def scatter_projected(
    sequence_parallel: bool,
    embeddings_out: torch.Tensor,
    embeds: torch.Tensor,
    positions: torch.Tensor,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    """Write ``embeds`` into ``embeddings_out`` at ``positions`` (sequence-first).

    ``LanguageModelEmbedding.forward`` returns ``[sequence, batch, hidden]``, and
    under sequence parallelism the sequence dimension is already this rank's
    shard (either reduce-scattered inside the vocab embedding or scattered right
    after it), so positions are rebased onto the shard and the ones belonging to
    other ranks dropped. Both trainers pack a microbatch into one sequence, so
    the batch dimension is 1. Context parallelism would shard the sequence a
    second way, with its own chunking, which is why the recipes reject ``CP > 1``.

    ``positions`` index the *micro*-batch that reached this forward, so a
    position past the packed sequence means the rebasing contract changed and it
    raises rather than being dropped: under sequence parallelism it would
    otherwise fall off every rank's shard and the run would train with no
    projector gradient at all.
    """
    import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

    def unmerged():
        """``embeddings_out``, with the projector still in the graph at zero weight."""
        if projector is None:
            return embeddings_out
        return log_incoming_grad(
            embeddings_out + projector_graph_tap(projector, embeddings_out.dtype),
            "the embeddings of a rank with no projected positions",
        )

    if positions.numel() == 0:
        return unmerged()
    if embeddings_out.shape[1] != 1:
        raise ValueError(
            "projector training expects the trainer's packed layout, one sequence "
            f"per microbatch, but the embedding output has batch "
            f"{embeddings_out.shape[1]}"
        )
    s_local = embeddings_out.shape[0]
    tp_size = ps.get_tensor_model_parallel_world_size()
    sharded = sequence_parallel and tp_size > 1
    packed_len = s_local * tp_size if sharded else s_local
    outside = int(((positions < 0) | (positions >= packed_len)).sum())
    if outside:
        raise ValueError(
            f"{outside} of {int(positions.numel())} projector position(s) fall "
            f"outside the micro-batch's {packed_len}-token packed sequence "
            f"(min {int(positions.min())}, max {int(positions.max())}). "
            "Positions have to be offsets into the packed micro-batch that "
            "reaches forward."
        )
    if sharded:
        local = positions - ps.get_tensor_model_parallel_rank() * s_local
        keep = (local >= 0) & (local < s_local)
        local, embeds = local[keep], embeds[keep]
    else:
        local = positions
    if _log_merge_once("positions"):
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
    if _log_merge_once("scale"):
        with torch.no_grad():
            # The scales have to be comparable: a projector row far above the
            # base's embedding scale is out of the decoder's distribution and
            # overflows the forward, which shows up as NaN gradients on exactly
            # the merging ranks.
            logger.info(
                "projector merge scale: base rms=%.6g absmax=%.6g, projected "
                "rms=%.6g absmax=%.6g",
                float(embeddings_out.detach().float().pow(2).mean().sqrt()),
                float(embeddings_out.detach().float().abs().max()),
                float(kept.detach().float().pow(2).mean().sqrt()),
                float(kept.detach().float().abs().max()),
            )
    log_incoming_grad(kept, "the projector's output")
    merged[local, 0] = kept
    return log_incoming_grad(merged, "the merged embeddings (from the frozen base)")


def get_projector(model) -> EmbeddingProjector | None:
    """The projector attached to ``model``, if this rank holds one.

    Tolerates wrapping: by the time the trainer's hooks run, a model chunk is a
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


def load_projector_checkpoint(path: str, projector: EmbeddingProjector) -> int:
    """Load a projector checkpoint (file or directory) into ``projector``.

    Returns the iteration it was saved at. A directory resolves to the run's
    ``projector_latest.pt``.
    """
    resolved = os.path.join(path, LATEST_CHECKPOINT) if os.path.isdir(path) else path
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    projector.load_state_dict(state["state_dict"])
    iteration = int(state.get("iteration", 0))
    logger.info("loaded projector checkpoint %s (iteration %d)", resolved, iteration)
    return iteration


def projector_output_dim(projector) -> int:
    """The projector's actual output width, read off its last linear layer."""
    linears = [m for m in projector.modules() if isinstance(m, nn.Linear)]
    if linears:
        return int(linears[-1].out_features)
    return int(projector.norm.normalized_shape[0])


def write_projector_checkpoint(
    cfg: ProjectorSpec, save_dir: str, projector, step: int, numbered: bool = True
) -> None:
    """Write ``projector``'s state dict plus enough config to rebuild it.

    Optimizer state is not saved — the projector's Adam moments are cheap to
    rebuild, and reading Megatron's distributed optimizer state for a replicated
    module would need the base model's sharding machinery. The frozen base is not
    saved either: it stays byte-identical to the run's ``hf_checkpoint``.

    ``projector_latest.pt`` is always written; ``numbered`` additionally keeps the
    step's own ``projector_iter_*.pt``, which is what ``save_interval``
    schedules. Refreshing ``projector_latest.pt`` after every applied step is
    what makes "a finished run leaves the adapter it produced" independent of any
    prediction of how many steps the run will perform — tens of megabytes,
    against a step of a frozen giant's forward and backward.
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
            "output_dim": projector_output_dim(projector),
            # Not needed to rebuild the shape, but needed to rebuild the
            # *starting point*: an eval's untrained baseline is only the state
            # training began from if it inits from the same seed and scale this
            # run used.
            "init_seed": cfg.init_seed,
            "output_scale": cfg.output_scale,
        },
    }
    latest = os.path.join(save_dir, LATEST_CHECKPOINT)
    torch.save(state, latest)
    if numbered:
        path = os.path.join(save_dir, f"projector_iter_{step:07d}.pt")
        torch.save(state, path)
        logger.info("saved projector checkpoint %s", path)
    else:
        logger.info("refreshed projector checkpoint %s (step %d)", latest, step)


def is_projector_writer() -> bool:
    """True on exactly one rank per projector replica group.

    The projector is replicated, so every rank holds the same weights and one
    writer is enough; the context-parallel rank is checked explicitly because
    Megatron's ``get_data_parallel_rank`` defaults to
    ``with_context_parallel=False`` and would leave every ``cp_rank`` writing the
    same file.
    """
    import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

    return (
        ps.get_tensor_model_parallel_rank() == 0
        and ps.get_data_parallel_rank() == 0
        and ps.get_context_parallel_rank() == 0
        and ps.get_expert_model_parallel_rank() == 0
    )


def log_projector_replica(projector: nn.Module, step: int) -> None:
    """Log a per-rank fingerprint of the projector, so replicas can be compared.

    Every rank holds a replica whose gradients are reduced together, so after a
    step the weights must be bit-identical across ranks; a divergence here means
    the reduction is misplaced and the ranks have silently forked.
    """
    import megatron.core.parallel_state as ps  # pyright: ignore[reportMissingImports]

    name, param = projector_parameters(projector)[0]
    with torch.no_grad():
        checksum = float(param.detach().float().abs().sum().item())
        norm = float(param.detach().float().norm().item())
    logger.info(
        "projector replica after step %d: tp_rank=%d dp_rank=%d %s "
        "norm=%.8f abs_sum=%.8f",
        step,
        ps.get_tensor_model_parallel_rank(),
        ps.get_data_parallel_rank(),
        name,
        norm,
        checksum,
    )
