"""The rollout half of a stitch run: a Modal Flash pool of SGLang replicas.

A stitch run is two halves that meet at the bulletin board. The trainer
(:mod:`modal_training_gym.train_recipes.stitch_recipe.train`) publishes sparse
weight deltas; the pool described here serves rollouts and applies those deltas
in place, so its configuration is *serving* configuration — which is why the
engine args come from the same :class:`SglangRecipe` the deploy path uses,
rather than a second copy of the same flags.

What this adds on top of ``SglangRecipe`` is only what a weight-syncing pool
needs and a static deployment does not: the forked SGLang runtime that exposes
``/stage_weight_update``, the sidecar's apply/commit behaviour, and the Flash
pool's shape.
"""

from __future__ import annotations

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    DEFAULT_SGLANG_RUNTIME,
    SGLangRuntime,
)

__all__ = ["StitchServeConfig"]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchServeConfig:
    """Configuration for the Flash rollout pool a stitch run trains against.

    ## Fields

    sglang : SglangRecipe
        Engine configuration for each replica — context length, KV memory
        fraction, prefill budget, and ``extra_server_args`` for anything else.
        ``gpu`` and ``tp`` describe one replica (the pool runs
        ``gpu:tp`` per container), and the trainer's
        ``rollout_num_gpus_per_engine`` is derived from ``tp`` so the two halves
        can't disagree. ``sglang_image``, ``deploy_strategy``, and
        ``environment_name`` are deployment fields this pool doesn't use and
        must be left at their defaults — the image comes from ``runtime``.
    runtime : SGLangRuntime
        The SGLang fork the pool serves on. It exposes ``/stage_weight_update``,
        which stitch's engine adapter drives to apply a delta without a restart;
        upstream SGLang has no such endpoint.
    concurrency : int
        Rollout requests one replica accepts, which also sets the engine's
        ``--max-running-requests`` and decode CUDA-graph batch size. A rollout
        step issues ``rollout_batch_size * n_samples_per_prompt`` requests
        across the pool.
    min_containers : int
        Replicas kept warm. Worth setting to a full rollout step's worth of
        concurrency for a large model: Flash holds requests while a replica cold
        starts, but a big MoE can take minutes to load — longer than the
        trainer's own HTTP retry budget for a rollout request.
    max_containers : int | None
        Upper bound on replicas. Default ``None`` (Flash scales freely).
    proxy_regions : list[str]
        Flash gateway proxy regions.
    commit_mode : str
        How a replica applies a staged version. ``"in_place"`` pauses, applies,
        and resumes, relying on per-version request stamping for isolation;
        ``"quiesce"`` drains in-flight requests first.
    delta_update_mode : str
        Where the engine applies a delta. ``"disk"`` patches a host-local
        checkpoint copy and reloads from it; ``"cpu"`` applies against a pinned
        CPU weight cache (needs ``--enable-cpu-weight-cache`` in the engine
        args).
    flush_cache_on_commit : bool
        Drop the engine's KV cache when a version commits. Off by default: the
        engine namespaces cached prefixes per version, so flushing only costs
        recompute.
    delta_volume_name : str
        Modal Volume backing the bulletin board. Empty → derived from the
        recipe name at build time.
    bulletin_root : str
        Mount path of the bulletin volume, in both halves.
    served_checkpoint_path : str
        Local checkpoint the replicas serve, and the baseline every sparse delta
        is applied against. Empty → the trainer half's ``hf_checkpoint`` (which
        is what the checkpoint-prep step builds), so the two halves can't
        disagree on the byte-exact baseline.
    env : dict[str, str]
        Extra environment for the serving image — the sampler side of a
        quantization contract (e.g. the ``FLASHINFER_NVFP4_*`` mirror of the
        trainer's ``NVTE_NVFP4_*`` settings) lives here.
    memory : tuple[int, int] | None
        Host-RAM (request, limit) MiB for a replica. A CPU weight cache needs
        roughly twice the checkpoint resident, plus staging headroom.
    ephemeral_disk : int | None
        Container disk MiB, for a disk-mode replica's local checkpoint copy.
    """

    sglang: SglangRecipe = field(default_factory=lambda: SglangRecipe(gpu="H200"))
    runtime: SGLangRuntime = DEFAULT_SGLANG_RUNTIME

    # ── Flash pool shape ────────────────────────────────────────────────────
    concurrency: int = 64
    min_containers: int = 3
    max_containers: int | None = None
    proxy_regions: list[str] = field(default_factory=lambda: ["us-east"])

    # ── Weight-sync sidecar ─────────────────────────────────────────────────
    commit_mode: str = "in_place"
    delta_update_mode: str = "disk"
    flush_cache_on_commit: bool = False

    # ── Bulletin board (shared with the trainer half) ────────────────────────
    delta_volume_name: str = ""
    bulletin_root: str = "/delta-bulletin"

    # ── Served baseline + runtime env ────────────────────────────────────────
    served_checkpoint_path: str = ""
    env: dict[str, str] = field(default_factory=dict)
    memory: tuple[int, int] | None = None
    ephemeral_disk: int | None = None

    def __post_init__(self) -> None:
        defaults = SglangRecipe()
        for name in ("sglang_image", "deploy_strategy", "environment_name"):
            if getattr(self.sglang, name) != getattr(defaults, name):
                raise ValueError(
                    f"SglangRecipe.{name} is a deployment field the stitch rollout "
                    "pool does not use; it serves on the forked runtime from "
                    "StitchServeConfig.runtime, brought up with the training app."
                )
        if self.commit_mode not in ("in_place", "quiesce"):
            raise ValueError(
                f"commit_mode must be 'in_place' or 'quiesce', got {self.commit_mode!r}"
            )
        if self.delta_update_mode not in ("disk", "cpu"):
            raise ValueError(
                f"delta_update_mode must be 'disk' or 'cpu', got {self.delta_update_mode!r}"
            )

    @property
    def gpus_per_replica(self) -> int:
        """GPUs one replica serves on — the engine's tensor-parallel degree."""
        return self.sglang.tp or 1

    @property
    def gpu(self) -> str:
        return str(self.sglang.gpu)

    @property
    def startup_timeout(self) -> int:
        return self.sglang.startup_timeout

    def engine_args(self, *, model_name: str) -> dict[str, str]:
        """SGLang server args for a replica.

        Structural args the pool always needs, then :meth:`SglangRecipe.server_args`
        (context length, KV fraction, prefill budget, ``extra_server_args``) over
        the top. Nothing delta-specific is passed: the engine applies deltas
        behind ``/stage_weight_update``, driven by the sidecar.
        Precision is deliberately not passed: it comes from the served
        checkpoint's own quant config, so a quantized baseline (NVFP4) loads as
        exported rather than being coerced by a ``--dtype`` flag.
        """
        args = {
            "--cuda-graph-max-bs-decode": str(self.concurrency),
            "--max-running-requests": str(self.concurrency),
            "--trust-remote-code": "",
        }
        args.update(self.sglang.server_args(served_model_name=model_name))
        return args
