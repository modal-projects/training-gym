"""The trainer half of a stitch run: miles in publish-only disaggregated mode.

This is a :class:`MilesRecipe` — same flags, same model/dataset/wandb converters
— with the ``miles_disagg`` deltas on top: no local rollout engines, sparse
weight deltas published to a bulletin board instead of an NCCL broadcast, and
stitch's request/publish hooks wired in.

Deriving from ``MilesRecipe`` is the point: a stitch run's trainer *is* a miles
trainer, so its ~100 flags are maintained in one place, and
:class:`~modal_training_gym.train_recipes.stitch_recipe.recipe.StitchRecipe`
takes the trainer as a field rather than being one.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.gpu_allocation import (
    GpuAllocation,
    validate_megatron_actor_parallelism,
)
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    MEGATRON_PATH,
    MILES_IMAGE_TAG,
    MILES_REPO_REF,
    MILES_REPO_URL,
)

__all__ = ["HOOK_CONFIG_FIELDS", "StitchTrainConfig"]

# Fields only stitch's own hooks read, off the trainer's args namespace. miles
# defines no such CLI flags, and its parser is Megatron's parse_known_args —
# passing them as flags drops them silently (the hooks then see their fallbacks,
# e.g. 60 request retries instead of the configured budget). The launcher merges
# these into ``custom_config_path`` instead, which miles setattrs onto args.
HOOK_CONFIG_FIELDS = frozenset(
    {
        "rollout_request_weight_version_mode",
        "rollout_request_weight_version_lag",
        "rollout_request_retry_attempts",
        "rollout_request_retry_sleep",
        "rollout_session_affinity_header",
    }
)

# Inherited MilesRecipe fields that must NOT reach this trainer's command line:
# they configure a local rollout engine, which a disaggregated run does not have
# (the Flash pool's engines are configured by the serving half instead).
_TRAINER_DROP = frozenset(
    {
        "sglang_mem_fraction_static",
        "sglang_config",
        "sglang_lora_backend",
        "sglang_lora_use_virtual_experts",
        # Build / preparation instructions, not miles flags.
        "miles_repo_url",
        "miles_repo_ref",
        "megatron_runtime_patches",
        "served_checkpoint_format",
        "bf16_checkpoint_path",
        "prep_env",
        "ephemeral_disk",
    }
    | HOOK_CONFIG_FIELDS
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchTrainConfig(MilesRecipe):
    """miles trainer for a disaggregated stitch run.

    Don't see the flag you need? Every :class:`MilesRecipe` field applies here,
    and any additional class attribute becomes a miles CLI flag.
    """

    # ── The miles fork + image that speak the bulletin protocol ─────────────
    docker_image: str = MILES_IMAGE_TAG
    miles_repo_url: str = MILES_REPO_URL
    miles_repo_ref: str = MILES_REPO_REF
    # git patches applied to the Megatron source tree at container start, by
    # absolute in-image path (see ``pins.MEGATRON_PATCH_DIR``).
    megatron_runtime_patches: list[str] = field(default_factory=list)

    gpu_type: str = "B200"
    ephemeral_disk: int | None = None

    # ── Checkpoint preparation (the served baseline) ──────────────────────
    # ``hf_checkpoint`` is what the pool serves and what deltas apply against, so
    # for a quantized run it is a *prepared local dir*, not a repo id: the app's
    # prepare_checkpoints step materializes BF16 masters at
    # ``bf16_checkpoint_path`` from ``source_hf_checkpoint`` and, when
    # ``served_checkpoint_format`` is nvfp4, converts them with miles' TE-direct
    # quantizer under ``prep_env``. Both sides of the export/serve pair must use
    # the same quantizer contract or the first delta apply fails on a checksum.
    served_checkpoint_format: str = "bf16"
    bf16_checkpoint_path: str = ""
    prep_env: dict[str, str] = field(default_factory=dict)

    # Inline dict → a node-local YAML file every Ray actor re-reads at the same
    # path (per-launch tmpdirs differ across nodes).
    te_precision_config_file: dict | str | None = None

    # ── Disaggregation: the pool owns every rollout GPU ─────────────────────
    colocate: bool = False
    rollout_num_gpus: int | None = 0
    rollout_num_gpus_per_engine: int = 1
    # Rollouts go out over HTTP to the pool's Flash gateway (the launcher fills
    # in the URL per run) rather than to in-process engines.
    use_miles_router: bool = True
    rollout_endpoint_url: str | None = None

    megatron_to_hf_mode: str = "bridge"
    save_interval: int | None = None

    # ── Weight sync: publish sparse deltas to the bulletin board ────────────
    update_weight_transfer_mode: str = "disk-delta"
    update_weight_delta_encoding: str = "xor"
    update_weight_delta_checksum: str = "xxh3-128"
    # rank-0 publish hook: advance the pointer, commit the Volume, wake the pool.
    custom_update_weight_post_write_path: str = "cookbook.common.hooks.commit_and_wake"

    # ── Rollout request gating (stitch hooks) ───────────────────────────────
    # Pins each rollout request to a served weight version; a lagging replica
    # returns a retryable 409 so requests flow across a weight update.
    custom_rollout_request_hook_path: str = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    rollout_request_weight_version_mode: str = "exact"
    rollout_request_weight_version_lag: int = 0
    rollout_request_retry_attempts: int = 240
    rollout_request_retry_sleep: float = 1.0
    # The trainer hits the Flash gateway directly, which routes session affinity
    # on Modal-Session-ID; emit that so GRPO siblings co-locate.
    rollout_session_affinity_header: str = "Modal-Session-ID"

    # Synchronous publish by default: async bounded-lag rollouts need the trainer
    # to wake the pool the moment it publishes, and Flash wake is a lookup by
    # deployed app name — which a single-call ephemeral run does not have, so
    # replicas would only self-sync on their poll and fall outside the lag bound.
    async_mode: bool = False

    environment: dict = field(
        default_factory=lambda: {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NVSHMEM_DISABLE_NCCL": "1",
            "NCCL_TIMEOUT_MS": "360000000",
        }
    )

    @model_validator(mode="after")
    def _validate_gpu_allocation(self) -> "StitchTrainConfig":
        """Replaces MilesRecipe's allocator, which reads ``colocate=False`` as
        "the trainer also owns rollout GPUs" and so rejects
        ``rollout_num_gpus=0``. Here the engines are a separate Flash pool,
        sized by the serving half, so only the actor cluster is checked."""
        if self.served_checkpoint_format not in ("bf16", "nvfp4"):
            raise ValueError(
                "served_checkpoint_format must be 'bf16' or 'nvfp4', got "
                f"{self.served_checkpoint_format!r}"
            )
        # prepare_checkpoints always materializes the masters, and a bf16 run
        # serves them directly.
        if self.bf16_checkpoint_path and not self.bf16_checkpoint_path.startswith("/"):
            raise ValueError(
                "bf16_checkpoint_path must be an in-container path, got "
                f"{self.bf16_checkpoint_path!r}"
            )
        validate_megatron_actor_parallelism(self)
        return self

    @property
    def gpu_allocation(self) -> GpuAllocation:
        """The actor cluster's GPUs. Rollout GPUs live in the Flash pool and are
        not part of this allocation."""
        actor_gpus = self.actor_num_nodes * self.actor_num_gpus_per_node
        return GpuAllocation(
            actor_gpus=actor_gpus,
            critic_gpus=0,
            rollout_gpus=0,
            total_gpus=actor_gpus,
            total_nodes=self.actor_num_nodes,
            gpus_per_node=self.actor_num_gpus_per_node,
            rollout_num_gpus_per_engine=self.rollout_num_gpus_per_engine,
            rollout_engines=0,
            colocate=False,
        )

    @property
    def megatron_pythonpath(self) -> str:
        """Where the trainer's Ray actors find source-only ``megatron.training``."""
        return MEGATRON_PATH

    def _fields(
        self,
        dataset: DatasetConfig | None = None,
        model: ModelConfig | None = None,
    ) -> dict[str, Any]:
        fields = super()._fields(dataset=dataset, model=model)
        for name in _TRAINER_DROP:
            fields.pop(name, None)
        # bridge mode loads HF weights directly as the reference. The reference
        # is the BF16 masters, never the served base: for a quantized run those
        # differ, and loading a quantized checkpoint as the trainer's weights
        # would train the packed bytes.
        if self.megatron_to_hf_mode == "bridge" and not fields.get("ref_load"):
            for candidate in (self.bf16_checkpoint_path, fields.get("hf_checkpoint")):
                if isinstance(candidate, str) and candidate:
                    fields["ref_load"] = candidate
                    break
        return fields
