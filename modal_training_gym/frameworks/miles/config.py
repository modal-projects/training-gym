"""Configuration for Miles RL training on Modal.

Miles is an RLVR training framework (built on Megatron + Ray) distributed as
a Docker image (`radixark/miles:…`) containing both the trainer and its
patched Megatron-LM stack. Users customize a run by providing "recipe args"
— a block of Miles CLI flags describing the model architecture + training
hyperparameters. The launcher adds the cluster + Ray glue.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from modal_training_gym.common import GPUType
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class MilesFrameworkConfig:
    """Miles configuration, including Modal infrastructure.

    The bulk of the training config lives in `recipe_args` — a string of
    Miles CLI flags copied from the upstream `recipes/*.args` files (one
    flag per line, `#`-comments allowed). The launcher parses it via
    `shlex`, appends `extra_args`, and tacks on the framework-level flags
    (hf checkpoint, save dir, actor/rollout topology) at the end.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "H100"
    # Miles ships a pre-built image with patched Megatron-LM + the trainer.
    miles_image: str = "radixark/miles:dev-202603231227"
    # Extra commands appended to the image build (e.g. apply a local patch,
    # pip-install a pin). Each entry is passed to `Image.run_commands(...)`.
    image_run_commands: list[str] = field(default_factory=list)

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 1

    @property
    def gpus_per_node(self) -> int:
        # Our multi-node training set-up requires 8 GPUs per node.
        return 8

    # ── Recipe + overrides ───────────────────────────────────────────────────
    # Raw Miles CLI flag block — comments + blank lines OK, shlex-parsed.
    recipe_args: str = ""
    # Extra flags appended after recipe_args, before the framework-enforced flags.
    extra_args: str = ""
    # Inline YAML overrides. When non-empty, written to /tmp/<run_id>.yaml
    # on the head rank and passed via `--custom-config-path`.
    custom_config_yaml: str = ""

    # ── Actor / rollout topology ─────────────────────────────────────────────
    # `colocate=True` reuses a single compute pool for actor + rollout.
    # When False, the launcher carves `rollout_num_gpus` out of the cluster.
    colocate: bool = True
    # When unset, actor_nodes defaults to `n_nodes` (colocated) or
    # `n_nodes - 1` (non-colocated).
    actor_nodes: int | None = None
    rollout_num_gpus: int | None = None

    # ── Modal app tags ───────────────────────────────────────────────────────
    app_tags: dict = field(default_factory=dict)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_arg_text(arg_text: str) -> str:
        """Strip `#` comments + blank lines from a recipe args block."""
        lines: list[str] = []
        for raw in arg_text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    def parsed_recipe_args(self) -> list[str]:
        """Shlex-parse `recipe_args` + `extra_args` into a flat argv list."""
        import shlex

        cleaned = "\n".join(
            part
            for part in (self._clean_arg_text(self.recipe_args), self._clean_arg_text(self.extra_args))
            if part
        )
        return shlex.split(cleaned) if cleaned else []

    def resolved_actor_nodes(self) -> int:
        if self.actor_nodes is not None:
            return self.actor_nodes
        if self.colocate:
            return self.n_nodes
        if self.n_nodes < 2:
            raise ValueError(
                "Non-colocated rollout needs spare cluster capacity. "
                "Either set colocate=True or bump n_nodes >= 2."
            )
        return self.n_nodes - 1

    def resolved_rollout_num_gpus(self) -> int | None:
        if self.colocate:
            return None
        if self.rollout_num_gpus is not None and self.rollout_num_gpus > 0:
            return self.rollout_num_gpus
        spare = (self.n_nodes - self.resolved_actor_nodes()) * self.gpus_per_node
        if spare < 1:
            raise ValueError(
                f"Non-colocated rollout has no spare GPUs — n_nodes={self.n_nodes}, "
                f"actor_nodes={self.resolved_actor_nodes()}, "
                f"gpus_per_node={self.gpus_per_node}."
            )
        return spare


class MilesConfig:
    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: MilesFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: MilesFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or MilesFrameworkConfig()

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_miles_app

        return build_miles_app(miles=self, name=name)
