"""Configuration for Harbor + Miles RL training on Modal.

Harbor extends Miles training with sandbox-based task evaluation: during
each rollout, the Miles custom agent function creates a Harbor sandbox,
runs the user-supplied agent, and returns the reward. Users supply:

- A Harbor agent class (by import path) that interacts with sandboxes.
- A dataset of Harbor task directories (each with an instruction file and
  a test/verification script).
- Standard Miles training hyperparameters (inherited from MilesFrameworkConfig).
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.frameworks.miles.config import (
    DATA_PATH,
    MilesFrameworkConfig,
)
from modal_training_gym.frameworks.miles.config import (
    _SKIP_FIELDS as _MILES_SKIP_FIELDS,
)

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration
    from modal_training_gym.common.wandb import WandbConfig

_HARBOR_SKIP_FIELDS = {
    "agent_import_path",
    "agent_model_name",
    "agent_kwargs",
    "environment_import_path",
    "sandbox_timeout_secs",
    "sandbox_idle_timeout_secs",
    "task_root",
    "task_glob",
    "instruction_path",
    "harbor_install_command",
    "miles_src_commit",
}

MILES_SRC_PATH = "/root/miles-src"

HARBOR_TRAIN_JSONL = DATA_PATH / "harbor_train.jsonl"

ROLLOUT_ROUTER_PORT = 30000
ROLLOUT_PROXY_PORT = 30001


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class HarborFrameworkConfig(MilesFrameworkConfig):
    """Harbor + Miles configuration.

    Inherits all Miles training settings (gpu, n_nodes, recipe_args, etc.)
    and adds Harbor agent/environment/task configuration. The launcher
    wires up the agent function, reward function, and rollout function
    automatically.
    """

    # ── Harbor agent ─────────────────────────────────────────────────────────
    agent_import_path: str = ""
    agent_model_name: str = "model"
    agent_kwargs: dict[str, Any] = field(default_factory=dict)

    # ── Harbor environment ───────────────────────────────────────────────────
    environment_import_path: str | None = None
    sandbox_timeout_secs: int = 60 * 30
    sandbox_idle_timeout_secs: int = 60 * 5

    # ── Harbor tasks (on the data volume) ────────────────────────────────────
    task_root: str = str(DATA_PATH / "tasks")
    task_glob: str = "*"
    instruction_path: str = "instruction.md"

    # ── Image ────────────────────────────────────────────────────────────────
    miles_image: str = "radixark/miles:dev-202604201238"
    # Miles source commit with --custom-agent-function-path support.
    # The base Docker image lacks this flag; the launcher clones and
    # reinstalls Miles from this commit to add it.
    miles_src_commit: str = "9a003644739f4e6dd509e2e8337e8ae7e571941c"
    harbor_install_command: str = (
        "uv pip install --system git+https://github.com/laude-institute/harbor.git"
    )

    def cli_args(self) -> list[str]:
        """Emit Miles CLI flags, skipping Harbor-specific fields."""
        skip = _MILES_SKIP_FIELDS | _HARBOR_SKIP_FIELDS
        out: list[str] = []
        for key, val in vars(self).items():
            if key.startswith("_") or key in skip or val is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(val, bool):
                if val:
                    out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out


class HarborConfig:
    """Top-level config composing dataset, model, wandb, and Harbor framework settings."""

    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: HarborFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: HarborFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or HarborFrameworkConfig()

    def build_app(self, *, name: str | None = None) -> "App":
        from .launcher import build_harbor_app

        return build_harbor_app(harbor=self, name=name)
