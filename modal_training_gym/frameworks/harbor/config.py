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
    CHECKPOINTS_PATH as CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH as HF_CACHE_PATH,
    MilesFrameworkConfig,
    model_training_overrides,
)
from modal_training_gym.frameworks.miles.config import (
    _SKIP_FIELDS as _MILES_SKIP_FIELDS,
)

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration, ModelTrainingConfig
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
    """Harbor + Miles configuration for sandbox-based RL training.

    Inherits all Miles training settings and adds Harbor
    agent/environment/task configuration. The launcher wires up the
    agent function, reward function, and rollout function automatically.

    ## Harbor Agent

    agent_import_path : str
        Python import path for the Harbor agent class. Default ``""``.
    agent_model_name : str
        Model name passed to the agent. Default ``"model"``.
    agent_kwargs : dict[str, Any]
        Extra keyword arguments for agent construction. Default ``{}``.

    ## Harbor Environment

    environment_import_path : str | None
        Import path for custom Harbor environment. Default ``None``.
    sandbox_timeout_secs : int
        Maximum sandbox execution time in seconds. Default ``1800`` (30 min).
    sandbox_idle_timeout_secs : int
        Sandbox idle timeout in seconds. Default ``300`` (5 min).

    ## Harbor Tasks

    task_root : str
        Root directory for Harbor task directories on the data volume.
        Default ``"/data/tasks"``.
    task_glob : str
        Glob pattern for discovering task directories. Default ``"*"``.
    instruction_path : str
        Relative path to the instruction file within each task dir.
        Default ``"instruction.md"``.

    ## Harbor RL Defaults

    input_key : str
        Dataset column key for model input prompts. Default ``"prompt"``.
    label_key : str
        Dataset column key for metadata/labels. Default ``"metadata"``.
    apply_chat_template : bool
        Apply the model's chat template to inputs. Default ``True``.
    enable_thinking : bool
        Enable thinking/reasoning mode in the model. Default ``True``.
    rollout_shuffle : bool
        Shuffle data during rollout generation. Default ``True``.

    ## Memory (Harbor Overrides)

    recompute_granularity : str
        Activation recomputation granularity. Default ``"selective"``.

    ## Model Overrides

    max_position_embeddings : int
        Maximum sequence position embeddings. Default ``32768``.
    untie_embeddings_and_output_weights : bool
        Untie input embeddings from output projection. Default ``True``.
    no_masked_softmax_fusion : bool
        Disable masked softmax fusion. Default ``True``.

    ## Rollout (Harbor)

    num_rollout : int
        Number of rollout episodes per training step. Default ``200``.
    rollout_batch_size : int
        Batch size for rollout generation. Default ``64``.
    rollout_max_response_len : int
        Maximum response length during rollout. Default ``1024``.
    sglang_mem_fraction_static : float
        SGLang static memory fraction. Default ``0.7``.

    ## Training (Harbor Overrides)

    train_iters : int
        Total training iterations. Default ``50``.
    global_batch_size : int
        Global batch size across all ranks. Default ``512``.

    ## Eval and Checkpointing (Harbor)

    eval_interval : int
        Evaluation interval in iterations. Default ``10``.
    save_interval : int
        Checkpoint save interval in iterations. Default ``10``.

    ## Image

    miles_image : str
        Docker image with Miles trainer. Default ``"radixark/miles:dev-202604201238"``.
    miles_src_commit : str
        Miles source commit with ``--custom-agent-function-path`` support.
    harbor_install_command : str
        Command to install Harbor in the image.
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

    # ── Harbor RL defaults ─────────────────────────────────────────────────
    input_key: str = "prompt"
    label_key: str = "metadata"
    apply_chat_template: bool = True
    enable_thinking: bool = True
    rollout_shuffle: bool = True

    # ── Memory ──────────────────────────────────────────────────────────────
    recompute_granularity: str = "selective"

    # ── Model overrides ─────────────────────────────────────────────────────
    max_position_embeddings: int = 32768
    untie_embeddings_and_output_weights: bool = True
    no_masked_softmax_fusion: bool = True

    # ── Rollout ─────────────────────────────────────────────────────────────
    num_rollout: int = 200
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 1024
    sglang_mem_fraction_static: float = 0.7

    # ── Training ────────────────────────────────────────────────────────────
    train_iters: int = 50
    global_batch_size: int = 512

    # ── Eval & Checkpointing ────────────────────────────────────────────────
    eval_interval: int = 10
    save_interval: int = 10

    # ── Image ────────────────────────────────────────────────────────────────
    miles_image: str = "radixark/miles:dev-202604201238"
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
    """Top-level wrapper that composes a Harbor + Miles RLVR training run.

    Bundles a model, dataset, W&B logging config, and Harbor-specific
    framework settings. Call ``build_app()`` to get a Modal ``App``
    ready to run.

    ## Fields

    dataset : DatasetConfig | None
        Dataset configuration with ``prepare()`` hook. Default ``None``.
    model : ModelConfiguration | None
        Model identity and download hook. Default ``None``.
    wandb : WandbConfig | None
        Weights & Biases logging config. Default ``None``.
    framework_config : HarborFrameworkConfig
        Harbor + Miles training and infrastructure settings.
        Default ``HarborFrameworkConfig()``.

    Methods
    -------
    build_app(name=None)
        Build and return a Modal ``App`` for this training run.
    """

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
