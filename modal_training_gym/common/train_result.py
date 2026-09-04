"""Post-training handle: look up checkpoints, serve them, write evals.

A :class:`TrainResult` is produced by ``train`` itself — one per call — and is
also persisted to the metadata volume keyed by ``training_run_id`` so that a
*separate* evaluation script can look it up after the fact without re-running
training.

Typical flow:

.. code-block:: python

    # training.py
    result = TrainConfig(...).train()          # blocks, returns the TrainResult
    # or, to launch and walk away:
    run = TrainConfig(...).launch()            # detached; returns a TrainingRun
    result = run.result()                      # optional wait

    # eval.py (anywhere, any time after `train` finishes):
    from modal_training_gym.common.train_result import TrainResult

    result = TrainResult.load(training_run_id)

    print(result.checkpoint_dir)                 # /checkpoints/my-app_train_...
    print(result.latest_checkpoint().path)       # .../iter_0000050

    from modal_training_gym import CustomDeployment
    from modal_training_gym.deploy_recipes import SglangRecipe
    deployment = CustomDeployment.launch(result.model, recipe=SglangRecipe())
    print(deployment.url)

Two design invariants:

1. ``TrainResult`` is **not** attached to the ``modal.App`` that
   ``TrainConfig`` builds internally — a training run has no "result" before
   ``train`` has executed. The ``modal.App`` object is a deployment handle,
   not a run handle; :class:`TrainingRun` is the run handle.
2. Every call to ``train`` can produce a *different* result (different
   ``training_run_id``, different ``checkpoint_dir``). Results live in the
   shared metadata volume so that eval scripts — which don't import the
   training closure and don't call ``train()`` — can still retrieve them by
   ``training_run_id``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import asdict, dataclass, field, fields
import copy
from typing import TYPE_CHECKING, Any

from modal import Volume

from modal_training_gym.common.framework import Framework
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_put_with_summary,
)

if TYPE_CHECKING:
    from modal_training_gym.common.checkpoint import Checkpoint
    from modal_training_gym.common.models import ModelConfig


TRAIN_RESULTS_STORE_NAME = MetadataStore.TRAIN_RESULTS.value


@dataclass
class TrainResult:
    """Result metadata and checkpoints for a completed training run.

    Args:
        app_name:
            Modal app name and default checkpoint volume prefix.
        framework:
            Training framework.
        training_run_id:
            Run ID and key in the shared results store.
        checkpoint_dir:
            Checkpoint directory inside the training container.
        model_config:
            Model used for training.
        group_id:
            ``TrainingGroup`` sweep ID.
        extra:
            Framework-specific run metadata.
    """

    app_name: str
    framework: Framework
    training_run_id: str
    checkpoint_dir: str = ""
    checkpoints_volume_name: str = ""
    checkpoints_mount_path: str = ""
    model_config: "ModelConfig | None" = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Set when this run was launched as part of a TrainingGroup sweep; shared
    # across every variant in the group so results can be compared together.
    group_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Persistence ────────────────────────────────────────────────────────

    def _to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.model_config is not None:
            d["model_config"] = {
                "model_name": self.model_config.model_name,
                "model_path": self.model_config.model_path,
            }
        return d

    @staticmethod
    def _summary_sort_key(item: dict[str, Any]) -> str:
        return str(item.get("training_run_id", ""))

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        return vol_put_with_summary(
            MetadataStore.TRAIN_RESULTS,
            self.training_run_id,
            self._to_dict(),
            summary_store=MetadataStore.TRAIN_RESULTS_SUMMARY,
            item_id_key="training_run_id",
            sort_key=self._summary_sort_key,
            reverse=True,
            is_async=is_async,
        )

    @staticmethod
    def _parse_model_config(d: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(d)
        mc = parsed.get("model_config")
        if isinstance(mc, dict):
            from modal_training_gym.common.models import ModelConfig

            parsed["model_config"] = ModelConfig(**mc)
        valid_fields = {f.name for f in fields(TrainResult)}
        unknown = {k: v for k, v in parsed.items() if k not in valid_fields}
        extra = parsed.get("extra")
        if not isinstance(extra, dict):
            extra = {}
        if unknown:
            parsed["extra"] = {**unknown, **extra}
        return {k: v for k, v in parsed.items() if k in valid_fields}

    @classmethod
    def from_training_run_id(cls, training_run_id: str) -> "TrainResult":
        """Load a ``TrainResult`` by ``training_run_id``.

        Returns:
            The persisted ``TrainResult``.

        Raises:
            KeyError: The given `training_run_id` isn't in the store.
        """
        return cls(
            **cls._parse_model_config(
                vol_get(MetadataStore.TRAIN_RESULTS, training_run_id)
            )
        )

    @classmethod
    def load(cls, training_run_id: str) -> "TrainResult":
        """Alias for ``from_training_run_id``."""
        return cls.from_training_run_id(training_run_id)

    def volume(self) -> "Volume":
        """Open the checkpoints volume.

        Returns:
            The Modal checkpoints ``Volume``.
        """
        volume_name = self.checkpoints_volume_name or f"{self.app_name}-checkpoints"
        return Volume.from_name(volume_name, create_if_missing=False)

    def checkpoints(self) -> list["Checkpoint"]:
        """List this run's checkpoints.

        Returns:
            This run's ``Checkpoint`` objects.
        """
        from modal_training_gym.common.checkpoint import _list_checkpoints
        from modal_training_gym.common.errors import TrainingGymConfigError

        if self.framework in {
            Framework.SLIME,
            Framework.SLIME.value,
            Framework.MILES,
            Framework.MILES.value,
        }:
            return _list_checkpoints(
                self.checkpoint_dir,
                self.checkpoints_volume_name or f"{self.app_name}-checkpoints",
                self.checkpoints_mount_path or "/checkpoints",
                include_hf=True,
                fallback_without_tracker=True,
                training_run_id=self.training_run_id,
                app_name=self.app_name,
            )
        raise TrainingGymConfigError(f"Unsupported framework: {self.framework}")

    def latest_checkpoint(self) -> "Checkpoint | None":
        """Return the last megatron ``iter_*`` checkpoint, if any."""
        from modal_training_gym.common.checkpoint import CheckpointType

        megatron = [
            checkpoint
            for checkpoint in self.checkpoints()
            if checkpoint.checkpoint_type is CheckpointType.megatron
        ]
        return megatron[-1] if megatron else None

    @property
    def model(self) -> "ModelConfig":
        """Build a ``ModelConfig`` for this training result.

        Returns:
            A copy whose ``model_path`` targets the latest checkpoint, or
            ``checkpoint_dir`` when no checkpoints are recorded.
        """
        if self.model_config is None:
            raise ValueError(
                "No model_config on this TrainResult. "
                "Was it saved by an older launcher?"
            )

        model = copy.copy(self.model_config)
        checkpoint = self.latest_checkpoint()
        if checkpoint is not None:
            model.model_path = checkpoint.path
        elif self.checkpoint_dir:
            model.model_path = self.checkpoint_dir

        if self.checkpoints_volume_name:
            setattr(model, "checkpoints_volume_name", self.checkpoints_volume_name)
        if self.checkpoints_mount_path:
            setattr(model, "checkpoints_mount_path", self.checkpoints_mount_path)
        return model
