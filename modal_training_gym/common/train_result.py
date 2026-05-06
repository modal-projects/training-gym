"""Post-training handle: look up checkpoints, serve them, write evals.

A :class:`TrainResult` is produced by ``train`` itself — one per call —
and is also persisted to a shared :class:`modal.Dict` keyed by
``training_run_id`` so that a *separate* evaluation script can look it up after
the fact without re-running training.

Typical flow:

.. code-block:: python

    # training.py
    app = TrainConfig(...).build_app()

    # Kick off training:
    #   modal run --detach training.py::app.train
    # The train function constructs a TrainResult, writes it to the
    # shared modal.Dict, and returns it.

    # eval.py (anywhere, any time after `train` finishes):
    from modal_training_gym.common.train_result import TrainResult

    result = TrainResult.load("my-app")                 # latest run
    # or
    result = TrainResult.load("my-app", training_run_id="...")   # pinned

    print(result.checkpoint_dir)            # /checkpoints/my-app_train_...
    print(result.latest_checkpoint_path())  # .../iter_0000050

    from modal_training_gym.common.deployment import DeploymentConfig
    deployment = DeploymentConfig(model=result.model).serve()
    print(deployment.url)

Two design invariants:

1. ``TrainResult`` is **not** attached to the ``modal.App`` returned by
   ``build_app()`` — a training run has no "result" before ``train`` has
   executed. The ``modal.App`` object is a deployment handle, not a run
   handle.
2. Every call to ``train`` can produce a *different* result (different
   ``training_run_id``, different ``checkpoint_dir``). We use a :class:`modal.Dict`
   named ``{app_name}-train-results`` as the shared store so that eval
   scripts — which don't import the training closure and don't call
   ``train()`` — can still retrieve results by ``training_run_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from modal_training_gym.frameworks.base import Framework
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

if TYPE_CHECKING:
    from modal import Volume
    from modal_training_gym.common.models import ModelConfig


TRAIN_RESULTS_STORE_NAME = MetadataStore.TRAIN_RESULTS.value


@dataclass
class TrainResult:
    """One completed training run's checkpoint handle.

    Constructed and persisted by each framework's ``train`` function at
    the end of a run, and loaded by eval scripts via :meth:`load`. All
    fields are pure data — no method connects to Modal until explicitly
    invoked.

    Fields
    ------
    app_name:
        Modal app name — also the prefix for the per-app checkpoints
        volume (``{app_name}-checkpoints``) and the shared results
        :class:`modal.Dict` (``{app_name}-train-results``).
    framework:
        Training framework identifier.
    training_run_id:
        Unique identifier for *this* specific training call. Keys the
        record in the shared :class:`modal.Dict`; embedded in
        ``checkpoint_dir`` for frameworks that scope checkpoints by run.
    checkpoint_dir:
        Absolute in-container path to this run's checkpoint directory.
        For slime (which
        writes a single flat ``iter_*`` tree at ``slime.save``) this is
        that save root.
    model_config:
        The ``ModelConfig`` used for training. The :attr:`model` property
        returns a copy with ``model_path`` pointing at the latest
        checkpoint, ready to serve.
    checkpoints_volume_name:
        Name of the checkpoints :class:`modal.Volume`.
    checkpoints_mount_path:
        Where the checkpoints volume is mounted in the training
        container (e.g. ``"/checkpoints"``). Reused as the serving
        mount path so in-volume absolute paths resolve unchanged.
    iteration_prefix:
        Prefix of per-iteration subdirectory names inside
        ``checkpoint_dir`` (e.g. ``"iter_"``).
        :meth:`latest_checkpoint_path` sorts directories matching this
        prefix lexicographically and returns the last one. Empty
        string disables iteration lookup — the checkpoint path is
        ``checkpoint_dir`` itself.
    extra:
        Free-form metadata a launcher may attach (e.g. wandb run name,
        rollout tunnel URL). Not used by the base class.
    """

    app_name: str
    framework: Framework
    training_run_id: str
    checkpoint_dir: str
    model_config: "ModelConfig | None" = None
    checkpoints_volume_name: str = ""
    checkpoints_mount_path: str = ""
    iteration_prefix: str = ""
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

    def save(self) -> None:
        """Persist this result to the shared metadata volume."""
        vol_put(MetadataStore.TRAIN_RESULTS, self.training_run_id, self._to_dict())

    @staticmethod
    def _parse_model_config(d: dict[str, Any]) -> dict[str, Any]:
        mc = d.get("model_config")
        if isinstance(mc, dict):
            from modal_training_gym.common.models import ModelConfig

            d["model_config"] = ModelConfig(**mc)
        return d

    @classmethod
    def load(cls, training_run_id: str) -> "TrainResult":
        """Load a completed run's result.

        Raises
        ------
        KeyError
            The given ``training_run_id`` isn't in the store.
        """
        return cls(
            **cls._parse_model_config(
                vol_get(MetadataStore.TRAIN_RESULTS, training_run_id)
            )
        )

    @classmethod
    def list_results(cls) -> list["TrainResult"]:
        return [
            cls(**cls._parse_model_config(r))
            for r in vol_list(MetadataStore.TRAIN_RESULTS)
        ]

    # ── Volume lookup ────────────────────────────────────────────────────

    def volume(self) -> "Volume":
        """Return a handle to the checkpoints :class:`modal.Volume`."""
        from modal import Volume

        return Volume.from_name(
            self.checkpoints_volume_name,
            create_if_missing=True,
        )

    def dashboard_url(self) -> str:
        """URL for browsing the checkpoints volume in the Modal dashboard."""
        return self.volume().get_dashboard_url()

    # ── Checkpoint enumeration ──────────────────────────────────────────

    def _volume_rel(self, absolute: str) -> str:
        """Convert an absolute path under ``checkpoints_mount_path`` to a
        volume-relative path suitable for :meth:`modal.Volume.iterdir`.
        """
        mount = self.checkpoints_mount_path.rstrip("/")
        if not mount:
            return absolute.lstrip("/")
        if absolute == mount:
            return ""
        if absolute.startswith(mount + "/"):
            return absolute[len(mount) + 1 :]
        return absolute.lstrip("/")

    def _listdir(self, rel_path: str) -> list[str]:
        """List one level of entries under a volume-relative path.

        Returns basenames only. Handles both flat listings (entries
        yield a basename as ``path``) and recursive listings (entries
        yield ``dir/file``) by filtering to the requested depth.
        """
        vol = self.volume()
        rel = rel_path.strip("/")
        prefix = rel + "/" if rel else ""
        prefix_depth = prefix.count("/")
        names: list[str] = []
        for entry in vol.iterdir(rel or "/"):
            entry_path = entry.path
            if prefix and not entry_path.startswith(prefix):
                continue
            remainder = entry_path[len(prefix) :] if prefix else entry_path
            if "/" in remainder:
                continue
            if prefix and entry_path.count("/") != prefix_depth:
                continue
            names.append(remainder)
        return names

    def list_checkpoints(self) -> list[str]:
        """Return per-iteration checkpoint directory names under
        ``checkpoint_dir``, sorted.

        Empty list when ``iteration_prefix`` is unset (frameworks that
        write a single flat output directory per run).
        """
        if not self.iteration_prefix:
            return []
        rel = self._volume_rel(self.checkpoint_dir)
        try:
            entries = self._listdir(rel)
        except FileNotFoundError:
            return []
        return sorted(
            e
            for e in entries
            if e.startswith(self.iteration_prefix) and not e.endswith("_hf")
        )

    def latest_checkpoint_path(self) -> str:
        """Absolute in-volume path of the latest checkpoint.

        For frameworks with an ``iteration_prefix``, the largest
        directory (lexicographic sort) starting with that prefix inside
        :attr:`checkpoint_dir`. For frameworks without iteration
        directories, :attr:`checkpoint_dir` itself.

        Raises
        ------
        FileNotFoundError
            No matching iteration directory exists yet. Typically means
            the run has not completed a save interval.
        """
        if not self.iteration_prefix:
            return self.checkpoint_dir
        ckpts = self.list_checkpoints()
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints under {self.checkpoint_dir} "
                f"with prefix {self.iteration_prefix!r}. "
                f"Has the training run completed a save interval yet?"
            )
        latest = ckpts[-1]
        hf_variant = f"{latest}_hf"
        if hf_variant in self._listdir(self._volume_rel(self.checkpoint_dir)):
            latest = hf_variant
        return f"{self.checkpoint_dir.rstrip('/')}/{latest}"

    # ── Output model ─────────────────────────────────────────────────────

    @property
    def model(self) -> "ModelConfig":
        """Return the model config with ``model_path`` pointing at the latest checkpoint.

        Returns a shallow copy of ``model_config`` so the stored
        instance is not mutated. The copy has ``model_path``,
        ``checkpoints_volume_name``, and ``checkpoints_mount_path``
        set so that ``DeploymentConfig(model=result.model).serve()``
        works out of the box.
        """
        import copy

        if self.model_config is None:
            raise ValueError(
                "No model_config on this TrainResult. "
                "Was it saved by an older launcher?"
            )
        m = copy.copy(self.model_config)
        m.model_path = self.latest_checkpoint_path()
        m.checkpoints_volume_name = self.checkpoints_volume_name
        m.checkpoints_mount_path = self.checkpoints_mount_path
        return m
