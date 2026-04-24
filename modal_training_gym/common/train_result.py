"""Post-training handle: look up checkpoints, serve them, write evals.

A :class:`TrainResult` is produced by ``train`` itself — one per call —
and is also persisted to a shared :class:`modal.Dict` keyed by
``run_id`` so that a *separate* evaluation script can look it up after
the fact without re-running training.

Typical flow:

.. code-block:: python

    # training.py
    app = MsSwiftConfig(...).build_app()

    # Kick off training:
    #   modal run --detach training.py::app.train
    # The train function constructs a TrainResult, writes it to the
    # shared modal.Dict, and returns it.

    # eval.py (anywhere, any time after `train` finishes):
    from modal_training_gym.common.train_result import TrainResult

    result = TrainResult.load("my-app")                 # latest run
    # or
    result = TrainResult.load("my-app", run_id="...")   # pinned

    print(result.checkpoint_dir)            # /checkpoints/my-app_train_...
    print(result.latest_checkpoint_path())  # .../iter_0000050

    serve_app = result.build_serve_app()    # vLLM app hosting the ckpt

Two design invariants:

1. ``TrainResult`` is **not** attached to the ``modal.App`` returned by
   ``build_app()`` — a training run has no "result" before ``train`` has
   executed. The ``modal.App`` object is a deployment handle, not a run
   handle.
2. Every call to ``train`` can produce a *different* result (different
   ``run_id``, different ``checkpoint_dir``). We use a :class:`modal.Dict`
   named ``{app_name}-train-results`` as the shared store so that eval
   scripts — which don't import the training closure and don't call
   ``train()`` — can still retrieve results by ``run_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal import App, Volume
    from modal_training_gym.common.models import ModelConfiguration


def _store_name(app_name: str) -> str:
    """Canonical name for the shared :class:`modal.Dict` store."""
    return f"{app_name}-train-results"


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
        Training framework identifier (``"slime"``, ``"ms-swift"``,
        ``"miles"``, ``"harbor"``). Eval scripts can switch on this to
        apply framework-specific decoding of the checkpoint layout.
    run_id:
        Unique identifier for *this* specific training call. Keys the
        record in the shared :class:`modal.Dict`; embedded in
        ``checkpoint_dir`` for frameworks that scope checkpoints by run.
    checkpoint_dir:
        Absolute in-container path to this run's checkpoint directory.
        For ms-swift / miles / harbor this is a run-specific
        subdirectory under the checkpoints mount; for slime (which
        writes a single flat ``iter_*`` tree at ``slime.save``) this is
        that save root.
    base_model:
        HuggingFace repo id of the base model — useful as a fallback
        ``served_model_name`` and for tokenizer-only evals.
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
    framework: str
    run_id: str
    checkpoint_dir: str
    base_model: str
    model_class: str = ""
    checkpoints_volume_name: str = ""
    checkpoints_mount_path: str = ""
    iteration_prefix: str = ""
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Persistence: shared modal.Dict ────────────────────────────────────

    def save(self) -> None:
        """Persist this result to the shared :class:`modal.Dict`.

        Called by each framework's ``train`` function at the end of a
        successful run (from rank 0 only — the others would race).
        Round-trips through :func:`dataclasses.asdict` so the record is
        trivially serializable and forward-compatible across launcher
        upgrades.
        """
        from modal import Dict

        store = Dict.from_name(_store_name(self.app_name), create_if_missing=True)
        store[self.run_id] = asdict(self)

    @classmethod
    def load(cls, app_name: str, run_id: str | None = None) -> "TrainResult":
        """Load a completed run's result from the shared store.

        Parameters
        ----------
        app_name:
            Same ``app_name`` the training run used (i.e. the
            ``modal.App`` name of the training app).
        run_id:
            Specific run to load. When ``None`` (default), returns the
            lexicographically-largest ``run_id`` — since all launchers
            embed a monotonic timestamp, that is the most recent run.

        Raises
        ------
        LookupError
            No runs have been saved for this app yet, or the given
            ``run_id`` isn't in the store.
        """
        from modal import Dict

        store = Dict.from_name(_store_name(app_name), create_if_missing=True)
        if run_id is None:
            keys = sorted(store.keys())
            if not keys:
                raise LookupError(
                    f"No training results saved for app {app_name!r}. "
                    f"Has `train` completed yet? "
                    f"(Results are written to the modal.Dict "
                    f"{_store_name(app_name)!r}.)"
                )
            run_id = keys[-1]
        if run_id not in store:
            raise LookupError(
                f"run_id {run_id!r} not found in results for app {app_name!r}. "
                f"Known runs: {sorted(store.keys())}"
            )
        return cls(**store[run_id])

    @classmethod
    def list_runs(cls, app_name: str) -> list[str]:
        """Return all ``run_id``s saved for ``app_name``, sorted oldest
        first (lexicographic ordering matches timestamp ordering)."""
        from modal import Dict

        store = Dict.from_name(_store_name(app_name), create_if_missing=True)
        return sorted(store.keys())

    # ── Volume lookup ────────────────────────────────────────────────────

    def volume(self) -> "Volume":
        """Return a handle to the checkpoints :class:`modal.Volume`.

        Always uses ``version=2`` — every framework launcher in this
        package provisions its checkpoints volume at v2.
        """
        from modal import Volume

        return Volume.from_name(
            self.checkpoints_volume_name,
            create_if_missing=True,
            version=2,
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
        except Exception:
            return []
        return sorted(e for e in entries if e.startswith(self.iteration_prefix))

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
        return f"{self.checkpoint_dir.rstrip('/')}/{ckpts[-1]}"

    # ── Output model ─────────────────────────────────────────────────────

    @property
    def model(self) -> "ModelConfiguration":
        """Return the original model class with ``model_path`` pointing at the checkpoint.

        Reconstructs the same model class used for training (e.g.
        ``Qwen3_4B``) so it retains architecture, presets, and
        ``download_model()`` logic. Only ``model_path`` is overridden
        to point at the trained checkpoint instead of the HF cache.
        """
        import importlib

        if self.model_class:
            module_path, class_name = self.model_class.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            instance.model_path = self.latest_checkpoint_path()
            return instance

        from modal_training_gym.common.models import ModelConfiguration

        return ModelConfiguration(
            model_name=self.base_model,
            model_path=self.latest_checkpoint_path(),
        )

    # ── Serving ──────────────────────────────────────────────────────────

    def build_serve_app(
        self,
        *,
        served_model_name: str | None = None,
        checkpoint_path: str | None = None,
        **vllm_kwargs: Any,
    ) -> "App":
        """Build a vLLM serving app pointing at a trained checkpoint.

        Thin wrapper over
        :func:`modal_training_gym.common.serve_vllm.build_vllm_serve_app`.
        The returned app's name embeds ``run_id`` so multiple trained
        checkpoints can be served concurrently without name collision.

        Parameters
        ----------
        served_model_name:
            ``--served-model-name`` passed to vLLM. Defaults to
            ``f"{app_name}-{run_id}"``.
        checkpoint_path:
            Absolute in-container checkpoint directory. Defaults to
            :meth:`latest_checkpoint_path`.
        **vllm_kwargs:
            Forwarded to ``build_vllm_serve_app``.
        """
        from modal_training_gym.common.serve_vllm import build_vllm_serve_app

        model_path = checkpoint_path or self.latest_checkpoint_path()
        return build_vllm_serve_app(
            app_name=f"{self.app_name}-{self.run_id}-serve",
            model_path=model_path,
            served_model_name=served_model_name or f"{self.app_name}-{self.run_id}",
            checkpoints_volume=self.checkpoints_volume_name,
            checkpoints_mount_path=self.checkpoints_mount_path,
            **vllm_kwargs,
        )

    # ── W&B integration ─────────────────────────────────────────────────

    def wandb_url(self) -> str | None:
        """Return the W&B run URL, or None if W&B info is not set."""
        if not self.wandb_run_id or not self.wandb_project:
            return None
        entity = self.wandb_entity or "_"
        return f"https://wandb.ai/{entity}/{self.wandb_project}/runs/{self.wandb_run_id}"

    def wandb_metrics(
        self,
        keys: list[str] | None = None,
        samples: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch training metrics from W&B.

        Parameters
        ----------
        keys:
            Metric keys to fetch (e.g. ``["train/loss", "train/reward"]``).
            When ``None``, fetches all logged metrics.
        samples:
            Max number of data points. Use ``0`` for full unsampled history.

        Returns
        -------
        List of dicts, one per logged step. Each dict has ``_step`` plus
        the requested metric keys.
        """
        import wandb

        api = wandb.Api()
        entity = self.wandb_entity or None
        path = f"{entity}/{self.wandb_project}/{self.wandb_run_id}" if entity else f"{self.wandb_project}/{self.wandb_run_id}"
        run = api.run(path)

        if samples == 0:
            rows = list(run.scan_history(keys=keys))
        else:
            df = run.history(keys=keys, samples=samples)
            rows = df.to_dict("records")
        return rows

    def wandb_summary(self) -> dict[str, Any]:
        """Fetch the W&B run summary (final metric values)."""
        import wandb

        api = wandb.Api()
        entity = self.wandb_entity or None
        path = f"{entity}/{self.wandb_project}/{self.wandb_run_id}" if entity else f"{self.wandb_project}/{self.wandb_run_id}"
        run = api.run(path)
        return dict(run.summary)
