"""Post-training handle: look up checkpoints, serve them, and write evals.

``build_app()`` attaches a :class:`TrainResult` to the returned
``modal.App`` as ``app.train_result``. The struct is pure data — it
does not talk to Modal until you call a query method — so it is safe
to import at module scope and hand around to evaluation scripts.

Typical flow:

.. code-block:: python

    app = my_training_run.build_app()
    result = app.train_result

    # After `modal run --detach tutorial.py::app.train` finishes:
    print(result.latest_checkpoint_path())          # /checkpoints/iter_0000050
    print(result.dashboard_url())                   # modal.com volume dashboard

    serve_app = result.build_serve_app()            # Modal app hosting the
                                                    # latest checkpoint via
                                                    # vLLM (OpenAI-compatible).

The same :class:`TrainResult` value can also be re-created from a
bare ``(app_name, framework, ...)`` tuple in an eval-only script
that does not need to re-build the training config, which is the
point: eval code should not depend on the trainer's launcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal import App, Volume


@dataclass
class TrainResult:
    """Handle to a training run's checkpoints volume.

    Constructed by each framework's ``build_*_app()`` and attached to
    the returned ``modal.App`` as ``app.train_result``. All fields
    describe *where* a completed run's artifacts live on Modal — none
    of them require the training function to have executed yet, so the
    handle is safe to hold in a deploy-time script.

    Fields
    ------
    app_name : str
        Modal app name — also the prefix for the per-app
        checkpoints volume.
    framework : str
        Training framework identifier (``"slime"``, ``"ms-swift"``,
        ``"miles"``, ``"harbor"``). Evaluation scripts can switch on
        this to apply framework-specific decoding of the checkpoint
        directory layout.
    checkpoints_volume_name : str
        Name of the Modal Volume holding checkpoints.
    checkpoints_mount_path : str
        Where the volume is mounted inside the training container
        (e.g. ``"/checkpoints"``). Used as the default serving mount
        path so absolute paths stored in the volume resolve
        unchanged.
    base_model : str
        HuggingFace repo id of the base model — useful as a fallback
        ``served_model_name`` and for tokenizer-only evals.
    checkpoint_subpath : str
        Framework-specific subdirectory within the volume where
        per-iteration checkpoints land. Empty string means the volume
        root. SLIME's ``--save`` path, ms-swift's
        ``{app_name}_{run_id}`` directory, and Miles/Harbor's
        ``{run_id}`` directory are all expressed here.
    iteration_prefix : str
        Prefix of per-iteration checkpoint directory names (e.g.
        ``"iter_"``). ``latest_checkpoint_path()`` sorts directories
        matching this prefix lexicographically and returns the last
        one. Empty string disables iteration lookup — the checkpoint
        path is the ``checkpoint_subpath`` directory itself.
    volume_version : int | None
        ``modal.Volume`` filesystem version. ``None`` means use the
        default. ms-swift uses ``version=2``; other frameworks don't
        set it.
    extra : dict[str, Any]
        Free-form metadata a framework launcher may attach (e.g. the
        specific ``run_id`` rank 0 chose, the wandb group). Not used
        by the base class — purely a scratchpad for launcher-specific
        bookkeeping.
    """

    app_name: str
    framework: str
    checkpoints_volume_name: str
    checkpoints_mount_path: str
    base_model: str
    checkpoint_subpath: str = ""
    iteration_prefix: str = ""
    volume_version: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Volume lookup ────────────────────────────────────────────────────

    def volume(self) -> "Volume":
        """Return a handle to the checkpoints :class:`modal.Volume`.

        The returned object is hydrated lazily — no network call fires
        until a method on it is invoked.
        """
        from modal import Volume

        kwargs: dict[str, Any] = {"create_if_missing": True}
        if self.volume_version is not None:
            kwargs["version"] = self.volume_version
        return Volume.from_name(self.checkpoints_volume_name, **kwargs)

    def dashboard_url(self) -> str:
        """URL for browsing the checkpoints volume in the Modal dashboard.

        Fully hydrates the volume and returns its dashboard URL.
        Prefer :meth:`list_checkpoints` or
        :meth:`latest_checkpoint_path` for programmatic access.
        """
        return self.volume().get_dashboard_url()

    # ── Checkpoint enumeration ──────────────────────────────────────────

    def _listdir(self, path: str) -> list[str]:
        """List one level of entries in the checkpoints volume.

        Returns directory *names* (not full paths), filtered to
        entries at the requested ``path`` only. Raises any underlying
        exception unchanged — callers must decide whether a missing
        directory is fatal.
        """
        vol = self.volume()
        prefix = path.rstrip("/") + "/" if path else ""
        prefix_depth = prefix.count("/")
        names: list[str] = []
        for entry in vol.iterdir(path or "/"):
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
        """Return per-iteration checkpoint directory names, sorted.

        Returns an empty list when ``iteration_prefix`` is unset,
        since there is no canonical "latest iteration" concept for
        frameworks that write a single flat directory.
        """
        if not self.iteration_prefix:
            return []
        try:
            entries = self._listdir(self.checkpoint_subpath)
        except Exception:
            return []
        return sorted(e for e in entries if e.startswith(self.iteration_prefix))

    def latest_checkpoint_path(self) -> str:
        """Return the absolute in-volume path of the latest checkpoint.

        For frameworks with an ``iteration_prefix``, this is the
        lexicographically-largest directory that starts with that
        prefix inside ``checkpoint_subpath``. For frameworks without
        per-iteration directories, this falls back to
        ``checkpoint_subpath`` itself (or the mount path when the
        subpath is also empty).

        Raises
        ------
        FileNotFoundError
            No checkpoint directories exist yet. Typically means the
            training function has not finished a save interval.
        """
        mount = self.checkpoints_mount_path.rstrip("/") or "/"
        if self.iteration_prefix:
            ckpts = self.list_checkpoints()
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints found under "
                    f"{self.checkpoints_volume_name}:"
                    f"/{self.checkpoint_subpath.strip('/')} with prefix "
                    f"{self.iteration_prefix!r}. Has the training run "
                    f"completed a save interval yet?"
                )
            parts = [
                p for p in (mount, self.checkpoint_subpath, ckpts[-1]) if p and p != "/"
            ]
            return "/" + "/".join(parts).lstrip("/")
        parts = [p for p in (mount, self.checkpoint_subpath) if p and p != "/"]
        return "/" + "/".join(parts).lstrip("/") if parts else mount

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
        :func:`modal_training_gym.common.serve_vllm.build_vllm_serve_app`
        that wires up this result's checkpoints volume and picks a
        sensible default ``model_path``.

        Parameters
        ----------
        served_model_name:
            ``--served-model-name`` passed to vLLM. Defaults to
            ``self.app_name``.
        checkpoint_path:
            Absolute in-container checkpoint directory. Defaults to
            :meth:`latest_checkpoint_path`, which queries Modal to find
            the newest iteration directory. Pass an explicit path to
            pin a specific iteration.
        **vllm_kwargs:
            Forwarded to ``build_vllm_serve_app`` (e.g. ``gpu``,
            ``n_gpu``, ``extra_vllm_args``).
        """
        from modal_training_gym.common.serve_vllm import build_vllm_serve_app

        model_path = checkpoint_path or self.latest_checkpoint_path()
        return build_vllm_serve_app(
            app_name=f"{self.app_name}-serve",
            model_path=model_path,
            served_model_name=served_model_name or self.app_name,
            checkpoints_volume=self.checkpoints_volume_name,
            checkpoints_mount_path=self.checkpoints_mount_path,
            **vllm_kwargs,
        )
