"""Mirror W&B-style scalar metrics into the Training Gym dashboard.

Two modes, selected by ``TRAINING_GYM_METRIC_MIRROR`` in the training
container and bootstrapped from a ``.pth`` file in site-packages (the same
trick ``trackio.py`` uses):

``tee``
    The framework keeps logging to the real ``wandb`` (or the Trackio adapter).
    An import hook patches ``wandb.sdk.wandb_run.Run.log`` once the package
    loads, so every ``wandb.log`` / ``run.log`` call also reaches the dashboard.
``sink``
    ``DashboardMetricConfig``: the dashboard is the only tracker. A synthetic
    ``wandb`` module covering the subset Slime and Miles use is installed, so
    no W&B account or Trackio server is needed.

Points are coalesced per step and shipped on the shared reporting queue
(``reporting.py``), so they inherit the per-run bearer token, the
never-block-training semantics and the process-exit drain.

The ``.pth`` imports this module in every interpreter of the training image
(Ray workers, SGLang servers, ...), so module import must stay cheap: the
reporting queue is only imported once the first metric is logged.
"""

from __future__ import annotations

import math
import os
import shlex
import socket
import sys
import threading
import time
import types
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Any, Callable, ClassVar

from modal_training_gym.common.metrics import MetricConfig

MIRROR_MODE_ENV = "TRAINING_GYM_METRIC_MIRROR"
MIRROR_TEE = "tee"
MIRROR_SINK = "sink"
TRAINING_RUN_ID_ENV = "TRAINING_GYM_TRAINING_RUN_ID"

MAX_STEP = 100_000_000
MAX_METRIC_KEY_LENGTH = 256
MAX_METRIC_KEYS_PER_POINT = 2000
FLUSH_INTERVAL_SECONDS = 2.0
MAX_PENDING_STEPS = 200

_SHIM_MARKER = "_training_gym_metric_mirror"
_TEE_MARKER = "_training_gym_metric_tee"
_PTH_FILE = "_training_gym_metric_mirror.pth"
_PTH_LINE = (
    "import os; not os.environ.get('TRAINING_GYM_METRIC_MIRROR') "
    "or __import__('modal_training_gym.common.metric_mirror', "
    "fromlist=['bootstrap']).bootstrap()\n"
)


# ── Scalar extraction ─────


def as_finite_float(value: Any) -> float | None:
    """``value`` as a finite float, or None for anything W&B would not chart.

    Accepts ints, floats and 0-d tensors/arrays (via ``item()``); rejects
    bools, strings, NaN/inf and media objects.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            return None
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return None
        try:
            scalar = item()
            if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
                return None
            number = float(scalar)
        except (TypeError, ValueError, RuntimeError):
            return None
    return number if math.isfinite(number) else None


def flatten_numeric(
    data: Mapping[str, Any], *, prefix: str = "", sep: str = "/"
) -> dict[str, float]:
    """Flatten ``{"train": {"loss": 0.4}}`` to ``{"train/loss": 0.4}``."""
    out: dict[str, float] = {}
    for raw_key, value in data.items():
        key = f"{prefix}{sep}{raw_key}" if prefix else str(raw_key)
        if isinstance(value, Mapping):
            out.update(flatten_numeric(value, prefix=key, sep=sep))
        else:
            number = as_finite_float(value)
            if number is None or not key or len(key) > MAX_METRIC_KEY_LENGTH:
                continue
            out[key] = number
        if len(out) >= MAX_METRIC_KEYS_PER_POINT:
            break
    return out


# ── Client-side buffer ─────


class MetricMirror:
    """Coalesces ``log`` calls per step and posts them in batches.

    Step semantics follow ``wandb.log``: with ``step=None`` data lands on the
    implicit step, which advances unless ``commit=False``; an explicit step
    moves the implicit step forward but never back. Repeated keys within a
    step overwrite (the dashboard merges the same way).
    """

    def __init__(
        self,
        training_run_id: str,
        *,
        source: str = "",
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
        max_pending_steps: int = MAX_PENDING_STEPS,
        clock: Callable[[], float] = time.time,
        send: Callable[[dict[str, Any], bool], None] | None = None,
    ) -> None:
        self.training_run_id = training_run_id
        self.source = source or f"{socket.gethostname()}:{os.getpid()}"
        self._flush_interval = flush_interval
        self._max_pending_steps = max_pending_steps
        self._clock = clock
        self._send = send or _send_batch
        self._lock = threading.Lock()
        self._pending: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._last_flush = clock()
        self._timer: threading.Timer | None = None

    @property
    def step(self) -> int:
        return self._next_step

    def log(
        self,
        data: Mapping[str, Any] | None,
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> int | None:
        """Record ``data`` at ``step``; returns the step used (None if dropped)."""
        metrics = flatten_numeric(data) if isinstance(data, Mapping) else {}
        now = self._clock()
        with self._lock:
            if step is None:
                step = self._next_step
                if commit is not False:
                    self._next_step += 1
            else:
                try:
                    step = int(step)
                except (TypeError, ValueError):
                    return None
                if step < 0 or step > MAX_STEP:
                    return None
                if step > self._next_step:
                    self._next_step = step
            if metrics:
                record = self._pending.get(step)
                if record is None:
                    record = {"step": step, "ts": now, "metrics": {}}
                    self._pending[step] = record
                record["ts"] = now
                record["metrics"].update(metrics)
            due = bool(self._pending) and (
                len(self._pending) >= self._max_pending_steps
                or now - self._last_flush >= self._flush_interval
            )
            if not due and self._pending:
                self._arm_timer_locked()
        if due:
            self.flush()
        return step

    def _arm_timer_locked(self) -> None:
        if self._timer is not None:
            return
        timer = threading.Timer(self._flush_interval, self.flush)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def flush(self, *, final: bool = False) -> int:
        """Ship every pending step; returns how many were sent."""
        with self._lock:
            timer, self._timer = self._timer, None
            points = [self._pending[step] for step in sorted(self._pending)]
            self._pending.clear()
            self._last_flush = self._clock()
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        if not points and not final:
            return 0
        payload = {
            "training_run_id": self.training_run_id,
            "points": points,
            "source": self.source,
            "final": final,
        }
        try:
            self._send(payload, final)
        except Exception:
            return 0
        return len(points)


def _send_batch(payload: dict[str, Any], final: bool) -> None:
    from modal_training_gym.common.reporting import _enqueue_metric_points

    _enqueue_metric_points(payload, final=final)


_ACTIVE_LOCK = threading.Lock()
_ACTIVE: MetricMirror | None = None
_ACTIVE_DISABLED = False


def active_mirror() -> MetricMirror | None:
    """The process-wide mirror, or None when the container has no run context."""
    global _ACTIVE, _ACTIVE_DISABLED
    if _ACTIVE is not None or _ACTIVE_DISABLED:
        return _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is not None or _ACTIVE_DISABLED:
            return _ACTIVE
        from modal_training_gym.common.reporting import (
            _phase_url,
            register_pre_drain_hook,
        )

        training_run_id = os.environ.get(TRAINING_RUN_ID_ENV, "").strip()
        if not training_run_id or not _phase_url():
            _ACTIVE_DISABLED = True
            return None
        mirror = MetricMirror(training_run_id)

        def _final_flush() -> None:
            mirror.flush(final=True)

        register_pre_drain_hook(_final_flush)
        _ACTIVE = mirror
        return mirror


def reset_active_mirror() -> None:
    """Forget the process-wide mirror (tests; the env may have changed)."""
    global _ACTIVE, _ACTIVE_DISABLED
    with _ACTIVE_LOCK:
        _ACTIVE = None
        _ACTIVE_DISABLED = False


def mirror_log(
    data: Any, *, step: int | None = None, commit: bool | None = None
) -> None:
    """Best-effort mirror of one ``wandb.log`` call; never raises."""
    try:
        mirror = active_mirror()
        if mirror is not None and isinstance(data, Mapping):
            mirror.log(data, step=step, commit=commit)
    except Exception:
        pass


def mirror_flush() -> None:
    mirror = _ACTIVE
    if mirror is not None:
        try:
            mirror.flush()
        except Exception:
            pass


# ── Config ─────


@dataclass
class DashboardMetricConfig(MetricConfig):
    """Log framework metrics to the Training Gym dashboard only.

    Slime and Miles keep calling ``wandb.*``; a W&B-shaped shim routes the
    scalars to the dashboard's Metrics tab. No W&B account, API key or Trackio
    server is involved.

    Attributes:
        project: Metric project name (used for the app tags and CLI flags).
        group: Group tag for related runs.
        exp_name: Run display name.
    """

    project: str = "training-gym"
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True

    provider: ClassVar[str] = "dashboard"  # pyright: ignore[reportIncompatibleMethodOverride]


def mirror_mode(metric: MetricConfig | None) -> str:
    """Which bootstrap mode ``metric`` needs in the container ("" for none)."""
    if metric is None:
        return ""
    if metric.provider == DashboardMetricConfig.provider:
        return MIRROR_SINK
    return MIRROR_TEE if getattr(metric, "mirror_to_dashboard", False) else ""


def mirror_pth_files() -> dict[str, str]:
    return {_PTH_FILE: _PTH_LINE}


def pth_install_command(files: Mapping[str, str]) -> str:
    """Shell command writing each ``.pth`` into the image's site-packages."""
    writes = "; ".join(
        f"pathlib.Path(site.getsitepackages()[0], {name!r}).write_text({line!r})"
        for name, line in files.items()
    )
    return f"python3 -c {shlex.quote(f'import pathlib, site; {writes}')}"


# ── Container bootstrap ─────


def bootstrap() -> None:
    """Entry point of the ``.pth`` file; selects the mode from the environment."""
    mode = os.environ.get(MIRROR_MODE_ENV, "").strip()
    if mode == MIRROR_SINK:
        install_wandb_shim()
    elif mode == MIRROR_TEE:
        install_wandb_tee()


# ── tee: patch the real wandb ─────


class _TeeLoader:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec: ModuleSpec) -> Any:
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        patch_wandb_module(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _WandbTeeFinder(MetaPathFinder):
    """Wraps the loader that imports ``wandb`` so the module is patched on load."""

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> ModuleSpec | None:
        if fullname != "wandb":
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            spec.loader = _TeeLoader(spec.loader)
            return spec
        return None


def install_wandb_tee() -> None:
    current = sys.modules.get("wandb")
    if current is not None:
        patch_wandb_module(current)
        return
    if any(isinstance(finder, _WandbTeeFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _WandbTeeFinder())


def patch_wandb_module(wandb: Any) -> bool:
    """Tee ``Run.log`` (which ``wandb.log`` binds to after ``init``).

    Idempotent; a no-op for the Trackio or dashboard shims, which mirror on
    their own.
    """
    if getattr(wandb, _SHIM_MARKER, False) or getattr(
        wandb, "_training_gym_trackio_adapter", False
    ):
        return False
    if getattr(wandb, _TEE_MARKER, False):
        return True
    try:
        run_module = sys.modules.get("wandb.sdk.wandb_run")
        if run_module is None:
            from importlib import import_module

            run_module = import_module("wandb.sdk.wandb_run")
        run_cls = run_module.Run
    except Exception:
        return False
    original_log = run_cls.log
    original_finish = getattr(run_cls, "finish", None)

    def log(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        step = kwargs.get("step", args[0] if len(args) > 0 else None)
        commit = kwargs.get("commit", args[1] if len(args) > 1 else None)
        current = getattr(self, "step", None)
        result = original_log(self, data, *args, **kwargs)
        mirror_log(
            data,
            step=step if step is not None else _int_or_none(current),
            commit=commit,
        )
        return result

    log.__name__ = original_log.__name__
    log.__doc__ = original_log.__doc__
    log.__wrapped__ = original_log  # type: ignore[attr-defined]
    run_cls.log = log

    if original_finish is not None:

        def finish(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original_finish(self, *args, **kwargs)
            finally:
                mirror_flush()

        finish.__name__ = original_finish.__name__
        finish.__doc__ = original_finish.__doc__
        run_cls.finish = finish

    setattr(wandb, _TEE_MARKER, True)
    return True


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


# ── sink: a W&B-shaped module over the dashboard ─────


class _Settings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Media:
    """Stand-in for ``wandb.Image``/``Table``/...; dropped by the flattener."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


_MEDIA_NAMES = frozenset(
    {
        "Image",
        "Table",
        "Histogram",
        "Video",
        "Audio",
        "Html",
        "Plotly",
        "Object3D",
        "Molecule",
        "Artifact",
    }
)


class _Config(dict):
    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        allow_val_change = kwargs.pop("allow_val_change", None)
        del allow_val_change
        for arg in args:
            if isinstance(arg, Mapping):
                super().update(arg)
            elif arg is not None:
                super().update(vars(arg))
        super().update(kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class DashboardRun:
    """The object ``wandb.init`` returns in ``sink`` mode."""

    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        project: str,
        group: str | None,
        config: Any,
        mirror: MetricMirror | None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.project = project
        self.group = group
        self.entity = ""
        self.url = ""
        self.dir = os.getcwd()
        self.tags: tuple[str, ...] = ()
        self.notes = ""
        self.summary: dict[str, Any] = {}
        self.config = _Config()
        if config is not None:
            self.config.update(config)
        self._mirror = mirror

    @property
    def step(self) -> int:
        return self._mirror.step if self._mirror is not None else 0

    def log(
        self,
        data: Mapping[str, Any],
        step: int | None = None,
        commit: bool | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        if not isinstance(data, Mapping):
            return
        for key, value in data.items():
            if as_finite_float(value) is not None:
                self.summary[key] = value
        if self._mirror is not None:
            self._mirror.log(data, step=step, commit=commit)

    def finish(self, *_args: Any, **_kwargs: Any) -> None:
        if self._mirror is not None:
            self._mirror.flush()

    def define_metric(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def save(self, *_args: Any, **_kwargs: Any) -> list[str]:
        return []

    def watch(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def unwatch(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_code(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def alert(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def mark_preempting(self) -> None:
        return None

    def __enter__(self) -> DashboardRun:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.finish()


def install_wandb_shim() -> None:
    """Expose the dashboard as the W&B subset used by Slime and Miles."""
    current = sys.modules.get("wandb")
    if current is not None and getattr(current, _SHIM_MARKER, False):
        return

    shim: Any = types.ModuleType("wandb")
    shim.__spec__ = ModuleSpec("wandb", loader=None, is_package=True)
    shim.__path__ = []
    setattr(shim, _SHIM_MARKER, True)
    shim.run = None
    shim.config = _Config()
    shim.summary = {}
    shim.Settings = _Settings
    shim.__version__ = "0.0.0+training-gym-dashboard"

    def init(*args: Any, **kwargs: Any) -> DashboardRun:
        project = kwargs.get("project", args[0] if args else "") or "training-gym"
        run_id = str(kwargs.get("id") or kwargs.get("name") or generate_id())
        run = DashboardRun(
            run_id=run_id,
            name=str(kwargs.get("name") or run_id),
            project=str(project),
            group=kwargs.get("group"),
            config=kwargs.get("config"),
            mirror=active_mirror(),
        )
        shim.run = run
        shim.config = run.config
        shim.summary = run.summary
        return run

    def log(
        data: Mapping[str, Any],
        step: int | None = None,
        commit: bool | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        run = shim.run
        if run is None:
            mirror_log(data, step=step, commit=commit)
            return
        run.log(data, step=step, commit=commit)

    def finish(*_args: Any, **_kwargs: Any) -> None:
        run = shim.run
        try:
            if run is not None:
                run.finish()
            else:
                mirror_flush()
        finally:
            shim.run = None

    def generate_id() -> str:
        return uuid.uuid4().hex[:8]

    def module_getattr(name: str) -> Any:
        if name in _MEDIA_NAMES:
            return _Media
        raise AttributeError(
            f"module 'wandb' (training-gym shim) has no attribute {name!r}"
        )

    shim.init = init
    shim.log = log
    shim.finish = finish
    shim.save = lambda *args, **kwargs: []
    shim.login = lambda *args, **kwargs: True
    shim.define_metric = lambda *args, **kwargs: None
    shim.watch = lambda *args, **kwargs: None
    shim.alert = lambda *args, **kwargs: None
    shim.setup = lambda *args, **kwargs: None
    shim.teardown = lambda *args, **kwargs: None
    shim.__getattr__ = module_getattr

    util: Any = types.ModuleType("wandb.util")
    util.__spec__ = ModuleSpec("wandb.util", loader=None)
    util.generate_id = generate_id
    sdk: Any = types.ModuleType("wandb.sdk")
    sdk.__spec__ = ModuleSpec("wandb.sdk", loader=None, is_package=True)
    sdk.__path__ = []
    lib: Any = types.ModuleType("wandb.sdk.lib")
    lib.__spec__ = ModuleSpec("wandb.sdk.lib", loader=None, is_package=True)
    lib.__path__ = []
    runid: Any = types.ModuleType("wandb.sdk.lib.runid")
    runid.__spec__ = ModuleSpec("wandb.sdk.lib.runid", loader=None)
    runid.generate_id = generate_id
    shim.util = util
    shim.sdk = sdk
    sdk.lib = lib
    lib.runid = runid

    sys.modules.update(
        {
            "wandb": shim,
            "wandb.util": util,
            "wandb.sdk": sdk,
            "wandb.sdk.lib": lib,
            "wandb.sdk.lib.runid": runid,
        }
    )
