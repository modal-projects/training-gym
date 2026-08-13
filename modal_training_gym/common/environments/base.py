"""Base abstractions for live, sandbox-backed RL environments.

This module is the environment analogue of ``common/models/base.py`` and
``common/dataset.py``: it defines the *shapes* that flow in and out of an
environment and the lifecycle base classes that concrete environments
(e.g. :mod:`modal_training_gym.common.environments.toolathlon`) implement.

Three layers, from generic to specific:

1. **I/O shapes** — :class:`ToolCall`, :class:`Observation`, :class:`StepResult`,
   :class:`EvalVerdict`. Plain dataclasses describing what an agent sends to an
   environment and what it gets back.
2. **Environment lifecycle** — :class:`Environment` (one live episode: ``step`` /
   ``evaluate`` / ``close``) and :class:`SandboxEnvironment` (an ``Environment``
   whose state lives in a Modal sandbox).
3. **Pooling + snapshots** — :class:`SandboxEnvironmentPool` (create/reuse warm
   sandboxes per acquire) and :class:`DirectorySnapshotLibrary` (capture and
   restore environment state as Modal directory snapshots keyed by
   ``(item, step)``).

Nothing here is specific to any benchmark, model, or tutorial. Subclasses
supply the image, the tool transport, and the grading.

Like ``ModelConfig`` / ``DatasetConfig``, these are plain base classes (not
``abc.ABC``): unimplemented hooks raise ``NotImplementedError`` so a subclass
can override only what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.base import ToolCall


@dataclass
class Observation:
    """The environment's response to a tool call.

    text : str
        Human/agent-readable result of the tool call (tool output, error message).
    is_error : bool
        Whether the tool call failed (e.g. a tool raised). Lets a caller bail out
        of a flailing episode without parsing ``text``.
    metadata : dict
        Optional structured extras (raw payload, status codes, …).
    """

    text: str = ""
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of one :meth:`Environment.step`.

    observation : Observation
        What the environment returned for the tool call.
    done : bool
        Whether the episode should end (terminal tool call, fatal error, …).
    info : dict
        Optional per-step diagnostics.
    """

    observation: Observation
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalVerdict:
    """Terminal grade of an environment's final state.

    passed : bool
        Whether the task's own evaluator considers the end state correct.
    detail : str
        Human-readable explanation (evaluator output, failure reason).
    harness_error : bool
        ``True`` when grading itself crashed (bad config / broken harness)
        rather than the task legitimately failing — so harness breakage isn't
        silently scored as a string of task failures.
    """

    passed: bool
    detail: str = ""
    harness_error: bool = False
    # For additional metadata like test cases
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Tool schemas ─────────────────────────────────────────────────────────────


def render_tool_catalog(tool_schemas: dict, desc_chars: int = 160) -> str:
    """Render tool schemas as compact ``name(arg, required*)`` lines."""
    lines = []
    for name in sorted(tool_schemas or {}):
        spec = tool_schemas[name]
        if isinstance(spec, dict) and ("parameters" in spec or "description" in spec):
            desc, params = spec.get("description", ""), spec.get("parameters", {})
        else:
            desc, params = "", spec
        props = (params or {}).get("properties", {}) if isinstance(params, dict) else {}
        required = (
            set((params or {}).get("required", []) or [])
            if isinstance(params, dict)
            else set()
        )
        sig = ", ".join(f"{key}*" if key in required else key for key in props)
        desc = " ".join(str(desc).split())[:desc_chars]
        lines.append(f"- {name}({sig})" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


def tool_schemas_to_openai(tool_schemas: dict) -> list[dict]:
    """Convert a tool-schema mapping into the OpenAI/HF ``tools=`` shape."""
    tools = []
    for name in sorted(tool_schemas or {}):
        spec = tool_schemas[name]
        if isinstance(spec, dict) and ("parameters" in spec or "description" in spec):
            desc, params = spec.get("description", ""), spec.get("parameters", {})
        else:
            desc, params = "", spec
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


# ── Environment lifecycle ───────────────────────────────────────────────────


class Environment:
    """A single live environment episode.

    Subclasses implement :meth:`step` (apply a tool call) and, when the task is
    gradable, :meth:`evaluate` (score the final state). :meth:`close` releases
    any resources. Supports the context-manager protocol so callers can write
    ``with pool.acquire(...) as env:``.
    """

    def step(self, action: ToolCall) -> StepResult:
        """Apply ``action`` and return the resulting :class:`StepResult`."""
        raise NotImplementedError(f"{type(self).__name__} has no step()")

    def evaluate(self) -> EvalVerdict:
        """Grade the environment's current/final state."""
        raise NotImplementedError(f"{type(self).__name__} has no evaluate()")

    def close(self) -> None:
        """Release resources held by this episode (idempotent, never raises)."""

    def __enter__(self) -> "Environment":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SandboxEnvironment(Environment):
    """An :class:`Environment` whose mutable state lives inside a Modal sandbox.

    Holds the ``modal.Sandbox`` handle; :meth:`close` terminates it. Subclasses
    implement :meth:`step` / :meth:`evaluate` by ``exec``-ing inside the sandbox.
    """

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def close(self) -> None:
        try:
            self.sandbox.terminate()
        except Exception:
            pass


# ── Pooling ──────────────────────────────────────────────────────────────


class SandboxEnvironmentPool:
    """Creates sandbox-backed environments on demand, sharing one Modal ``App``.

    Subclasses override :meth:`build_image` (the base sandbox image) and
    :meth:`acquire` (turn acquire-args into a ready :class:`Environment`), and
    can use :meth:`create_sandbox` as a helper. The default :meth:`release`
    closes the env.

    **Cloudpickle safety.** A pool is often instantiated client-side (e.g. a
    base eval) and then captured when a rollout/reward function is cloudpickled
    *by value* to remote workers. Live ``modal.App`` handles hold asyncio
    futures and aren't picklable, so :meth:`__getstate__` drops them; they are
    re-created lazily via :meth:`app_handle` on the worker.
    """

    def __init__(self) -> None:
        self._app: Any = None

    @property
    def app_name(self) -> str:
        """Stable Modal app name shared by this pool class' sandboxes.

        The default is derived from the pool class, not stored on the instance.
        Some rollout code creates one pool object per task/environment; if those
        instances synthesize unique app names, Modal sees one app per sandbox.
        Keeping this read-only by default preserves the desired
        many-sandboxes-to-one-app shape.
        """
        return f"{type(self).__name__.replace('_', '-').lower()}-sandboxes"

    def __getstate__(self) -> dict:
        # Drop the live App handle so cloudpickle can ship the pool by value.
        state = dict(self.__dict__)
        state["_app"] = None
        return state

    def app_handle(self) -> Any:
        """The shared ``modal.App`` (created/looked up lazily)."""
        import modal

        if self._app is None:
            app_name = self.app_name
            if not app_name:
                raise TrainingGymConfigError(
                    f"{type(self).__name__}.app_name must be set"
                )
            self._app = modal.App.lookup(app_name, create_if_missing=True)
        return self._app

    def build_image(self) -> Any:
        """Return the base ``modal.Image`` sandboxes are created from."""
        raise NotImplementedError(f"{type(self).__name__} has no build_image()")

    def create_sandbox(
        self,
        *,
        image: Any = None,
        command: tuple[str, ...] = ("sleep", "infinity"),
        **kwargs: Any,
    ) -> Any:
        """Create a long-lived sandbox from ``image`` (defaults to :meth:`build_image`)."""
        import modal

        return modal.Sandbox._experimental_create(
            *command,
            image=image if image is not None else self.build_image(),
            app=self.app_handle(),
            **kwargs,
        )

    def acquire(self, *args: Any, **kwargs: Any) -> Environment:
        """Return a ready :class:`Environment` for the given acquire-args."""
        raise NotImplementedError(f"{type(self).__name__} has no acquire()")

    def release(self, env: Environment) -> None:
        """Release an environment acquired from this pool."""
        env.close()


# ── Directory snapshots ─────────────────────────────────────────────────────


class DirectorySnapshotLibrary:
    """Capture/restore environment state as Modal directory snapshots.

    A snapshot library is a ``modal.Dict`` mapping ``"{item}/{step}"`` to a
    *directory* snapshot ``Image`` of ``root`` at that step. The env pool mounts
    these by key to restore the world to a given step.

    **The remountable-snapshot rule.** ``mount_image`` of a directory snapshot
    only works if the snapshotted path was itself a *mounted overlay*. So
    :meth:`build_item` mounts an empty ``Image.from_scratch()`` at ``root``
    *before* seeding state, then ``snapshot_directory(root)``. Snapshotting a
    plain base-image directory yields a snapshot Modal's backend can't remount.

    Handles are looked up lazily so the library can be imported on remote
    workers without a Modal round-trip at import time.
    """

    def __init__(self, catalog_name: str, root: str) -> None:
        self.catalog_name = catalog_name
        self.root = root
        self._catalog: Any = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_catalog"] = None
        return state

    def catalog(self) -> Any:
        """The backing ``modal.Dict`` (created/looked up lazily)."""
        import modal

        if self._catalog is None:
            self._catalog = modal.Dict.from_name(
                self.catalog_name, create_if_missing=True
            )
        return self._catalog

    @staticmethod
    def key(item: str, step: int) -> str:
        return f"{item}/{step}"

    def get(self, item: str, step: int) -> Any:
        """The snapshot ``Image`` for ``(item, step)``."""
        return self.catalog()[self.key(item, step)]

    def has(self, item: str, step: int) -> bool:
        return self.key(item, step) in self.catalog()

    # TODO(joyliu-q/atoniolo76) Switch to named images when feature releases
    def missing_steps(self, item: str, n_steps: int) -> list[int]:
        """Steps in ``0..n_steps`` with no cataloged snapshot.

        Reads each present entry (not just a membership check) so the check
        also refreshes the entry's server-side TTL — ``modal.Dict`` entries
        expire after 7 days of inactivity (no reads or writes).
        """
        catalog = self.catalog()
        return [
            step
            for step in range(n_steps + 1)
            if catalog.get(self.key(item, step)) is None
        ]

    def mount(self, sandbox: Any, item: str, step: int) -> None:
        """Mount the ``(item, step)`` snapshot at ``root`` inside ``sandbox``."""
        sandbox.mount_image(self.root, self.get(item, step))

    def build_item(self, sandbox: Any, item: str, n_steps: int, seed, advance) -> int:
        """Snapshot ``root`` at every step ``0..n_steps`` of one item.

        ``sandbox`` is a freshly created sandbox. The sequence is:

        1. mount an empty overlay at ``root`` (so the snapshot is remountable);
        2. ``seed(sandbox)`` — materialize the step-0 state into the overlay
           (and start any helper process needed to advance, e.g. a tool server);
        3. for each step, ``snapshot_directory(root)`` then, unless it's the last
           step, ``advance(sandbox, step)`` to mutate the world from ``step`` to
           ``step + 1`` (e.g. replay one expert tool call).

        Returns the number of snapshots written (``n_steps + 1``).
        """
        import modal

        # Empty writable overlay so the directory snapshot is remountable.
        sandbox.mount_image(self.root, modal.Image.from_scratch())
        seed(sandbox)
        catalog = self.catalog()
        for step in range(n_steps + 1):
            catalog[self.key(item, step)] = sandbox.snapshot_directory(self.root)
            if step < n_steps:
                advance(sandbox, step)
        return n_steps + 1
