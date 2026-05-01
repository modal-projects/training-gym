# ── Checkpoint ───────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Checkpoint:
    """A single discovered checkpoint on the local filesystem."""

    name: str
    path: str
    timestamp: float


class CheckpointConfig:
    """Checkpoint directory layout, shared across training frameworks.

    Each framework config holds a ``checkpoint`` field of this type.
    Set ``iteration_prefix`` to describe how the framework names
    per-iteration subdirectories (e.g. ``"iter_"``).  Leave it as
    ``""`` for frameworks that write a single flat checkpoint per run.
    """

    iteration_prefix: str = ""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def list_local_checkpoints(self, checkpoint_dir: str) -> list[Checkpoint]:
        """List per-iteration checkpoint directories on the local filesystem.

        Returns :class:`Checkpoint` entries sorted by name for each
        subdirectory under *checkpoint_dir* that starts with
        ``iteration_prefix``.  Returns ``[]`` when the prefix is empty
        or the directory does not exist.
        """
        if not self.iteration_prefix or not os.path.isdir(checkpoint_dir):
            return []
        root = checkpoint_dir.rstrip("/")
        results: list[Checkpoint] = []
        for d in sorted(os.listdir(root)):
            if not d.startswith(self.iteration_prefix):
                continue
            full = f"{root}/{d}"
            results.append(
                Checkpoint(
                    name=d,
                    path=full,
                    timestamp=os.stat(full).st_mtime,
                )
            )
        return results

    def latest_checkpoint(self, checkpoint_dir: str) -> Checkpoint:
        """Return the latest checkpoint under *checkpoint_dir*.

        For frameworks with an ``iteration_prefix``, returns the
        lexicographically largest matching subdirectory.  For frameworks
        without iteration directories, returns *checkpoint_dir* itself.
        """
        if not self.iteration_prefix:
            return Checkpoint(
                name=os.path.basename(checkpoint_dir),
                path=checkpoint_dir,
                timestamp=os.stat(checkpoint_dir).st_mtime
                if os.path.isdir(checkpoint_dir)
                else 0.0,
            )
        ckpts = self.list_local_checkpoints(checkpoint_dir)
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints under {checkpoint_dir} "
                f"with prefix {self.iteration_prefix!r}."
            )
        return ckpts[-1]

