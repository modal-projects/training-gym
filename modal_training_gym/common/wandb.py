"""Weights & Biases config shared across training frameworks.

Attach a `WandbConfig` to a framework config's `wandb` field; the fields it
contributes (`use_wandb`, `wandb_project`, …) are merged into the framework's
flat field dict at cli-gen time. Attaching a `WandbConfig` implies
`use_wandb=True`; don't attach to disable.

Conversion:
  - `WandbConfig.to_fields()`        → flat dict of framework field values
  - `WandbConfig.from_fields(flat)`  → build a `WandbConfig` from a framework's
                                       flat field dict (reverse direction)
"""

from __future__ import annotations

from typing import Any

# Framework-side field names that a WandbConfig contributes. Keep in sync with
# the attribute ↔ framework-field mapping in `to_fields()` / `from_fields()`.
WANDB_FIELDS: tuple[str, ...] = (
    "use_wandb",
    "wandb_project",
    "wandb_group",
    "wandb_key",
    "disable_wandb_random_suffix",
)


class WandbConfig:
    """Weights & Biases run metadata.

    Class attrs are the user-facing names (`project`, `group`, `key`). They
    map to framework-side field names (`wandb_project`, `wandb_group`,
    `wandb_key`) via `to_fields()`.
    """

    project: str = ""
    group: str = ""
    key: str = ""  # typically injected from WANDB_API_KEY at launch time
    disable_random_suffix: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_fields(self) -> dict[str, Any]:
        """Flat framework-side dict. Presence implies `use_wandb=True`."""
        return {
            "use_wandb": True,
            "wandb_project": self.project,
            "wandb_group": self.group,
            "wandb_key": self.key,
            "disable_wandb_random_suffix": self.disable_random_suffix,
        }

    @classmethod
    def from_fields(cls, fields: dict[str, Any], **extra: Any) -> "WandbConfig":
        """Build a `WandbConfig` from a framework's flat field dict."""
        kwargs: dict[str, Any] = {}
        if "wandb_project" in fields:
            kwargs["project"] = fields["wandb_project"]
        if "wandb_group" in fields:
            kwargs["group"] = fields["wandb_group"]
        if "wandb_key" in fields:
            kwargs["key"] = fields["wandb_key"]
        if "disable_wandb_random_suffix" in fields:
            kwargs["disable_random_suffix"] = fields["disable_wandb_random_suffix"]
        kwargs.update(extra)
        return cls(**kwargs)
