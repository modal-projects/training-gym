"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeConfig emits `--wandb-project`, ms-swift emits
`--wandb_project` + `--wandb_exp_name`).
"""

from __future__ import annotations

from typing import Any


class WandbConfig:
    project: str = ""
    group: str = ""  # wandb "group" tag
    exp_name: str = ""  # wandb run name
    key: str = ""  # typically injected from WANDB_API_KEY at launch time
    disable_random_suffix: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
