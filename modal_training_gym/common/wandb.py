"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeConfig emits `--wandb-project`, ms-swift emits
`--wandb_project` + `--wandb_exp_name`).
"""

from __future__ import annotations

from typing import Any


class WandbConfig:
    """Weights & Biases logging configuration shared across all frameworks.

    Each framework config converts these fields into its own CLI flags
    (e.g. SlimeConfig emits ``--wandb-project``, ms-swift emits
    ``--wandb_project`` + ``--wandb_exp_name``).

    ## Fields

    project : str
        W&B project name. Default ``""``.
    group : str
        W&B group tag for organizing related runs. Default ``""``.
    exp_name : str
        W&B run display name. Default ``""``.
    key : str
        W&B API key. Usually injected via ``WANDB_API_KEY`` at launch
        time rather than hardcoded. Default ``""``.
    disable_random_suffix : bool
        When ``True``, suppresses the random suffix that W&B appends to
        run names. Default ``True``.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    key: str = ""
    disable_random_suffix: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
