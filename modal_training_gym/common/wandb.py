"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeRecipe emits `--wandb-project`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration shared across all frameworks.

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
    modal_wandb_secret_name : str
        Name of the Modal secret containing the W&B API key. Default ``"wandb-secret"``.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    key: str = ""
    disable_random_suffix: bool = True
    modal_wandb_secret_name: str = "wandb-secret"


def append_megatron_wandb_args(
    argv: list[str], wandb: "WandbConfig | None"
) -> list[str]:
    """Append Megatron W&B CLI args for a configured run."""
    if wandb is None:
        return argv

    argv.append("--use-wandb")
    if wandb.project:
        argv.extend(["--wandb-project", wandb.project])
    if wandb.group:
        argv.extend(["--wandb-group", wandb.group])
    if wandb.key:
        argv.extend(["--wandb-key", wandb.key])
    if wandb.disable_random_suffix:
        argv.append("--disable-wandb-random-suffix")
    return argv
