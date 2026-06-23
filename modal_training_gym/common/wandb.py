"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeRecipe emits `--wandb-project`).
"""

from __future__ import annotations

import os
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


def preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """
    Returns the resolved W&B entity for constructing deep-links to individual runs.
    """
    key = os.environ.get("WANDB_API_KEY", "") or (wandb_cfg.key or "")
    if not key:
        raise RuntimeError(
            "W&B logging is enabled (recipe.wandb=...) but no WANDB_API_KEY is "
            f"available — add it to the Modal secret "
            f"'{wandb_cfg.modal_wandb_secret_name}' (or set wandb.key=), or drop "
            "wandb= from the recipe to disable logging."
        )

    import wandb

    project = wandb_cfg.project or "uncategorized"
    try:
        wandb.login(key=key, verify=True, relogin=True)
        probe = wandb.init(
            project=project,
            name="_preflight",
            settings=wandb.Settings(silent=True, init_timeout=60),
        )
        entity = probe.entity
        probe_path = f"{probe.entity}/{probe.project}/{probe.id}"
        wandb.finish()
        try:
            wandb.Api(api_key=key).run(probe_path).delete()
        except Exception:
            pass  # a leftover empty "_preflight" run is harmless
    except Exception as exc:
        raise RuntimeError(
            f"W&B pre-flight failed for project '{project}': {exc}\n"
            f"The W&B key in Modal secret '{wandb_cfg.modal_wandb_secret_name}' can't "
            "log there (bad/expired key, or no write access to its entity). Point "
            "recipe.wandb at a project/entity you can write to, fix the secret, or "
            "drop wandb= to disable logging."
        ) from exc
    return entity
