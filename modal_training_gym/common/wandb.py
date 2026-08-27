"""Weights & Biases run metadata.

Pure data — each framework config writes its own converter from this to its
specific CLI flags (e.g. SlimeRecipe emits `--wandb-project`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import quote

from modal_training_gym.common.metrics import MetricConfig


@dataclass
class WandbConfig(MetricConfig):
    """Weights & Biases logging configuration shared across all frameworks.

    ## Fields

    project : str
        W&B project name. Default ``""``.
    entity : str
        W&B entity/team slug. Optional; when omitted, preflight resolves the
        default entity for the configured API key.
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
    entity: str = ""
    group: str = ""
    exp_name: str = ""
    key: str = ""
    disable_random_suffix: bool = True
    modal_wandb_secret_name: str = "wandb-secret"

    provider: ClassVar[str] = "wandb"

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        env = super().runtime_env(run_id=run_id, entity=entity)
        if run_id:
            env.update(WANDB_RUN_ID=run_id, WANDB_RESUME="allow")
        if entity:
            env["WANDB_ENTITY"] = entity
        return env

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        entity = (entity or self.entity).strip()
        project = self.project.strip()
        if not entity or not project:
            return None
        base = f"https://wandb.ai/{quote(entity, safe='')}/{quote(project, safe='')}"
        return f"{base}/runs/{quote(run_id, safe='')}" if run_id else base


def preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Return the resolved W&B entity for constructing deep links."""
    key = os.environ.get("WANDB_API_KEY", "") or wandb_cfg.key
    if not key:
        raise RuntimeError(
            "W&B logging is enabled but no WANDB_API_KEY is available - add it "
            f"to Modal secret '{wandb_cfg.modal_wandb_secret_name}', set "
            "metrics.key, or drop metrics= to disable logging."
        )

    import wandb

    project = wandb_cfg.project or "uncategorized"
    entity = wandb_cfg.entity or os.environ.get("WANDB_ENTITY", "")
    try:
        wandb.login(key=key, verify=True, relogin=True)
        probe = wandb.init(
            project=project,
            entity=entity or None,
            name="_preflight",
            settings=wandb.Settings(silent=True, init_timeout=60),
        )
        entity = probe.entity
        probe_path = f"{probe.entity}/{probe.project}/{probe.id}"
        wandb.finish()
        try:
            wandb.Api(api_key=key).run(probe_path).delete()
        except Exception:
            pass
    except Exception as exc:
        raise RuntimeError(
            f"W&B pre-flight failed for project '{project}': {exc}\n"
            f"The key in Modal secret '{wandb_cfg.modal_wandb_secret_name}' "
            "cannot log there. Fix the secret or drop metrics=."
        ) from exc
    return entity
