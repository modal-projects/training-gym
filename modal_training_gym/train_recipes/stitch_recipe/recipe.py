"""Recipe for disaggregated GRPO training on Modal via stitch.

A stitch run is two halves that meet at a weight-delta bulletin board, so the
recipe is two fields::

    StitchRecipe(
        train=StitchTrainConfig(...),   # the miles actor cluster that publishes
        serve=StitchServeConfig(...),   # the Flash pool that applies
    )

The trainer half is a :class:`MilesRecipe` — stitch is trainer-agnostic, so the
trainer is a *field* rather than a base class and its flags are maintained in one
place instead of copied. The serving half wraps the same
:class:`~modal_training_gym.deploy_recipes.sglang_recipe.recipe.SglangRecipe` the
deploy path uses.

Cross-half settings (rollout parallelism, the delta transport contract, the
served baseline, W&B) are *derived*, never set twice: a mismatch would otherwise
only surface as a stalled rollout twenty minutes into a Modal run.

This is the training-gym packaging of the ``stitch`` ``miles_disagg`` cookbook
(https://github.com/modal-projects/stitch/tree/main/cookbook/miles_disagg): the
``stitch`` library supplies the bulletin protocol + sidecar + miles hooks, and
this recipe + :func:`build_stitch_app` play the role the cookbook's config +
``app.py`` play there.

It launches like the other recipes — ``TrainConfig(model=..., dataset=...,
recipe=StitchRecipe(...)).train()`` — with the Flash rollout pool coming up as
part of the same app.
"""

from __future__ import annotations

import json
from dataclasses import field
from typing import Any

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    JSON_CONFIG_FIELDS,
    YAML_CONFIG_FIELDS,
)
from modal_training_gym.train_recipes.stitch_recipe.serve import StitchServeConfig
from modal_training_gym.train_recipes.stitch_recipe.train import (
    HOOK_CONFIG_FIELDS,
    StitchTrainConfig,
)

__all__ = [
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "HF_CACHE_PATH",
    "HOOK_CONFIG_FIELDS",
    "JSON_CONFIG_FIELDS",
    "YAML_CONFIG_FIELDS",
    "StitchRecipe",
    "StitchServeConfig",
    "StitchTrainConfig",
    "fields_to_argv",
]


def fields_to_argv(fields: dict[str, Any]) -> list[str]:
    """miles argv for a field dict, matching :meth:`MilesRecipe.cli_args`.

    The trainer runs from a plain field dict rather than the recipe object (the
    recipe doesn't survive the trip to a Ray actor), so the same encoding lives
    here as a function.
    """
    out: list[str] = []
    for key, val in fields.items():
        if val is None or val is False or val == "":
            continue
        flag = f"--{key.replace('_', '-')}"
        if val is True:
            out.append(flag)
        elif isinstance(val, dict) and key in JSON_CONFIG_FIELDS:
            out += [flag, json.dumps(val)]
        elif isinstance(val, list):
            out += [flag] + [str(v) for v in val]
        else:
            out += [flag, str(val)]
    return out


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class StitchRecipe(BaseTrainRecipe):
    """Disaggregated GRPO on Modal: a publishing trainer and a syncing pool.

    ## Fields

    train : StitchTrainConfig
        The miles actor cluster. Every :class:`MilesRecipe` field applies,
        defaulted for a publish-only disaggregated run.
    serve : StitchServeConfig
        The Modal Flash pool of SGLang replicas that serves rollouts and applies
        published deltas in place.
    name : str
        Modal app name. Empty → derived from the model name.
    app_tags : dict
        Extra Modal app tags, merged over the standard training-gym ones.
    wandb : WandbConfig | None
        Applied to the trainer half and to the app's dashboard tags.
    """

    recipe_type: RecipeType = RecipeType.STITCH

    train: StitchTrainConfig = field(default_factory=StitchTrainConfig)
    serve: StitchServeConfig = field(default_factory=StitchServeConfig)

    name: str = ""
    app_tags: dict = field(default_factory=dict)
    wandb: WandbConfig | None = None

    @model_validator(mode="after")
    def _resolve_halves(self) -> StitchRecipe:
        """Derive the settings both halves have to agree on, and reject the ones
        that can't be reconciled."""
        train, serve = self.train, self.serve
        if not isinstance(train, StitchTrainConfig):
            raise TypeError(
                "StitchRecipe.train must be a StitchTrainConfig (stitch's miles "
                f"trainer); got {type(train).__name__}"
            )
        if self.wandb is not None:
            train.wandb = self.wandb
        # Rollout parallelism is a property of a replica, so the pool owns it and
        # the trainer follows: miles' router sizes its fan-out from this.
        train.rollout_num_gpus_per_engine = serve.gpus_per_replica
        # Publish-only: rollouts come from the pool, so the actor cluster holds
        # no engines of its own.
        train.rollout_num_gpus = 0
        train.colocate = False
        # The bulletin board is a shared Modal Volume the sidecar reloads from,
        # so neither half can opt out of the transport alone.
        if train.update_weight_transfer_mode != "disk-delta":
            raise ValueError(
                "the stitch pool applies sparse deltas from the bulletin Volume; "
                "train.update_weight_transfer_mode must be 'disk-delta', got "
                f"{train.update_weight_transfer_mode!r}"
            )
        # The replicas serve — and apply deltas against — exactly what checkpoint
        # prep built for the trainer to export from.
        if not serve.served_checkpoint_path:
            serve.served_checkpoint_path = train.hf_checkpoint
        elif serve.served_checkpoint_path != train.hf_checkpoint:
            raise ValueError(
                "the pool's baseline must be the trainer's export baseline: "
                f"serve.served_checkpoint_path={serve.served_checkpoint_path!r} "
                f"!= train.hf_checkpoint={train.hf_checkpoint!r}"
            )
        if train.served_checkpoint_format == "nvfp4":
            # A quantized baseline is built by prepare_checkpoints, so it has to
            # be a local path on the checkpoints Volume, not a repo id — and the
            # trainer needs separate BF16 masters to train from.
            if not str(train.hf_checkpoint).startswith("/"):
                raise ValueError(
                    "an nvfp4 run serves the locally converted checkpoint; "
                    f"train.hf_checkpoint must be a path, got {train.hf_checkpoint!r}"
                )
            if not str(train.bf16_checkpoint_path).startswith("/"):
                raise ValueError(
                    "an nvfp4 run trains from BF16 masters; "
                    "train.bf16_checkpoint_path must be a path, got "
                    f"{train.bf16_checkpoint_path!r}"
                )
        return self

    # ── Converters (delegated to the trainer half) ──────────────────────────

    @staticmethod
    def _resolve_data_paths(ds: DatasetConfig) -> tuple[str, dict[str, str] | None]:
        return StitchTrainConfig._resolve_data_paths(ds)

    def miles_fields(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """Resolved miles CLI fields (name → value), excluding infra + the fields
        the trainer injects per launch (``rollout_endpoint_url``,
        ``update_weight_disk_dir``, ``custom_config_path``)."""
        return self.train._fields(dataset=dataset, model=model)

    def cli_args(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> list[str]:
        """The trainer half's miles CLI argv. YAML config fields
        (:data:`YAML_CONFIG_FIELDS`) are materialized to files by the launcher,
        which then appends the resolved flags."""
        return self.train.cli_args(dataset=dataset, model=model)

    def to_payload(
        self,
        *,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """Plain-data miles args the trainer runs with."""
        train = self.train
        return {
            "fields": self.miles_fields(model=model, dataset=dataset),
            "async_mode": train.async_mode,
            "miles_model_script": train.miles_model_script,
        }
