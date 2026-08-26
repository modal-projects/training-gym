import dataclasses as _dc
import json
import os
from abc import ABC
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from modal_training_gym.train_recipes.gpu_allocation import (
    GpuAllocation,
    resolve_gpu_allocation,
)

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.metrics import MetricConfig
    from modal_training_gym.common.models import ModelConfig

# ── Volume mount paths (shared by every framework) ───────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# Recipe fields whose dict values are emitted as JSON CLI arguments.
JSON_CONFIG_FIELDS = ("train_env_vars", "apply_chat_template_kwargs", "multimodal_keys")


class RecipeType(Enum):
    SLIME = "slime"
    MILES = "miles"


class BaseTrainRecipe(ABC):
    recipe_type: RecipeType
    model_config_class: ClassVar["type[ModelConfig] | None"] = None

    # Fields consumed by the Modal launcher (image build, cluster topology,
    # callable shipping) and never forwarded to the framework CLI. Every
    # framework recipe overrides this; a name missing from it leaks onto the
    # command line and the framework's argparse aborts with "unrecognized
    # arguments".
    _SKIP_FIELDS: ClassVar[frozenset[str]] = frozenset()

    # Field holding the inline-YAML escape-hatch dict. ``_emit_fields`` renames
    # it to ``_ESCAPE_HATCH_FLAG`` on the command line, and keys inside it always
    # win over same-named top-level fields.
    _ESCAPE_HATCH_FIELD: ClassVar[str] = "extra_config"
    _ESCAPE_HATCH_FLAG: ClassVar[str] = "custom_config_path"

    # ── Callable → import path ────────────────────────────────────────────────

    @staticmethod
    def _callable_path(fn: Callable) -> str:
        """Dotted import path a container can resolve this callable by.

        A function defined in a ``__main__`` tutorial script has no importable
        module name, so fall back to its file stem — which is what the launcher
        mounts it as. ``__pending__`` marks a callable with no readable source
        file, for the launcher to fill in when it ships it.
        """
        mod = getattr(fn, "__module__", None) or ""
        name = getattr(fn, "__qualname__", None) or fn.__name__
        if mod == "__main__":
            import inspect

            try:
                src_file = inspect.getfile(fn)
                mod = Path(src_file).stem if os.path.isfile(src_file) else "__pending__"
            except (TypeError, OSError):
                mod = "__pending__"
        return f"{mod}.{name}"

    @classmethod
    def _path_or_callable_path(cls, value: "Callable | str | None") -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return cls._callable_path(value)

    # ── Model presets ─────────────────────────────────────────────────────────

    # TOOD(joy): Remove merge_model_recipe logic everywhere.
    @classmethod
    def get_base_recipe(cls, model_config: "ModelConfig") -> "BaseTrainRecipe | None":
        """Known-model preset recipe for ``model_config``, or ``None``.

        ``TrainConfig.merge_model_recipe`` merges the returned preset onto the
        fields a user left unset. ``None`` means "no preset for this model" and
        the user's recipe is used as written.

        A framework that only supports an explicit allow-list of models (e.g.
        slime) may instead raise ``TrainingGymConfigError`` for a model it has
        no preset for, rather than returning ``None``.
        """
        return None

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        """Preflight the parallelism plan. Overridden per framework."""
        return None

    # ── Container → framework flag converters ────────────────────────────────

    @staticmethod
    def _resolve_data_paths(
        ds: "DatasetConfig",
    ) -> tuple[str, dict[str, str] | None]:
        """Derive on-volume file paths from a dataset's properties."""
        hf_repo = getattr(ds, "hf_repo", "")
        name = hf_repo.replace("/", "_") if hf_repo else type(ds).__name__
        ext = "jsonl" if ds.output_format == "jsonl" else "parquet"
        split = getattr(ds, "hf_split", "train")
        prompt_data = f"{DATA_PATH}/{name}/{split}.{ext}"
        if getattr(ds, "writes_eval_paths", True):
            return prompt_data, {"eval": f"{DATA_PATH}/{name}/eval.{ext}"}
        return prompt_data, None

    @classmethod
    def _dataset_to_fields(cls, ds: "DatasetConfig") -> dict[str, Any]:
        prompt_data, eval_paths = cls._resolve_data_paths(ds)
        eval_prompt_data: list[str] | None = None
        if eval_paths:
            eval_prompt_data = [
                v for name, path in eval_paths.items() for v in (name, path)
            ]
        return {
            "prompt_data": prompt_data,
            "eval_prompt_data": eval_prompt_data,
            "input_key": ds.input_key,
            "label_key": ds.label_key,
            "apply_chat_template": ds.apply_chat_template,
        }

    @staticmethod
    def _metrics_to_fields(metric: "MetricConfig") -> dict[str, Any]:
        from modal_training_gym.common.metrics import metric_cli_fields

        return metric_cli_fields(metric)

    # ── CLI serialization ─────────────────────────────────────────────────────

    def _field_values(self) -> dict[str, Any]:
        """Every declared field's current value, as the starting point for CLI emission."""
        return {f.name: getattr(self, f.name) for f in _dc.fields(self)}

    def _escape_hatch_keys(self) -> tuple[str, ...]:
        """Keys set in the inline-YAML escape hatch, on either side of materialization.

        ``prepare_launch_config`` replaces the dict with a path to the YAML it
        wrote *before* the launcher calls ``cli_args``, so reading the field
        alone would see a ``str`` and silently skip the precedence rule in
        ``_emit_fields``. It records the keys on the recipe as it materializes;
        fall back to those.
        """
        val = getattr(self, self._ESCAPE_HATCH_FIELD, None)
        if isinstance(val, dict):
            return tuple(val)
        return tuple(getattr(self, "_materialized_config_keys", ()) or ())

    def _emit_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Drop launcher-only fields and let the escape hatch win over same-named flags.

        The escape hatch is an explicit per-recipe override, but the frameworks
        resolve ``--<flag>`` CLI args *over* the YAML custom-config — so a key a
        recipe sets in both places would be clobbered by the field's own flag
        (e.g. a ``qkv_format="thd"`` default overriding an ``extra_config``
        ``"bshd"``). Drop any such flag so the YAML value stands.
        """
        out = {k: v for k, v in fields.items() if k not in self._SKIP_FIELDS}
        hatch = self._ESCAPE_HATCH_FIELD
        for key in self._escape_hatch_keys():
            if key != hatch:
                out.pop(key, None)
        if hatch in out:
            out[self._ESCAPE_HATCH_FLAG] = out.pop(hatch)
        return out

    def _fields(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        """Recipe fields to emit as CLI flags, merged with dataset/model/wandb.

        Not abstract: lightweight subclasses (e.g. test doubles) may skip it,
        in which case ``cli_args`` is unavailable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _fields() to use cli_args()"
        )

    def cli_args(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> list[str]:
        out: list[str] = []
        for key, val in self._fields(dataset=dataset, model=model).items():
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

    # ── Cluster topology ──────────────────────────────────────────────────────

    @property
    def total_nodes(self) -> int:
        return self.gpu_allocation.total_nodes

    @property
    def gpu_allocation(self) -> GpuAllocation:
        return resolve_gpu_allocation(self, warn=False)
