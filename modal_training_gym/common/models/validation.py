"""Model configs exercised by the CI validation run.

One registry for every framework. Each entry names the model, its
``ModelConfig``, the framework whose base recipe trains it, and whether CI is
allowed to launch it from a pull request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .base import ModelConfig
from .kimi_k2_5 import Kimi_K2_5
from .kimi_k2_6 import Kimi_K2_6
from .qwen3_0_6b import Qwen3_0_6B
from .qwen3_1_7b import Qwen3_1_7B
from .qwen3_4b import Qwen3_4B
from .qwen3_6_35b import Qwen3_6_35B
from .qwen3_8b import Qwen3_8B
from .qwen3_asr_1_7b import Qwen3_ASR_1_7B
from .qwen3_vl_8b import Qwen3_VL_8B


class Framework(str, Enum):
    # ``str`` mixin so a Framework serializes straight to its value in plain
    # ``json.dumps`` (e.g. TrainResult, whose ``asdict`` payload doesn't coerce
    # enums) — matching the SlimeStatus/MilesStatus convention.
    SLIME = "slime"
    MILES = "miles"


@dataclass(frozen=True)
class _ValidationConfig:
    """One model/framework pair the validation harness knows how to run."""

    # Short name used as ``check --model`` and as the CI matrix entry.
    name: str

    model_config: type[ModelConfig]

    # Which framework's ``get_base_recipe`` trains this model.
    framework: Framework

    # Whether a pull request may launch this run.
    ci_enabled: bool = True

    @property
    def model_name(self) -> str:
        """The HF repo id, e.g. ``moonshotai/Kimi-K2.5``."""
        return self.model_config.model_name

    @classmethod
    def select(
        cls, framework: Framework | None = None, *, ci_only: bool = True
    ) -> list["_ValidationConfig"]:
        """Registry entries, name-sorted, narrowed to what CI may launch."""
        return sorted(
            (
                config
                for config in VALIDATION_CONFIGS
                if (framework is None or config.framework is framework)
                and (config.ci_enabled or not ci_only)
            ),
            key=lambda config: config.name,
        )

    @classmethod
    def find(cls, name: str) -> "_ValidationConfig":
        """Look up an entry by short name, case-insensitively."""
        wanted = name.strip().lower()
        for config in VALIDATION_CONFIGS:
            if config.name.lower() == wanted:
                return config
        available = ", ".join(config.name for config in cls.select(ci_only=False))
        raise ValueError(f"unknown model {name!r}; available: {available}")


VALIDATION_CONFIGS: set[_ValidationConfig] = {
    _ValidationConfig("Qwen3-0.6B", Qwen3_0_6B, Framework.SLIME),
    _ValidationConfig("Qwen3-1.7B", Qwen3_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-4B", Qwen3_4B, Framework.SLIME),
    _ValidationConfig("Qwen3-8B", Qwen3_8B, Framework.SLIME),
    _ValidationConfig("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-VL-8B-Instruct", Qwen3_VL_8B, Framework.SLIME),
    _ValidationConfig("Qwen3.6-35B-A3B", Qwen3_6_35B, Framework.SLIME),
    _ValidationConfig("Kimi-K2.5", Kimi_K2_5, Framework.MILES, ci_enabled=False),
    _ValidationConfig("Kimi-K2.6", Kimi_K2_6, Framework.MILES, ci_enabled=False),
}
