"""Model configs supported by the CI validation run.

One registry for every framework. Each entry names the model, its
``ModelConfig``, the framework whose base recipe trains it, and whether a
pull request fans it out automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..framework import Framework
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


@dataclass(frozen=True)
class _ValidationConfig:
    """One model/framework pair the validation harness runs."""

    # Short name used as ``check --model`` and as the CI matrix entry.
    name: str
    model_config: type[ModelConfig]
    # Which framework's ``get_base_recipe`` trains this model.
    framework: Framework
    # Whether a pull request fans this model out automatically. ``False`` is
    # not "never in CI" — a workflow_dispatch naming it still runs it; it only
    # keeps the model out of the diff-driven PR matrix.
    run_on_pr: bool = True

    @property
    def model_name(self) -> str:
        """The HF repo id, e.g. ``moonshotai/Kimi-K2.5``."""
        return self.model_config.model_name

    @classmethod
    def select(
        cls, framework: Framework | None = None, *, pr_only: bool = True
    ) -> list["_ValidationConfig"]:
        """Registry entries, name-sorted, by default only the PR-matrix set.

        Narrow is the default here, unlike the ``list`` CLI: the caller that
        matters is ``diff_impact``, and a bare ``select()`` that quietly
        included Kimi would put 16 x 8 H200 on a pull request.
        """
        return sorted(
            (
                config
                for config in VALIDATION_CONFIGS
                if (framework is None or config.framework is framework)
                and (config.run_on_pr or not pr_only)
            ),
            key=lambda config: config.name,
        )

    @classmethod
    def find(cls, name: str) -> "_ValidationConfig":
        """Look up an entry by short name or HF repo id, case-insensitively."""
        wanted = name.strip().lower()
        for config in VALIDATION_CONFIGS:
            if wanted in (config.name.lower(), config.model_name.lower()):
                return config
        available = ", ".join(config.name for config in cls.select(pr_only=False))
        raise ValueError(f"unknown model {name!r}; available: {available}")


VALIDATION_CONFIGS: set[_ValidationConfig] = {
    _ValidationConfig("Qwen3-0.6B", Qwen3_0_6B, Framework.SLIME),
    _ValidationConfig("Qwen3-1.7B", Qwen3_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-4B", Qwen3_4B, Framework.SLIME),
    _ValidationConfig("Qwen3-8B", Qwen3_8B, Framework.SLIME),
    _ValidationConfig("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-VL-8B-Instruct", Qwen3_VL_8B, Framework.SLIME),
    _ValidationConfig("Qwen3.6-35B-A3B", Qwen3_6_35B, Framework.SLIME),
    # Too large to fan out on a PR (16 x 8 H200), but still dispatchable by
    # name. Flipping run_on_pr is the only change needed to gate PRs on one.
    _ValidationConfig("Kimi-K2.5", Kimi_K2_5, Framework.MILES, run_on_pr=False),
    _ValidationConfig("Kimi-K2.6", Kimi_K2_6, Framework.MILES, run_on_pr=False),
}
