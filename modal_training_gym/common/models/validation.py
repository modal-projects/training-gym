"""Model configs supported by the CI validation run.

One registry for every framework. Each entry names the model, its
``ModelConfig``, the framework whose base recipe trains it, and whether a
pull request fans it out automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..framework import Framework
from .base import ModelConfig
from .gemma4_26b_a4b import Gemma4_26B_A4B
from .glm_5_2 import GLM_5_2, GLM_5_2_5Layer
from .moonlight_16b_a3b_instruct import Moonlight_16B_A3B_Instruct
from .qwen3_0_6b import Qwen3_0_6B
from .qwen3_1_7b import Qwen3_1_7B
from .qwen3_4b import Qwen3_4B
from .qwen3_5_0_8b import Qwen3_5_0_8B
from .qwen3_5_2b import Qwen3_5_2B
from .qwen3_5_4b import Qwen3_5_4B
from .qwen3_5_9b import Qwen3_5_9B
from .qwen3_6_27b import Qwen3_6_27B
from .qwen3_6_35b import Qwen3_6_35B
from .qwen3_8_27b import Qwen3_8_27B
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
    # Whether a pull request fans this model out automatically.
    # A workflow_dispatch naming it still runs it.
    run_on_pr: bool = True
    # Validate the model's projector-only recipe instead of its base one.
    # A model can have both (Qwen3.6-35B-A3B trains RL and trains a projector),
    # and ``get_base_recipe`` only knows the base one, so the entry says which.
    projector: bool = False

    @property
    def model_name(self) -> str:
        """The Hugging Face repository id."""
        return self.model_config.model_name

    @classmethod
    def select(
        cls, framework: Framework | None = None, *, pr_only: bool = True
    ) -> list["_ValidationConfig"]:
        """Registry entries, name-sorted, by default only the PR-matrix set.

        Narrow is the default here, unlike the ``list`` CLI, because the caller
        that matters is ``diff_impact`` and dispatch-only models must stay out
        of pull request matrices.
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
        matches = [
            config
            for config in VALIDATION_CONFIGS
            if wanted in (config.name.lower(), config.model_name.lower())
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # A projector entry shares its base model's HF repo id, so adding one
            # would make that id ambiguous and stop resolving. It is a variant of
            # the base entry rather than a peer, so the repo id keeps meaning the
            # base training run and the projector is asked for by name.
            base = [config for config in matches if not config.projector]
            if len(base) == 1:
                return base[0]
            choices = ", ".join(sorted(config.name for config in matches))
            raise ValueError(f"ambiguous model {name!r}; use one of: {choices}")
        available = ", ".join(config.name for config in cls.select(pr_only=False))
        raise ValueError(f"unknown model {name!r}; available: {available}")


VALIDATION_CONFIGS: set[_ValidationConfig] = {
    _ValidationConfig("Qwen3-0.6B", Qwen3_0_6B, Framework.SLIME),
    _ValidationConfig("Qwen3-1.7B", Qwen3_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-4B", Qwen3_4B, Framework.SLIME),
    _ValidationConfig("Qwen3-8B", Qwen3_8B, Framework.SLIME),
    _ValidationConfig("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-VL-8B-Instruct", Qwen3_VL_8B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-0.8B", Qwen3_5_0_8B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-2B", Qwen3_5_2B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-4B", Qwen3_5_4B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-4B-Miles", Qwen3_5_4B, Framework.MILES),
    _ValidationConfig("Qwen3.5-9B", Qwen3_5_9B, Framework.SLIME),
    _ValidationConfig("Qwen3.6-27B", Qwen3_6_27B, Framework.SLIME, run_on_pr=False),
    _ValidationConfig("Qwen3.6-35B-A3B", Qwen3_6_35B, Framework.SLIME),
    _ValidationConfig("Qwen3.8-27B", Qwen3_8_27B, Framework.SLIME, run_on_pr=False),
    _ValidationConfig(
        "Moonlight-16B-A3B-Instruct",
        Moonlight_16B_A3B_Instruct,
        Framework.MILES,
    ),
    _ValidationConfig(
        "Gemma-4-26B-A4B-it", Gemma4_26B_A4B, Framework.MILES, run_on_pr=False
    ),
    # Projector-only GLM-5.2: the 5-layer checkpoint is the single-node shape
    # upstream smoke-tests on, the full model is 8 nodes. Neither fans out on a
    # PR — dispatch them by name.
    _ValidationConfig(
        "GLM-5.2-5layer-Projector",
        GLM_5_2_5Layer,
        Framework.MILES,
        run_on_pr=False,
    ),
    _ValidationConfig("GLM-5.2-Projector", GLM_5_2, Framework.MILES, run_on_pr=False),
    # Projector-only Qwen3.6-35B-A3B on slime. One 8xH100 node, but the base is
    # a 35B MoE conversion, so it is dispatch-only like the other 27B+ entries.
    _ValidationConfig(
        "Qwen3.6-35B-A3B-Projector",
        Qwen3_6_35B,
        Framework.SLIME,
        run_on_pr=False,
        projector=True,
    ),
}
