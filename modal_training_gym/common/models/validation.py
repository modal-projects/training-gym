"""Model configs exercised by the CI validation run.

One registry for every framework. Each entry names the model, its
``ModelConfig``, the framework whose base recipe trains it, and whether it is
cheap enough to gate PRs on.

The framework has to be declared rather than derived:
``SlimeRecipe.get_base_recipe`` returns a recipe for every model it is asked
about, so "does slime own this model?" is not a question the recipe classes
can answer.

Consumers:
  - ``scripts/validate_model_configs.py`` dispatches ``check`` on ``framework``
  - ``scripts/diff_impact.py`` builds the per-PR matrix from ``gates_prs``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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


class ValidationFramework(StrEnum):
    """Framework a validation run trains on."""

    SLIME = "slime"
    MILES = "miles"


@dataclass(frozen=True)
class ValidationTarget:
    """One model/framework pair the validation harness knows how to run."""

    name: str
    """Short name used as ``check --model`` and as the CI matrix entry."""

    model_config: type[ModelConfig]

    framework: ValidationFramework
    """Which framework's ``get_base_recipe`` trains this model."""

    gates_prs: bool = True
    """Whether a pull request may launch this run.

    ``False`` keeps a target reachable by name (the CLI, or a
    ``workflow_dispatch`` naming it) while keeping it out of every PR matrix —
    the escape hatch for models too large to gate merges on, like Kimi on
    16 x 8 H200.
    """

    @property
    def model_name(self) -> str:
        """The HF repo id, e.g. ``moonshotai/Kimi-K2.5``."""
        return self.model_config.model_name

    def matches(self, name: str) -> bool:
        """Whether ``name`` refers to this target, by short name or repo id."""
        wanted = name.strip().lower()
        return wanted in (self.name.lower(), self.model_name.lower())


VALIDATION_TARGETS: tuple[ValidationTarget, ...] = (
    ValidationTarget("Qwen3-0.6B", Qwen3_0_6B, ValidationFramework.SLIME),
    ValidationTarget("Qwen3-1.7B", Qwen3_1_7B, ValidationFramework.SLIME),
    ValidationTarget("Qwen3-4B", Qwen3_4B, ValidationFramework.SLIME),
    ValidationTarget("Qwen3-8B", Qwen3_8B, ValidationFramework.SLIME),
    ValidationTarget("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B, ValidationFramework.SLIME),
    ValidationTarget("Qwen3-VL-8B-Instruct", Qwen3_VL_8B, ValidationFramework.SLIME),
    ValidationTarget("Qwen3.6-35B-A3B", Qwen3_6_35B, ValidationFramework.SLIME),
    # Kimi only has a base recipe on miles, and that recipe is 16 x 8 H200 —
    # far too large to launch from every PR. Dispatch-only until a miles recipe
    # exists that is cheap enough to gate merges on; flipping gates_prs on such
    # a model is the only change needed to start validating it per-PR.
    ValidationTarget(
        "Kimi-K2.5", Kimi_K2_5, ValidationFramework.MILES, gates_prs=False
    ),
    ValidationTarget(
        "Kimi-K2.6", Kimi_K2_6, ValidationFramework.MILES, gates_prs=False
    ),
)


def validation_targets(
    framework: ValidationFramework | None = None,
    *,
    gating_only: bool = False,
) -> tuple[ValidationTarget, ...]:
    """Registry entries, optionally narrowed to one framework or the PR set."""
    return tuple(
        target
        for target in VALIDATION_TARGETS
        if (framework is None or target.framework == framework)
        and (not gating_only or target.gates_prs)
    )


def find_validation_target(name: str) -> ValidationTarget:
    """Look up a target by short name or HF repo id, case-insensitively."""
    for target in VALIDATION_TARGETS:
        if target.matches(name):
            return target
    available = ", ".join(target.name for target in VALIDATION_TARGETS)
    raise ValueError(f"unknown model {name!r}; available: {available}")


def _assert_unique_names(targets: tuple[ValidationTarget, ...]) -> None:
    """Guard against two entries answering to the same name.

    ``find_validation_target`` returns the first match, so a duplicate would
    silently shadow a model rather than fail.

    Keyed on target *identity*, not on the name: a target whose short name is
    also its repo id registers the same key twice and must be allowed, while
    two separate entries sharing a name — the copy-paste mistake this is here
    to catch — must not be, even when the entries are otherwise equal (a frozen
    dataclass compares by value, so ``is`` is the load-bearing part).
    """
    seen: dict[str, ValidationTarget] = {}
    for target in targets:
        for key in (target.name.lower(), target.model_name.lower()):
            other = seen.get(key)
            if other is not None and other is not target:
                raise ValueError(
                    f"validation targets {other.name!r} ({other.model_name}) and "
                    f"{target.name!r} ({target.model_name}) both answer to {key!r}"
                )
            seen[key] = target


_assert_unique_names(VALIDATION_TARGETS)
