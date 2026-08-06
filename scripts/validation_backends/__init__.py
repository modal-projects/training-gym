"""Per-framework halves of the model validation harness.

``scripts/validate_model_configs.py`` owns everything that is the same for
every framework: the CLI, the result dataclass, the markdown summary and the
PR comment. A backend owns the three things that genuinely differ per
framework — which recipe trains the model, which dataset it trains on, and
which CLI overrides mean anything — so adding a framework is one new module
here plus registry entries in ``common/models/validation.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.validation import (
    ValidationFramework,
    ValidationTarget,
)

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfig
    from modal_training_gym.train_recipes.base import BaseTrainRecipe


@dataclass(frozen=True)
class RecipeOverrides:
    """``check`` flags that adjust the recipe before the run.

    Declared once for every framework so the CLI stays uniform; a backend
    lists the ones it understands in ``supported_overrides`` and the rest are
    rejected with an error rather than silently ignored.
    """

    eval_interval: int | None = None
    save_interval: int | None = None
    docker_image: str | None = None
    non_colocated: bool = False

    def set_fields(self) -> tuple[str, ...]:
        """Names of the overrides the caller actually passed."""
        return tuple(
            field.name
            for field in fields(self)
            if getattr(self, field.name) != field.default
        )


def _flag(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


class ValidationBackend(ABC):
    """How one framework turns a registry entry into a training run."""

    framework: ClassVar[ValidationFramework]

    supported_overrides: ClassVar[frozenset[str]] = frozenset(
        {"eval_interval", "save_interval"}
    )
    """``RecipeOverrides`` fields this framework can honor."""

    def build_recipe(
        self,
        target: ValidationTarget,
        model_config: "ModelConfig",
        step_count: int,
        overrides: RecipeOverrides,
    ) -> "BaseTrainRecipe":
        """The model's base recipe, tuned for a short validation run."""
        unsupported = [
            _flag(name)
            for name in overrides.set_fields()
            if name not in self.supported_overrides
        ]
        if unsupported:
            raise TrainingGymConfigError(
                f"{', '.join(unsupported)} not supported for {self.framework} "
                f"validation ({target.name} trains on {self.framework})"
            )
        recipe = self._build_recipe(target, model_config, step_count, overrides)
        recipe.num_rollout = step_count
        if overrides.eval_interval is not None:
            recipe.eval_interval = overrides.eval_interval
        if overrides.save_interval is not None:
            recipe.save_interval = overrides.save_interval
        return recipe

    @abstractmethod
    def _build_recipe(
        self,
        target: ValidationTarget,
        model_config: "ModelConfig",
        step_count: int,
        overrides: RecipeOverrides,
    ) -> "BaseTrainRecipe":
        """Framework-specific recipe lookup and tuning."""

    @abstractmethod
    def pick_dataset(
        self,
        target: ValidationTarget,
        model_config: "ModelConfig",
        recipe: "BaseTrainRecipe",
        step_count: int,
    ) -> "DatasetConfig":
        """The dataset this model validates against."""

    def docker_image(self, recipe: "BaseTrainRecipe") -> str | None:
        """Image the run will use, when the framework pins one explicitly."""
        return None


def backend_for(framework: ValidationFramework) -> ValidationBackend:
    """The backend that trains models on ``framework``.

    Imported lazily so that validating a model on one framework never imports
    the other framework's recipes or datasets — a broken miles backend must not
    be able to take down the slime validation that gates PRs.
    """
    if framework == ValidationFramework.SLIME:
        from .slime import SlimeValidationBackend

        return SlimeValidationBackend()
    if framework == ValidationFramework.MILES:
        from .miles import MilesValidationBackend

        return MilesValidationBackend()
    raise TrainingGymConfigError(f"no validation backend for framework {framework!r}")


__all__ = [
    "RecipeOverrides",
    "ValidationBackend",
    "backend_for",
]
