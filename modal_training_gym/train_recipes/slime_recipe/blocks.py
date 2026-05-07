from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


class SlimeRecipeBlock(Protocol):
    def apply(self, recipe: "SlimeRecipe") -> "SlimeRecipe":
        ...


@dataclass
class MultiTurn:
    generate_function: Callable
    reward_function: Callable | None = None
    max_turns: int | None = None
    log_multi_turn: bool | None = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def apply(self, recipe: "SlimeRecipe") -> "SlimeRecipe":
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")

        custom_config = dict(recipe.custom_config_path or {})
        if self.max_turns is not None:
            custom_config["max_turns"] = int(self.max_turns)
        if self.log_multi_turn is not None:
            custom_config["log_multi_turn"] = bool(self.log_multi_turn)
        custom_config.update(dict(self.config_overrides))

        overrides: dict[str, Any] = {
            "custom_generate_function": self.generate_function,
            "custom_config_path": custom_config,
        }
        if self.reward_function is not None:
            overrides["custom_rm_function"] = self.reward_function
        return recipe.with_overrides(**overrides)
