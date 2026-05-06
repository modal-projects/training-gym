import dataclasses as _dc

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.frameworks.slime import build_slime_app
from modal_training_gym.train_recipes.base import BaseTrainRecipe, RecipeType
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


def _merge_recipe(base: SlimeRecipe, overrides: SlimeRecipe) -> SlimeRecipe:
    base_fields = {f.name: getattr(base, f.name) for f in _dc.fields(base)}
    default_recipe = SlimeRecipe()
    for f in _dc.fields(overrides):
        user_val = getattr(overrides, f.name)
        default_val = getattr(default_recipe, f.name)
        if user_val != default_val:
            base_fields[f.name] = user_val
    return SlimeRecipe(**base_fields)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class TrainConfig:
    # ── Composed configs (required) ─────────────────────────────────────────
    dataset: DatasetConfig
    model: ModelConfig
    recipe: BaseTrainRecipe


    # ── Public API ────────────────────────────────────────────────────────────

    def _build_app(self):
        recipe_type = self.recipe.recipe_type
        if recipe_type == RecipeType.SLIME:
            base_recipe = SlimeRecipe.get_base_recipe(self.model)
            if self.recipe is not None:
                combined = _merge_recipe(base_recipe, self.recipe)
            else:
                combined = base_recipe
            return build_slime_app(
                slime=combined,
                model=self.model,
                dataset=self.dataset,
            )
        raise ValueError(f"Unknown recipe type: {recipe_type}")

    def train(self) -> TrainResult:
        """Build the app, run training, and return the TrainResult."""
        import modal

        app = self._build_app()
        with modal.enable_output():
            with app.run():
                result_dict = app.train.remote()
        result = TrainResult(**TrainResult._parse_model_config(result_dict))
        print(f"Training complete: {result.training_run_id}")
        return result
