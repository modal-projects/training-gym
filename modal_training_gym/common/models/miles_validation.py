"""Model configs supported by the miles CI validation run.

Kept separate from ``validation.py`` (slime): the two frameworks validate
different models, and a model listed here must have a base miles recipe
(``MilesRecipe.get_base_recipe``) rather than a slime one.
"""

from .base import ModelConfig
from .kimi_k2_5 import Kimi_K2_5
from .kimi_k2_6 import Kimi_K2_6

# Every model with a base miles recipe. Runnable by name with
# ``scripts/validate_miles_model_configs.py check --model <name>``.
MILES_MODELS: tuple[tuple[str, type[ModelConfig]], ...] = (
    ("Kimi-K2.5", Kimi_K2_5),
    ("Kimi-K2.6", Kimi_K2_6),
)

# Models the miles validation workflow runs on its own. Deliberately empty:
# the only miles recipes today are Kimi on 16 x 8 H200, far too large to launch
# from every PR, so miles validation stays opt-in (workflow_dispatch or the
# CLI). Add a model here once it is cheap enough to gate PRs on.
MILES_VALIDATABLE_MODELS: tuple[tuple[str, type[ModelConfig]], ...] = ()
