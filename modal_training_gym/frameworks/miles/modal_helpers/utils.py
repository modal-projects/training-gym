"""Miles launcher helpers.

Shared implementations live in :mod:`modal_training_gym.common.launcher_utils`;
this module keeps miles' parametrization and the historical import path.
"""

from modal_training_gym.common.launcher_utils import (
    build_train_cmd as _build_train_cmd,
    get_checkpoint_conversion_policy as _get_checkpoint_conversion_policy,
    is_local_checkpoint_ref as is_local_checkpoint_ref,
    prepare_launch_config as _prepare_launch_config,
    resolve_checkpoint_ref as resolve_checkpoint_ref,
)


def get_checkpoint_conversion_policy(
    miles_cfg, model=None
) -> tuple[int, int, list[str]]:
    return _get_checkpoint_conversion_policy(
        miles_cfg,
        model=model,
        single_rank_mtp=False,
        extended_arch_args=False,
        arch_args_model_script_attr=None,
    )


def prepare_miles_config(miles_cfg, model, tmpdir: str) -> None:
    from modal_training_gym.train_recipes.miles_recipe.recipe import YAML_CONFIG_FIELDS

    _prepare_launch_config(
        miles_cfg, model, tmpdir, yaml_config_fields=YAML_CONFIG_FIELDS
    )


def build_train_cmd(miles_cfg, miles_root: str, model=None, dataset=None) -> str:
    return _build_train_cmd(
        miles_cfg,
        miles_root,
        model=model,
        dataset=dataset,
        model_script_attr="miles_model_script",
    )
