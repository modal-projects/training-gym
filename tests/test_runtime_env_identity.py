"""The run identity must survive a recipe's own environment dict."""

from __future__ import annotations

from pathlib import Path


def _launcher_source() -> str:
    import modal_training_gym.frameworks.slime.launcher as launcher_module

    return Path(launcher_module.__file__).read_text()


def test_recipe_environment_cannot_override_the_run_identity() -> None:
    """`**slime.environment` is splatted into runtime_env, so ordering decides.

    Later keys win in a dict literal, and slime.environment is arbitrary
    user config. Splatted after the gym's own variables it can silently
    reassign the run id that keys the metric run, the run record, and the
    framework status callbacks -- with no error, since any string is valid.
    """
    source = _launcher_source()
    runtime_env = source[source.index("runtime_env = {") :]

    assert runtime_env.index("**slime.environment") < runtime_env.index(
        '"TRAINING_GYM_TRAINING_RUN_ID": training_run_id'
    )
