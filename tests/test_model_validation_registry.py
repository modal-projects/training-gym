"""Invariants of the cross-framework model validation registry.

Before the frameworks shared a harness, a model too large to gate PRs on was
kept out of CI by living in a different module. Now it is kept out by
``ci_enabled=False`` alone, so that flag needs a test: a regression here spends
real GPU hours on every pull request.
"""

from __future__ import annotations

import pytest

from modal_training_gym.common.models.validation import (
    VALIDATION_CONFIGS,
    Framework,
    _ValidationConfig,
)
from scripts.diff_impact import (
    FRAMEWORK_VALIDATION_HARNESS_PATHS,
    REPO_ROOT,
    SHARED_VALIDATION_HARNESS_PATHS,
    affected_models,
)
from scripts.validation_backends import build_recipe_and_dataset


def _diff_touching(*repo_relative_paths: str) -> str:
    """A minimal unified diff naming the given paths."""
    return "\n".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}"
        for path in repo_relative_paths
    )


ALL_CONFIGS = _ValidationConfig.select(ci_only=False)
NON_GATING_CONFIGS = [c for c in ALL_CONFIGS if not c.ci_enabled]


def test_registry_has_both_frameworks_represented():
    frameworks = {config.framework for config in VALIDATION_CONFIGS}
    assert frameworks == set(Framework)


def test_registry_names_are_unique():
    """A copy-pasted entry must fail a test, not silently shadow a model.

    ``_ValidationConfig.find`` returns the first case-insensitive match, so two
    entries answering to one name would hide whichever came second.
    """
    names = [config.name.lower() for config in ALL_CONFIGS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_config_resolves_by_name_case_insensitively(config):
    assert _ValidationConfig.find(config.name) is config
    assert _ValidationConfig.find(config.name.upper()) is config
    assert _ValidationConfig.find(f"  {config.name.lower()} ") is config


def test_unknown_model_names_are_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        _ValidationConfig.find("not-a-real-model")


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_every_config_builds_a_recipe_on_its_declared_framework(config):
    """The declared framework must actually have a base recipe for the model.

    This is the check that a wrong ``framework`` value would otherwise only
    fail on a GPU, minutes into a run.
    """
    recipe, dataset = build_recipe_and_dataset(
        config.framework, config.model_config(), step_count=1
    )
    assert recipe is not None
    assert dataset is not None


def test_docker_image_override_is_rejected_on_a_recipe_without_one():
    """A flag naming a field the recipe lacks must error, not no-op.

    Recipes don't validate assignment, so a bare ``setattr`` would attach a
    dead attribute and ``--docker-image`` on slime would silently do nothing.
    """
    from scripts.validate_model_configs import _set_recipe_field

    slime_config = _ValidationConfig.select(Framework.SLIME)[0]
    recipe, _ = build_recipe_and_dataset(
        slime_config.framework, slime_config.model_config(), step_count=1
    )

    with pytest.raises(Exception, match="--docker-image"):
        _set_recipe_field(recipe, "docker_image", "example/image:tag")


def test_list_excludes_non_gating_models_by_default():
    from scripts.validate_model_configs import available_model_names

    gating = set(available_model_names())
    everything = set(available_model_names(include_non_gating=True))

    assert gating < everything
    for config in NON_GATING_CONFIGS:
        assert config.name not in gating
        assert config.name in everything


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_validation_dataset_unpickles_without_the_scripts_directory(config, tmp_path):
    """Every backend dataset must survive the trip into a training container.

    Backend modules live under ``scripts/``, which is absent from the training
    image, so a dataset pickled by reference crashes remotely during data
    preparation. Unpickling in a subprocess whose ``sys.path`` has no
    ``scripts/`` entry reproduces that container exactly.
    """
    import base64
    import subprocess
    import sys
    import textwrap

    import cloudpickle

    from scripts.validate_model_configs import _ship_dataset_definition

    _, dataset = build_recipe_and_dataset(config.framework, config.model_config(), 1)

    _ship_dataset_definition(dataset)
    payload = cloudpickle.dumps(dataset)

    # Blocking the import is a truer stand-in for the image than trimming
    # sys.path: modal_training_gym is installed from this same tree, so the
    # entry that makes the backends importable is the one it needs too.
    probe = textwrap.dedent("""
        import base64, pickle, sys

        class Blocked:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("scripts", "validation_backends"):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocked())
        obj = pickle.loads(base64.b64decode(sys.argv[1]))
        print(type(obj).__name__)
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe, base64.b64encode(payload).decode()],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"{config.name} dataset does not survive into a training container:\n"
        f"{result.stderr}"
    )
    assert result.stdout.strip() == type(dataset).__name__


@pytest.mark.skipif(not NON_GATING_CONFIGS, reason="every model is CI-enabled")
def test_no_diff_can_select_a_non_gating_model():
    """The load-bearing invariant: PRs never launch a ci_enabled=False model.

    The broadest possible trigger — every harness path at once, which forces a
    full re-validation — still must not select one.
    """
    every_harness_path = SHARED_VALIDATION_HARNESS_PATHS.union(
        *FRAMEWORK_VALIDATION_HARNESS_PATHS.values()
    )
    diff = _diff_touching(
        *(str(path.relative_to(REPO_ROOT)) for path in every_harness_path)
    )

    selected = set(affected_models(diff))
    assert selected, "a full-harness diff should still select the PR-gating models"
    for config in NON_GATING_CONFIGS:
        assert config.name not in selected


def test_framework_harness_change_only_revalidates_that_framework():
    """A miles-only change must not re-run the slime set, and vice versa."""
    slime_models = {c.name for c in _ValidationConfig.select(Framework.SLIME)}
    miles_models = {c.name for c in _ValidationConfig.select(Framework.MILES)}

    slime_paths = FRAMEWORK_VALIDATION_HARNESS_PATHS["slime"]
    diff = _diff_touching(*(str(p.relative_to(REPO_ROOT)) for p in slime_paths))
    selected = set(affected_models(diff))

    assert selected == slime_models
    assert not selected & (miles_models - slime_models)


def test_shared_harness_change_revalidates_every_gating_model():
    gating = {c.name for c in _ValidationConfig.select()}
    diff = _diff_touching("scripts/validate_model_configs.py")

    assert set(affected_models(diff)) == gating
