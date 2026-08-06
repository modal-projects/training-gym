"""Invariants of the cross-framework model validation registry.

Before the frameworks shared a harness, a model too large to gate PRs on was
kept out of CI by living in a different module. Now it is kept out by
``gates_prs=False`` alone, so that flag needs a test: a regression here spends
real GPU hours on every pull request.
"""

from __future__ import annotations

import pytest

from modal_training_gym.common.models.validation import (
    VALIDATION_TARGETS,
    ValidationFramework,
    find_validation_target,
    validation_targets,
)
from scripts.diff_impact import (
    FRAMEWORK_VALIDATION_HARNESS_PATHS,
    REPO_ROOT,
    SHARED_VALIDATION_HARNESS_PATHS,
    affected_models,
)
from scripts.validation_backends import RecipeOverrides, backend_for


def _diff_touching(*repo_relative_paths: str) -> str:
    """A minimal unified diff naming the given paths."""
    return "\n".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}"
        for path in repo_relative_paths
    )


NON_GATING_TARGETS = [t for t in VALIDATION_TARGETS if not t.gates_prs]


def test_registry_has_both_frameworks_represented():
    frameworks = {target.framework for target in VALIDATION_TARGETS}
    assert frameworks == set(ValidationFramework)


@pytest.mark.parametrize("target", VALIDATION_TARGETS, ids=lambda t: t.name)
def test_target_resolves_by_short_name_and_repo_id(target):
    assert find_validation_target(target.name) is target
    assert find_validation_target(target.model_name) is target
    assert find_validation_target(target.name.upper()) is target


@pytest.mark.parametrize("target", VALIDATION_TARGETS, ids=lambda t: t.name)
def test_every_target_builds_a_recipe_on_its_declared_framework(target):
    """The declared framework must actually have a base recipe for the model.

    This is the check that a wrong ``framework`` value would otherwise only
    fail on a GPU, minutes into a run.
    """
    backend = backend_for(target.framework)
    assert backend.framework == target.framework

    recipe = backend.build_recipe(
        target, target.model_config(), step_count=1, overrides=RecipeOverrides()
    )
    assert recipe is not None
    assert recipe.num_rollout == 1


def test_unknown_model_names_are_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        find_validation_target("not-a-real-model")


def test_unsupported_override_is_rejected_rather_than_ignored():
    """A framework-only flag must error on the wrong framework, not no-op."""
    slime_target = validation_targets(ValidationFramework.SLIME, gating_only=True)[0]
    backend = backend_for(slime_target.framework)

    with pytest.raises(Exception, match="--docker-image"):
        backend.build_recipe(
            slime_target,
            slime_target.model_config(),
            step_count=1,
            overrides=RecipeOverrides(docker_image="example/image:tag"),
        )


def test_recipe_overrides_reports_only_explicitly_set_fields():
    assert RecipeOverrides().set_fields() == ()
    assert RecipeOverrides(save_interval=3).set_fields() == ("save_interval",)
    assert set(RecipeOverrides(docker_image="x", non_colocated=True).set_fields()) == {
        "docker_image",
        "non_colocated",
    }


def test_list_excludes_non_gating_models_by_default():
    from scripts.validate_model_configs import available_model_names

    gating = set(available_model_names())
    everything = set(available_model_names(include_non_gating=True))

    assert gating < everything
    for target in NON_GATING_TARGETS:
        assert target.name not in gating
        assert target.name in everything


@pytest.mark.skipif(not NON_GATING_TARGETS, reason="every target gates PRs")
def test_no_diff_can_select_a_non_gating_model():
    """The load-bearing invariant: PRs never launch a gates_prs=False model.

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
    for target in NON_GATING_TARGETS:
        assert target.name not in selected


def test_framework_harness_change_only_revalidates_that_framework():
    """A miles-only change must not re-run the slime set, and vice versa."""
    slime_models = {
        t.name for t in validation_targets(ValidationFramework.SLIME, gating_only=True)
    }
    miles_models = {
        t.name for t in validation_targets(ValidationFramework.MILES, gating_only=True)
    }

    slime_paths = FRAMEWORK_VALIDATION_HARNESS_PATHS["slime"]
    diff = _diff_touching(*(str(p.relative_to(REPO_ROOT)) for p in slime_paths))
    selected = set(affected_models(diff))

    assert selected == slime_models
    assert not selected & (miles_models - slime_models)


def test_shared_harness_change_revalidates_every_gating_model():
    gating = {t.name for t in validation_targets(gating_only=True)}
    diff = _diff_touching("scripts/validate_model_configs.py")

    assert set(affected_models(diff)) == gating
