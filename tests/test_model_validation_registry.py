"""Invariants of the cross-framework model validation registry."""

from __future__ import annotations

import json

import pytest

from modal_training_gym.common.models.qwen3_0_6b import Qwen3_0_6B
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


ALL_CONFIGS = _ValidationConfig.select(pr_only=False)
DISPATCH_ONLY_CONFIGS = [c for c in ALL_CONFIGS if not c.run_on_pr]


def test_registry_has_both_frameworks_represented():
    frameworks = {config.framework for config in VALIDATION_CONFIGS}
    assert frameworks == set(Framework)


def test_registry_uses_the_packages_one_framework_enum():
    """A second same-valued enum would fail every dispatch branch.

    ``build_recipe_and_dataset`` and ``diff_impact._base_recipe_for`` both
    dispatch on identity, so a caller holding ``common.framework.Framework``
    would fall through to "no validation backend" if the registry declared its
    own copy — equal by value, a different class.
    """
    from modal_training_gym.common.framework import Framework as CanonicalFramework

    assert Framework is CanonicalFramework
    recipe, dataset = build_recipe_and_dataset(
        CanonicalFramework.SLIME, Qwen3_0_6B(), step_count=1
    )
    assert recipe is not None and dataset is not None


def test_registry_names_are_unique():
    """Every model/framework pair needs an unambiguous dispatch name."""
    seen: dict[str, _ValidationConfig] = {}
    for config in ALL_CONFIGS:
        key = config.name.lower()
        other = seen.setdefault(key, config)
        assert other is config, f"duplicate validation name: {config.name!r}"


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_config_resolves_by_name_case_insensitively(config):
    assert _ValidationConfig.find(config.name) is config
    assert _ValidationConfig.find(config.name.upper()) is config
    assert _ValidationConfig.find(f"  {config.name.lower()} ") is config


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_config_resolves_by_hf_repo_id(config):
    """``check -m Qwen/Qwen3-4B`` must keep working, not just the short name."""
    matches = [c for c in ALL_CONFIGS if c.model_name == config.model_name]
    base = [c for c in matches if not c.projector]
    if len(matches) == 1:
        assert _ValidationConfig.find(config.model_name) is config
        assert _ValidationConfig.find(config.model_name.upper()) is config
    elif len(base) == 1:
        # A projector entry is a variant of the base entry, not a peer, so the
        # repo id keeps resolving to the base training run.
        assert _ValidationConfig.find(config.model_name) is base[0]
    else:
        with pytest.raises(ValueError, match="ambiguous model"):
            _ValidationConfig.find(config.model_name)


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
        config.framework,
        config.model_config(),
        step_count=1,
        projector=config.projector,
    )
    assert recipe is not None
    assert dataset is not None


def test_list_shows_every_model_by_default_and_narrows_with_pr_only():
    """A dev listing models should see the dispatch-only ones without a flag.

    The narrowing is the opt-in, so ``--pr-only`` is what a matrix asks for.
    """
    from scripts.validate_model_configs import available_model_names

    everything = set(available_model_names())
    pr_set = set(available_model_names(pr_only=True))

    assert pr_set <= everything
    if DISPATCH_ONLY_CONFIGS:
        assert pr_set < everything
    for config in DISPATCH_ONLY_CONFIGS:
        assert config.name in everything
        assert config.name not in pr_set


def test_blank_dispatch_asks_for_the_pr_only_set():
    """The workflow's blank-models branch must narrow explicitly.

    ``list`` prints the whole registry, so a blank dispatch must retain the
    explicit narrowing even when no dispatch-only models are registered.
    """
    import re

    workflow = (REPO_ROOT / ".github/workflows/validate-models.yml").read_text()
    unnarrowed = re.findall(
        r"validate_model_configs\.py list(?! --names-only --pr-only)", workflow
    )

    assert "validate_model_configs.py list --names-only --pr-only" in workflow
    assert not unnarrowed, "a `list` in the workflow is missing matrix-only flags"


def test_baselines_are_fetched_only_for_models_that_ran(tmp_path):
    """Baseline lookup follows the results directory, not the registry.

    Widening ``available_model_names`` to the whole registry would otherwise
    have the PR comment job scan artifacts for models that did not run.
    """
    from scripts.download_perf_baseline import models_with_results

    (tmp_path / "validate-result-Qwen3-4B.json").write_text(
        json.dumps({"base_model_name": "Qwen3-4B", "succeeded": True})
    )

    assert models_with_results(tmp_path) == ["Qwen3-4B"]
    assert models_with_results(tmp_path / "missing") == []


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_validation_dataset_unpickles_without_the_scripts_directory(config, tmp_path):
    """Every backend dataset must survive the trip into a training container.

    Backend modules live under ``scripts/``, which is absent from the training
    image, so a dataset pickled by reference crashes remotely during data
    preparation. Unpickling in a subprocess whose ``sys.path`` has no
    ``scripts/`` entry reproduces that container exactly.

    Pickled with ``modal._serialization``, which is what actually ships the
    dataset: it uses Modal's vendored cloudpickle, whose by-value registry is
    separate from the installed cloudpickle's.
    """
    import base64
    import subprocess
    import sys
    import textwrap

    from modal._serialization import serialize

    from scripts.validate_model_configs import _ship_dataset_definition

    _, dataset = build_recipe_and_dataset(
        config.framework, config.model_config(), 1, projector=config.projector
    )

    _ship_dataset_definition(dataset)
    payload = serialize(dataset)

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


@pytest.mark.skipif(not DISPATCH_ONLY_CONFIGS, reason="every model runs on PRs")
def test_no_diff_can_select_a_dispatch_only_model():
    """The load-bearing invariant: PRs never launch a run_on_pr=False model.

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
    assert selected, "a full-harness diff should still select the PR-matrix set"
    for config in DISPATCH_ONLY_CONFIGS:
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


def test_framework_change_does_not_narrow_shared_harness_impact():
    gating = {c.name for c in _ValidationConfig.select()}
    miles_launcher = next(iter(FRAMEWORK_VALIDATION_HARNESS_PATHS["miles"]))
    diff = _diff_touching(
        "scripts/validate_model_configs.py",
        str(miles_launcher.relative_to(REPO_ROOT)),
    )

    assert set(affected_models(diff)) == gating


def test_framework_change_preserves_shared_class_impact():
    gating = {c.name for c in _ValidationConfig.select()}
    miles_launcher = next(iter(FRAMEWORK_VALIDATION_HARNESS_PATHS["miles"]))
    diff = _diff_touching(
        "modal_training_gym/common/models/base.py",
        str(miles_launcher.relative_to(REPO_ROOT)),
    )

    assert set(affected_models(diff)) == gating
