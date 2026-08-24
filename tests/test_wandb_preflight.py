"""The W&B pre-flight: turn a recurring mid-training wandb CommError into an
early, actionable failure (missing key, or no write access).
"""

import sys
import types

import pytest

from modal_training_gym.common.wandb import WandbConfig, preflight_wandb
from modal_training_gym.frameworks.slime.launcher import (
    _preflight_wandb as _slime_preflight_wandb,
)

_CFG = WandbConfig(project="qwen3-asr-rl", modal_wandb_secret_name="wandb-secret")


def _stub_wandb(monkeypatch, **attrs):
    """Stand in for the lazily-imported ``wandb`` module with a fake exposing *attrs*.

    ``preflight_wandb`` does ``import wandb`` inside the function, so preloading a
    fake into ``sys.modules`` intercepts it — no real library, network, or login.
    """
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(**attrs))


def test_preflight_raises_clear_error_without_key(monkeypatch):
    """No WANDB_API_KEY → a clear error naming the missing var, before any GPU work."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        preflight_wandb(_CFG)


def test_preflight_wraps_access_failure(monkeypatch):
    """Key present but W&B rejects the write → the raw wandb error is
    re-raised as a RuntimeError that names the project."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    def login_without_write_access(**_):
        raise Exception("user does not have models write access")

    _stub_wandb(monkeypatch, login=login_without_write_access)
    with pytest.raises(RuntimeError, match="W&B pre-flight failed.*qwen3-asr-rl"):
        preflight_wandb(_CFG)


def test_preflight_returns_entity(monkeypatch):
    """Successful preflight returns the W&B entity string."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    class _FakeRun:
        entity = "my-team"
        project = "qwen3-asr-rl"
        id = "abc123"

    class _FakeApi:
        def __init__(self, **_):
            pass

        def run(self, path):
            return types.SimpleNamespace(delete=lambda: None)

    _stub_wandb(
        monkeypatch,
        login=lambda **_: None,
        init=lambda **_: _FakeRun(),
        finish=lambda: None,
        Settings=lambda **_: {},
        Api=_FakeApi,
    )
    entity = preflight_wandb(_CFG)
    assert entity == "my-team"


def test_slime_preflight_delegates_to_common(monkeypatch):
    """The slime launcher's _preflight_wandb delegates to the common function."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    class _FakeRun:
        entity = "slime-team"
        project = "qwen3-asr-rl"
        id = "xyz789"

    class _FakeApi:
        def __init__(self, **_):
            pass

        def run(self, path):
            return types.SimpleNamespace(delete=lambda: None)

    _stub_wandb(
        monkeypatch,
        login=lambda **_: None,
        init=lambda **_: _FakeRun(),
        finish=lambda: None,
        Settings=lambda **_: {},
        Api=_FakeApi,
    )
    entity = _slime_preflight_wandb(_CFG)
    assert entity == "slime-team"


def test_wandb_config_uses_the_provider_neutral_recipe_field():
    from modal_training_gym import MetricConfig, Qwen3_4b_Recipe
    from modal_training_gym.common.launcher_helpers import build_app_tags

    with pytest.raises(TypeError, match="abstract"):
        MetricConfig()

    metric = WandbConfig(project="training")
    recipe = Qwen3_4b_Recipe(metrics=metric)

    assert isinstance(metric, MetricConfig)
    assert recipe.metrics is metric
    assert "--use-wandb" in recipe.cli_args()
    assert "label" not in metric.metadata(run_id="run-1")
    tags = build_app_tags(
        framework="slime",
        model=types.SimpleNamespace(model_name="Qwen/Qwen3-4B"),
        recipe_app_tags={},
        metrics=metric,
    )
    assert tags["_modal_metric_project"] == "training"
    assert tags["_modal_wandb_project"] == "training"
