import pytest

from modal_training_gym.common.metrics import metric_cli_fields, metric_runtime_env
from modal_training_gym.common.wandb import WandbConfig


@pytest.mark.parametrize("environment_key", [None, "secret-from-environment"])
def test_wandb_credentials_use_environment_instead_of_command(
    monkeypatch, environment_key
):
    if environment_key:
        monkeypatch.setenv("WANDB_API_KEY", environment_key)
    else:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
    metric = WandbConfig(project="test", key="secret-from-config")
    fields = metric_cli_fields(metric)
    assert "wandb_key" not in fields
    assert "secret-from-config" not in str(fields)
    env = metric_runtime_env(metric, run_id="run-1")
    assert env["WANDB_API_KEY"] == (environment_key or "secret-from-config")
    assert env["WANDB_RUN_ID"] == "run-1"
    assert metric.key == "secret-from-config"


def test_empty_wandb_key_is_not_forwarded(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    env = metric_runtime_env(WandbConfig(project="test"), run_id="run-1")
    assert "WANDB_API_KEY" not in env
