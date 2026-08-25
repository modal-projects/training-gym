from __future__ import annotations

import sys
import types
from dataclasses import fields
from importlib.util import find_spec
from typing import Any

from modal_training_gym.common.metrics import apply_metric_image
from modal_training_gym.common.trackio import TrackioConfig, install_wandb_shim


def test_trackio_config_is_provider_specific_without_provider_or_label_fields():
    from modal_training_gym import TrackioConfig as PublicTrackioConfig

    config = TrackioConfig(
        project="rl",
        group="baseline",
        space_id="modal-labs/training-metrics",
        bucket_id="modal-labs/training-metrics",
    )

    assert config.provider == "trackio"
    assert PublicTrackioConfig is TrackioConfig
    assert {field.name for field in fields(config)}.isdisjoint({"provider", "label"})
    assert config.runtime_env(run_id="run-a2") == {
        "TRAINING_GYM_METRIC_PROVIDER": "trackio",
        "TRAINING_GYM_TRACKIO_RUN_NAME": "run-a2",
        "TRACKIO_SPACE_ID": "modal-labs/training-metrics",
        "TRACKIO_BUCKET_ID": "modal-labs/training-metrics",
    }
    assert config.url() == (
        "https://huggingface.co/spaces/modal-labs/training-metrics?project=rl"
    )
    assert config.metadata(run_id="run-a2")["url"] == (
        "https://huggingface.co/spaces/modal-labs/training-metrics"
        "?project=rl&runs=run-a2"
    )
    assert "label" not in config.metadata(run_id="run-a2")


def test_trackio_dashboard_urls_do_not_expose_credentials():
    config = TrackioConfig(
        server_url="https://user:password@metrics.example.com:8443/path"
        "?write_token=secret#fragment"
    )
    assert config.url(run_id="run-a2") == (
        "https://metrics.example.com:8443/path?project=training-gym&runs=run-a2"
    )

    config.project = "rl"
    config.dashboard_url = (
        "https://metrics.example.com/view?project=rl&write_token=secret"
    )
    assert config.url(run_id="run-a2") == (
        "https://metrics.example.com/view?project=rl&runs=run-a2"
    )


class _FakeImage:
    def __init__(self) -> None:
        self.packages: list[str] = []
        self.commands: list[str] = []

    def uv_pip_install(self, package: str) -> _FakeImage:
        self.packages.append(package)
        return self

    def run_commands(self, command: str) -> _FakeImage:
        self.commands.append(command)
        return self


def test_trackio_image_installs_trackio_and_the_conditional_wandb_adapter():
    image = _FakeImage()
    result = apply_metric_image(image, TrackioConfig())

    assert result is image
    assert image.packages == ["trackio==0.34.0"]
    assert len(image.commands) == 1
    assert "_training_gym_trackio.pth" in image.commands[0]
    assert "TRAINING_GYM_METRIC_PROVIDER" in image.commands[0]


def test_trackio_wandb_adapter_covers_the_framework_surface(monkeypatch):
    calls: dict[str, Any] = {}

    class _FakeRun:
        name = "fallback-name"
        config = {"learning_rate": 1e-5}

    fake_trackio = types.ModuleType("trackio")

    def fake_init(**kwargs: Any) -> _FakeRun:
        calls["init"] = kwargs
        return _FakeRun()

    def fake_finish(*args: Any, **kwargs: Any) -> None:
        calls["finish"] = (args, kwargs)

    fake_trackio.init = fake_init
    fake_trackio.log = lambda data, step=None: calls.setdefault("logs", []).append(
        (data, step)
    )
    fake_trackio.finish = fake_finish
    fake_trackio.save = lambda path: path
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    for module_name in (
        "wandb",
        "wandb.util",
        "wandb.sdk",
        "wandb.sdk.lib",
        "wandb.sdk.lib.runid",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setenv("TRAINING_GYM_TRACKIO_RUN_NAME", "training-run-a2")

    install_wandb_shim()

    import wandb
    from wandb.sdk.lib.runid import generate_id

    assert find_spec("wandb") is wandb.__spec__

    settings = wandb.Settings(mode="shared")
    run = wandb.init(
        project="rl",
        entity="ignored",
        group="baseline",
        name="framework-name",
        config={"learning_rate": 1e-5},
        settings=settings,
        reinit=True,
        dir="/tmp/wandb",
    )

    assert calls["init"] == {
        "project": "rl",
        "name": "training-run-a2",
        "group": "baseline",
        "config": {"learning_rate": 1e-5},
        "resume": "allow",
        "embed": False,
    }
    assert run.id == "training-run-a2"
    assert wandb.run.id == "training-run-a2"
    assert wandb.config == {"learning_rate": 1e-5}
    assert calls["logs"] == [({}, -1)]
    assert len(generate_id()) == 8
    assert wandb.define_metric("train/*", step_metric="train/step") is None

    wandb.log({"loss": 0.5})
    wandb.finish()
    assert calls["logs"] == [({}, -1), ({"loss": 0.5}, None)]
    assert wandb.run is None
