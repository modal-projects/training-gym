from __future__ import annotations

import sys
import types
from dataclasses import fields
from importlib.util import find_spec
from typing import Any

from modal_training_gym.common.metrics import apply_metric_image
from modal_training_gym.common.trackio import (
    TrackioConfig,
    install_wandb_shim,
)


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


def test_deploy_to_modal_returns_a_self_hosted_config(monkeypatch):
    calls = {}

    def fake_deploy(**kwargs):
        calls.update(kwargs)
        return "https://example--training-gym-trackio.modal.run"

    monkeypatch.setattr(
        "modal_training_gym.common.trackio._deploy_modal_dashboard", fake_deploy
    )

    config = TrackioConfig.deploy_to_modal(
        project="rl",
        group="baseline",
        app_name="my-trackio",
    )

    assert calls == {
        "app_name": "my-trackio",
        "volume_name": "my-trackio-data",
        "modal_secret_name": "_my-trackio-write-token",
    }
    assert config == TrackioConfig(
        project="rl",
        group="baseline",
        server_url="https://example--training-gym-trackio.modal.run",
        dashboard_url="https://example--training-gym-trackio.modal.run",
        modal_secret_name="_my-trackio-write-token",
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

    # This is the Trackio 0.34.0 keyword surface used by the adapter.
    def fake_init(
        project: str,
        name: str | None = None,
        group: str | None = None,
        space_id: str | None = None,
        server_url: str | None = None,
        bucket_id: str | None = None,
        config: dict[str, Any] | None = None,
        resume: str = "never",
        embed: bool = True,
    ) -> _FakeRun:
        calls["init"] = {
            "project": project,
            "name": name,
            "group": group,
            "space_id": space_id,
            "server_url": server_url,
            "bucket_id": bucket_id,
            "config": config,
            "resume": resume,
            "embed": embed,
        }
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
    monkeypatch.setenv("TRACKIO_SPACE_ID", "modal-labs/training-metrics")
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://metrics.example.com")
    monkeypatch.setenv("TRACKIO_BUCKET_ID", "modal-labs/training-metrics")

    install_wandb_shim()

    import wandb
    from wandb.sdk.lib.runid import generate_id

    assert find_spec("wandb") is wandb.__spec__

    settings = wandb.Settings(mode="shared")
    run = wandb.init(
        project="rl",
        entity="ignored",
        id="framework-id",
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
        "space_id": "modal-labs/training-metrics",
        "server_url": "https://metrics.example.com",
        "bucket_id": "modal-labs/training-metrics",
    }
    assert run.id == "training-run-a2"
    assert wandb.run.id == "training-run-a2"
    assert wandb.config == {"learning_rate": 1e-5}
    assert calls["logs"] == [({}, -1)]
    assert len(generate_id()) == 8
    assert wandb.define_metric("train/*", step_metric="train/step") is None

    wandb.log({"loss": 0.5}, commit=False, sync=True)
    assert wandb.save("metrics.json", base_path="/tmp", policy="now") == (
        "metrics.json"
    )
    wandb.finish(exit_code=0, quiet=True)
    assert calls["logs"] == [({}, -1), ({"loss": 0.5}, None)]
    assert calls["finish"] == ((), {})
    assert wandb.run is None
