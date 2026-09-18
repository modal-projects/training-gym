"""Experiment metric tracker configuration shared across frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricConfig(ABC):
    """Defines metric tracker metadata, environment variables, and links.

    Attributes:
        project:
            Metric project name.
        group:
            Group tag for related runs.
        exp_name:
            Run display name.
        disable_random_suffix:
            Preserve the configured run name.
        mirror_to_dashboard:
            Also send every scalar the framework logs to the Training Gym
            dashboard's Metrics tab. The external tracker keeps receiving
            everything; the dashboard only gets numbers.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True
    mirror_to_dashboard: bool = True

    @property
    @abstractmethod
    def provider(self) -> str:
        """Machine-readable metric provider identifier."""

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        return {"TRAINING_GYM_METRIC_PROVIDER": self.provider}

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        return None

    def metadata(self, *, entity: str = "", run_id: str = "") -> dict[str, str]:
        metadata = {
            "provider": self.provider,
            "project": self.project,
            "group": self.group,
            "entity": entity,
            "run_id": run_id,
        }
        if url := self.url(entity=entity, run_id=run_id):
            metadata["url"] = url
        return metadata


def metric_cli_fields(metric: MetricConfig) -> dict[str, Any]:
    fields = {
        "use_wandb": True,
        "wandb_project": metric.project,
        "wandb_group": metric.group,
        "disable_wandb_random_suffix": metric.disable_random_suffix,
    }
    return fields


def metric_runtime_env(
    metric: MetricConfig | None, *, run_id: str, entity: str = ""
) -> dict[str, str]:
    if metric is None:
        return {}
    from modal_training_gym.common.metric_mirror import MIRROR_MODE_ENV, mirror_mode

    env = metric.runtime_env(run_id=run_id, entity=entity)
    if mode := mirror_mode(metric):
        env[MIRROR_MODE_ENV] = mode
    return env


def metric_metadata(
    metric: MetricConfig | None, *, entity: str = "", run_id: str = ""
) -> dict[str, str]:
    return {} if metric is None else metric.metadata(entity=entity, run_id=run_id)


def metric_secrets(metric: MetricConfig) -> list[Any]:
    if metric.provider == "trackio":
        from modal_training_gym.common.trackio import trackio_secrets

        return trackio_secrets(metric)
    if metric.provider == "wandb":
        from modal import Secret

        return [Secret.from_name(getattr(metric, "modal_wandb_secret_name"))]
    return []


def apply_metric_image(image: Any, metric: MetricConfig | None) -> Any:
    """Install the provider package and the ``.pth`` bootstraps in one layer.

    Each ``.pth`` line is inert unless its environment variable is set, so the
    image can be shared by runs with different metric configs.
    """
    if metric is None:
        return image
    from modal_training_gym.common.metric_mirror import (
        mirror_pth_files,
        pth_install_command,
    )

    pth_files: dict[str, str] = {}
    if metric.provider == "trackio":
        from modal_training_gym.common.trackio import (
            install_trackio_package,
            trackio_pth_files,
        )

        image = install_trackio_package(image, metric)
        pth_files.update(trackio_pth_files())
    pth_files.update(mirror_pth_files())
    return image.run_commands(pth_install_command(pth_files))


def preflight_metric(metric: MetricConfig | None) -> str:
    if metric is not None and metric.provider == "wandb":
        from modal_training_gym.common.wandb import preflight_wandb

        return preflight_wandb(metric)
    if metric is not None and metric.provider == "trackio":
        from modal_training_gym.common.trackio import preflight_trackio

        return preflight_trackio(metric)
    return ""
