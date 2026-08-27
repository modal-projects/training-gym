"""Experiment metric tracker configuration shared across frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MetricConfig(ABC):
    """Base class for metric tracker configurations.

    ## Fields

    project : str
        Metric project name. Default ``""``.
    group : str
        Group tag for related runs. Default ``""``.
    exp_name : str
        Run display name. Default ``""``.
    disable_random_suffix : bool
        Whether the tracker should preserve the configured run name. Default
        ``True``.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True

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
    if metric.provider == "wandb":
        fields["wandb_key"] = getattr(metric, "key", "")
    return fields


def metric_runtime_env(
    metric: MetricConfig | None, *, run_id: str, entity: str = ""
) -> dict[str, str]:
    return {} if metric is None else metric.runtime_env(run_id=run_id, entity=entity)


def metric_metadata(
    metric: MetricConfig | None, *, entity: str = "", run_id: str = ""
) -> dict[str, str]:
    return {} if metric is None else metric.metadata(entity=entity, run_id=run_id)


def metric_secrets(metric: MetricConfig) -> list[Any]:
    if metric.provider != "wandb":
        return []
    from modal import Secret

    return [Secret.from_name(getattr(metric, "modal_wandb_secret_name"))]


def preflight_metric(metric: MetricConfig | None) -> str:
    if metric is not None and metric.provider == "wandb":
        from modal_training_gym.common.wandb import preflight_wandb

        return preflight_wandb(metric)
    return ""
