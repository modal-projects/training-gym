"""Experiment-tracker configuration shared across training frameworks."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import quote, urlsplit, urlunsplit

from modal_training_gym.common.launcher_utils import redact_url_credentials


__all__ = [
    "MetricConfig",
    "TrackioConfig",
    "WandbConfig",
    "apply_metric_image_overrides",
    "inline_metric_secrets",
    "metric_cli_fields",
    "metric_runtime_env",
    "metrics_metadata",
    "named_metric_secrets",
    "preflight_metrics",
    "preflight_wandb",
]


@dataclass
class MetricConfig:
    """Base class for training metric tracker configuration.

    Subclasses describe a concrete tracker (``WandbConfig``, ``TrackioConfig``)
    and carry the actual fields. Framework recipes still translate these into
    their native flags; today Slime and Miles expose W&B-compatible logging
    flags, so non-W&B providers adapt through that path.
    """

    provider: ClassVar[str] = "metrics"
    label: ClassVar[str] = "Metrics"

    @property
    def project(self) -> str:
        return ""

    @property
    def group(self) -> str:
        return ""

    @property
    def exp_name(self) -> str:
        return ""

    @property
    def disable_random_suffix(self) -> bool:
        return True

    @property
    def entity(self) -> str:
        return ""

    @property
    def key(self) -> str:
        return ""

    @key.setter
    def key(self, value: str) -> None:
        _ = value

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        _ = run_id, entity
        return {"TRAINING_GYM_METRICS_PROVIDER": self.provider}

    def inline_secret_env(self) -> dict[str, str]:
        """Credential env vars delivered via an inline Modal Secret.

        Kept out of the Ray ``runtime_env``, which Ray persists in the job
        record and serves from the (tunnel-forwarded) dashboard API.
        """
        return {}

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        _ = entity, run_id
        return None

    def to_metadata(self, *, entity: str = "", run_id: str = "") -> dict[str, str]:
        data = {
            "provider": self.provider,
            "label": self.label,
            "project": self.project,
            "group": self.group,
            "entity": entity,
            "run_id": run_id,
        }
        if url := self.url(entity=entity, run_id=run_id):
            data["url"] = url
        return data


@dataclass
class WandbConfig(MetricConfig):
    """Weights & Biases logging configuration shared across all frameworks.

    ## Fields

    project : str
        W&B project name. Default ``""``.
    entity : str
        W&B entity/team slug. Optional; when omitted, preflight resolves the
        default entity for the configured API key.
    group : str
        W&B group tag for organizing related runs. Default ``""``.
    exp_name : str
        W&B run display name. Default ``""``.
    key : str
        W&B API key. Usually injected via ``WANDB_API_KEY`` at launch
        time rather than hardcoded. Default ``""``.
    disable_random_suffix : bool
        When ``True``, suppresses the random suffix that W&B appends to
        run names. Default ``True``.
    modal_wandb_secret_name : str
        Name of the Modal secret containing the W&B API key. Default ``"wandb-secret"``.
    """

    project: str = ""
    entity: str = ""
    group: str = ""
    exp_name: str = ""
    key: str = ""
    disable_random_suffix: bool = True
    modal_wandb_secret_name: str = "wandb-secret"

    provider: ClassVar[str] = "wandb"
    label: ClassVar[str] = "W&B"

    @property
    def modal_secret_name(self) -> str:
        return self.modal_wandb_secret_name

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        env = super().runtime_env(run_id=run_id, entity=entity)
        if run_id:
            env["WANDB_RUN_ID"] = run_id
            env["WANDB_RESUME"] = "allow"
        if entity:
            env["WANDB_ENTITY"] = entity
        return env

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        clean_entity = (entity or self.entity).strip()
        clean_project = self.project.strip()
        clean_run_id = run_id.strip()
        if not clean_entity or not clean_project:
            return None
        base = (
            f"https://wandb.ai/{quote(clean_entity, safe='')}/"
            f"{quote(clean_project, safe='')}"
        )
        return f"{base}/runs/{quote(clean_run_id, safe='')}" if clean_run_id else base


@dataclass
class TrackioConfig(MetricConfig):
    """Trackio logging configuration using Slime/Miles' W&B-compatible path.

    ## Fields

    project : str
        Trackio project name. Default ``""``.
    group : str
        Group tag for organizing related runs. Default ``""``.
    exp_name : str
        Run display name. Default ``""``.
    disable_random_suffix : bool
        When ``True``, suppresses the random suffix appended to run names.
        Default ``True``.
    space_id : str
        Hugging Face Space ID hosting the Trackio dashboard
        (e.g. ``"org/space"``). Default ``""``.
    server_url : str
        URL of a self-hosted Trackio server. Default ``""``.
    dashboard_url : str
        Explicit dashboard link stored with the run record. Default ``""``.
    bucket_id : str
        Storage bucket ID for Trackio artifacts. Default ``""``.
    token : str
        Hugging Face token (delivered as ``HF_TOKEN`` via an inline Modal
        Secret, never through the Ray runtime env). Becomes the container's
        ``HF_TOKEN``, overriding local and workspace tokens for everything
        that reads it — including gated model downloads — so it must be
        scoped accordingly. Usually injected via the named Modal secret
        rather than hardcoded. Default ``""``.
    write_token : str
        Trackio write token (delivered as ``TRACKIO_WRITE_TOKEN`` via an
        inline Modal Secret, never through the Ray runtime env). Usually
        injected via the named Modal secret rather than hardcoded.
        Default ``""``.
    modal_secret_name : str
        Name of the Modal secret providing tokens. Optional — used only when
        present. Default ``"huggingface-secret"``.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True
    space_id: str = ""
    server_url: str = ""
    dashboard_url: str = ""
    bucket_id: str = ""
    token: str = ""
    write_token: str = ""
    modal_secret_name: str = "huggingface-secret"

    provider: ClassVar[str] = "trackio"
    label: ClassVar[str] = "Trackio"

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        env = super().runtime_env(run_id=run_id, entity=entity)
        if self.space_id:
            env["TRACKIO_SPACE_ID"] = self.space_id
        if (
            self.server_url
            and redact_url_credentials(self.server_url) == self.server_url
        ):
            env["TRACKIO_SERVER_URL"] = self.server_url
        if self.bucket_id:
            env["TRACKIO_BUCKET_ID"] = self.bucket_id
        return env

    def inline_secret_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if (
            self.server_url
            and redact_url_credentials(self.server_url) != self.server_url
        ):
            env["TRACKIO_SERVER_URL"] = self.server_url
        if self.token:
            env["HF_TOKEN"] = self.token
        if self.write_token:
            env["TRACKIO_WRITE_TOKEN"] = self.write_token
        return env

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        _ = entity, run_id
        if self.dashboard_url:
            return redact_url_credentials(_strip_url_userinfo(self.dashboard_url))
        if self.space_id:
            return f"https://huggingface.co/spaces/{quote(self.space_id, safe='/')}"
        if self.server_url:
            parts = urlsplit(self.server_url)
            host = parts.netloc.rsplit("@", 1)[-1]
            return urlunsplit((parts.scheme, host, parts.path, "", ""))
        return None


def _strip_url_userinfo(url: str) -> str:
    """Drop URL userinfo so stored links stay clickable instead of carrying
    a redaction placeholder."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[-1]))


def named_metric_secrets(metric: MetricConfig | None) -> list[Any]:
    """Named Modal Secret for ``metric``, if any."""
    import modal

    if metric is None:
        return []
    raw_name = getattr(metric, "modal_secret_name", None)
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    if name is None:
        return []
    secret = modal.Secret.from_name(name)
    if isinstance(metric, WandbConfig):
        return [secret]
    if not isinstance(metric, TrackioConfig) or name != "huggingface-secret":
        return [secret]
    try:
        # Modal Secret references are lazy; hydrate checks whether Trackio's
        # optional default workspace secret actually exists before attaching it.
        secret.hydrate()
        return [secret]
    except Exception:
        return []


def inline_metric_secrets(metric: MetricConfig | None) -> list[Any]:
    """Inline Modal Secret carrying explicit config credentials, if any."""
    import modal

    if metric is None:
        return []
    inline_env = metric.inline_secret_env()
    return [modal.Secret.from_dict(inline_env)] if inline_env else []


def metric_cli_fields(metric: MetricConfig) -> dict[str, Any]:
    fields = {
        "use_wandb": True,
        "wandb_project": metric.project,
        "wandb_group": metric.group,
        "disable_wandb_random_suffix": metric.disable_random_suffix,
    }
    if isinstance(metric, WandbConfig):
        fields["wandb_key"] = metric.key
    return fields


def metric_runtime_env(
    metric: MetricConfig | None, *, run_id: str, entity: str = ""
) -> dict[str, str]:
    return {} if metric is None else metric.runtime_env(run_id=run_id, entity=entity)


def metrics_metadata(
    metric: MetricConfig | None, *, entity: str = "", run_id: str = ""
) -> dict[str, str]:
    return {} if metric is None else metric.to_metadata(entity=entity, run_id=run_id)


def apply_metric_image_overrides(image: Any, metric: MetricConfig | None) -> Any:
    if not isinstance(metric, TrackioConfig):
        return image
    return image.uv_pip_install("trackio").run_commands(_TRACKIO_WANDB_SHIM_COMMAND)


def preflight_metrics(metric: MetricConfig | None) -> str:
    if metric is None:
        return ""
    if isinstance(metric, WandbConfig):
        return preflight_wandb(metric)
    if isinstance(metric, TrackioConfig):
        import trackio  # noqa: F401

        return ""
    return ""


def preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Return the resolved W&B entity for constructing deep-links."""
    key = os.environ.get("WANDB_API_KEY", "") or (wandb_cfg.key or "")
    if not key:
        raise RuntimeError(
            "W&B logging is enabled but no WANDB_API_KEY is available - add it "
            f"to the Modal secret '{wandb_cfg.modal_wandb_secret_name}' (or set "
            "wandb.key=), or drop the metrics config to disable logging."
        )

    import wandb

    project = wandb_cfg.project or "uncategorized"
    entity = wandb_cfg.entity or os.environ.get("WANDB_ENTITY", "")
    try:
        wandb.login(key=key, verify=True, relogin=True)
        probe = wandb.init(
            project=project,
            entity=entity or None,
            name="_preflight",
            settings=wandb.Settings(silent=True, init_timeout=60),
        )
        entity = probe.entity
        probe_path = f"{probe.entity}/{probe.project}/{probe.id}"
        wandb.finish()
        try:
            wandb.Api(api_key=key).run(probe_path).delete()
        except Exception:
            pass
    except Exception as exc:
        raise RuntimeError(
            f"W&B pre-flight failed for project '{project}': {exc}\n"
            f"The W&B key in Modal secret '{wandb_cfg.modal_wandb_secret_name}' "
            "can't log there (bad/expired key, or no write access to its "
            "entity). Point metrics at a project/entity you can write to, fix "
            "the secret, or drop metrics= to disable logging."
        ) from exc
    return entity


# Installed at image-build time; base64-encoded so run_commands gets a single
# line (see common/patches.py for the same convention).
_TRACKIO_WANDB_SHIM_SCRIPT = r'''import pathlib
import site

code = """import os
import sys

if os.environ.get("TRAINING_GYM_METRICS_PROVIDER") == "trackio":
    import trackio as wandb
    sys.modules["wandb"] = wandb
"""

for path in site.getsitepackages():
    package_dir = pathlib.Path(path)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "training_gym_trackio_wandb_shim.py").write_text(code)
    (package_dir / "training_gym_trackio_wandb_shim.pth").write_text(
        "import training_gym_trackio_wandb_shim\n"
    )
'''

_TRACKIO_WANDB_SHIM_COMMAND = (
    "echo "
    + base64.b64encode(_TRACKIO_WANDB_SHIM_SCRIPT.encode()).decode()
    + " | base64 -d | python3"
)
