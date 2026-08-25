"""Trackio metric logging configuration and W&B compatibility adapter."""

from __future__ import annotations

import os
import shlex
import sys
import types
import uuid
from importlib import import_module
from importlib.machinery import ModuleSpec
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from modal_training_gym.common.metrics import MetricConfig


TRACKIO_VERSION = "0.34.0"
_RUN_NAME_ENV = "TRAINING_GYM_TRACKIO_RUN_NAME"
_SHIM_MARKER = "_training_gym_trackio_adapter"
_PTH_LINE = (
    "import os; os.environ.get('TRAINING_GYM_METRIC_PROVIDER') != 'trackio' "
    "or __import__('modal_training_gym.common.trackio', "
    "fromlist=['install_wandb_shim']).install_wandb_shim()\n"
)


@dataclass
class TrackioConfig(MetricConfig):
    """Trackio logging configuration shared across all frameworks.

    Trackio can log to a Hugging Face Space or a self-hosted server. Training
    images install Trackio automatically and adapt the W&B calls made by the
    underlying framework.

    ## Fields

    project : str
        Trackio project name. Default ``""`` (uses ``"training-gym"``).
    group : str
        Group tag for related runs. Default ``""``.
    exp_name : str
        Run display name. Default ``""``.
    disable_random_suffix : bool
        Whether the framework should preserve the configured group name.
        Default ``True``.
    space_id : str
        Hugging Face Space ID, such as ``"owner/trackio"``. Optional.
    server_url : str
        URL of a self-hosted Trackio server. Optional.
    dashboard_url : str
        Explicit dashboard URL. Optional; otherwise derived from ``space_id``
        or ``server_url``.
    bucket_id : str
        Hugging Face Bucket used by the Trackio Space. Optional.
    modal_secret_name : str
        Modal Secret containing ``HF_TOKEN`` or ``TRACKIO_WRITE_TOKEN``.
        The standard optional ``"huggingface-secret"`` is used by default.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True
    space_id: str = ""
    server_url: str = ""
    dashboard_url: str = ""
    bucket_id: str = ""
    modal_secret_name: str = "huggingface-secret"

    provider: ClassVar[str] = "trackio"  # pyright: ignore[reportIncompatibleMethodOverride]

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        env = super().runtime_env(run_id=run_id, entity=entity)
        for key, value in (
            (_RUN_NAME_ENV, run_id),
            ("TRACKIO_SPACE_ID", self.space_id),
            ("TRACKIO_SERVER_URL", self.server_url),
            ("TRACKIO_BUCKET_ID", self.bucket_id),
        ):
            if value:
                env[key] = value
        return env

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        if self.dashboard_url:
            url = self.dashboard_url
        else:
            space_id = self.space_id.strip().strip("/")
            if "/" in space_id:
                url = "https://huggingface.co/spaces/" + quote(space_id, safe="/")
            elif self.server_url:
                url = self.server_url
            else:
                return None
        return _without_credentials(
            url, project=self.project or "training-gym", run_id=run_id
        )


def _without_credentials(url: str, *, project: str = "", run_id: str = "") -> str:
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().split("?", 1)[0].split("#", 1)[0]
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"api_key", "hf_token", "token", "write_token"}
    ]
    query_keys = {key.lower() for key, _ in query_items}
    if project and "project" not in query_keys:
        query_items.append(("project", project))
    if run_id and not query_keys.intersection({"run_ids", "runs"}):
        query_items.append(("runs", run_id))
    query = urlencode(query_items)
    return urlunsplit((parsed.scheme, host, parsed.path, query, ""))


def trackio_secrets(config: TrackioConfig) -> list[Any]:
    if not config.modal_secret_name or config.modal_secret_name == "huggingface-secret":
        return []
    from modal import Secret

    return [Secret.from_name(config.modal_secret_name)]


def apply_trackio_image(image: Any) -> Any:
    install_code = (
        "import pathlib, site; "
        "pathlib.Path(site.getsitepackages()[0], "
        f"'_training_gym_trackio.pth').write_text({_PTH_LINE!r})"
    )
    return image.uv_pip_install(f"trackio=={TRACKIO_VERSION}").run_commands(
        f"python3 -c {shlex.quote(install_code)}"
    )


def preflight_trackio(_config: TrackioConfig) -> str:
    try:
        import trackio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Trackio logging is enabled, but Trackio is missing from the training image."
        ) from exc
    return ""


class _Settings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _RunProxy:
    def __init__(self, run: Any, resume_name: str) -> None:
        self._run = run
        self._resume_name = resume_name

    @property
    def id(self) -> str:
        return self._resume_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)


def install_wandb_shim() -> None:
    """Expose Trackio as the W&B subset used by Slime and Miles."""
    current = sys.modules.get("wandb")
    if current is not None and getattr(current, _SHIM_MARKER, False):
        return

    trackio: Any = import_module("trackio")

    shim: Any = types.ModuleType("wandb")
    shim.__spec__ = ModuleSpec("wandb", loader=None, is_package=True)
    shim.__path__ = []
    setattr(shim, _SHIM_MARKER, True)
    shim.run = None
    shim.config = {}
    shim.Settings = _Settings

    def init(*args: Any, **kwargs: Any) -> _RunProxy:
        project = kwargs.pop("project", args[0] if args else "") or "training-gym"
        requested_name = kwargs.pop("id", "") or os.environ.get(_RUN_NAME_ENV, "")
        requested_name = requested_name or kwargs.pop("name", "")
        resume = kwargs.pop("resume", "allow" if requested_name else "never")
        kwargs.pop("settings", None)
        run = trackio.init(
            project=project,
            name=requested_name or None,
            group=kwargs.pop("group", None),
            config=kwargs.pop("config", None),
            resume=resume,
            embed=False,
        )
        # Materialize the remote run before Slime's worker processes resume it.
        trackio.log({}, step=-1)
        proxy = _RunProxy(run, requested_name or run.name)
        shim.run = proxy
        shim.config = run.config
        return proxy

    def finish(*args: Any, **kwargs: Any) -> Any:
        try:
            return trackio.finish(*args, **kwargs)
        finally:
            shim.run = None

    def generate_id() -> str:
        return uuid.uuid4().hex[:8]

    shim.init = init
    shim.log = trackio.log
    shim.finish = finish
    shim.save = trackio.save
    shim.login = lambda **kwargs: True
    shim.define_metric = lambda *args, **kwargs: None
    shim.__getattr__ = lambda name: getattr(trackio, name)

    util: Any = types.ModuleType("wandb.util")
    util.__spec__ = ModuleSpec("wandb.util", loader=None)
    util.generate_id = generate_id
    sdk: Any = types.ModuleType("wandb.sdk")
    sdk.__spec__ = ModuleSpec("wandb.sdk", loader=None, is_package=True)
    sdk.__path__ = []
    lib: Any = types.ModuleType("wandb.sdk.lib")
    lib.__spec__ = ModuleSpec("wandb.sdk.lib", loader=None, is_package=True)
    lib.__path__ = []
    runid: Any = types.ModuleType("wandb.sdk.lib.runid")
    runid.__spec__ = ModuleSpec("wandb.sdk.lib.runid", loader=None)
    runid.generate_id = generate_id
    shim.util = util
    shim.sdk = sdk
    sdk.lib = lib
    lib.runid = runid

    sys.modules.update(
        {
            "wandb": shim,
            "wandb.util": util,
            "wandb.sdk": sdk,
            "wandb.sdk.lib": lib,
            "wandb.sdk.lib.runid": runid,
        }
    )
