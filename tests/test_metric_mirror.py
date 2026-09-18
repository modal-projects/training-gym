"""Container-side metric mirror: flattening, step semantics, batching, shims."""

from __future__ import annotations

import sys
import types

import pytest

from modal_training_gym.common import metric_mirror, reporting
from modal_training_gym.common.metric_mirror import (
    MIRROR_MODE_ENV,
    DashboardMetricConfig,
    MetricMirror,
    flatten_numeric,
    install_wandb_shim,
    mirror_mode,
    patch_wandb_module,
)
from modal_training_gym.common.metrics import (
    apply_metric_image,
    metric_runtime_env,
    metric_secrets,
    preflight_metric,
)
from modal_training_gym.common.trackio import TrackioConfig
from modal_training_gym.common.wandb import WandbConfig


class _Scalar:
    def __init__(self, value, shape=()):
        self._value = value
        self.shape = shape

    def item(self):
        return self._value


def test_flatten_keeps_finite_numbers_and_drops_everything_else():
    flat = flatten_numeric(
        {
            "train": {"loss": 0.5, "lr": 1e-4, "nested": {"deep": 2}},
            "reward": 1,
            "tensor": _Scalar(3.5),
            "vector": _Scalar(1.0, shape=(4,)),
            "flag": True,
            "name": "qwen",
            "nan": float("nan"),
            "inf": float("inf"),
            "none": None,
            "image": object(),
        }
    )
    assert flat == {
        "train/loss": 0.5,
        "train/lr": 1e-4,
        "train/nested/deep": 2.0,
        "reward": 1.0,
        "tensor": 3.5,
    }


def _mirror(**kwargs) -> tuple[MetricMirror, list[dict], dict[str, float]]:
    sent: list[dict] = []
    clock = {"now": 1000.0}
    mirror = MetricMirror(
        "run-1",
        source="host:1",
        clock=lambda: clock["now"],
        send=lambda payload, final: sent.append(payload),
        **kwargs,
    )
    return mirror, sent, clock


def test_implicit_step_and_commit_follow_wandb_semantics():
    mirror, sent, _clock = _mirror(flush_interval=3600)
    assert mirror.log({"a": 1}) == 0
    assert mirror.log({"b": 2}, commit=False) == 1
    assert mirror.log({"c": 3}) == 1
    assert mirror.log({"d": 4}, step=10) == 10
    assert mirror.log({"e": 5}) == 10
    assert mirror.log({"f": 6}, step=3) == 3  # explicit older step still lands
    assert mirror.log({"g": 7}) == 11
    mirror.flush()
    assert [(p["step"], p["metrics"]) for p in sent[0]["points"]] == [
        (0, {"a": 1.0}),
        (1, {"b": 2.0, "c": 3.0}),
        (3, {"f": 6.0}),
        (10, {"d": 4.0, "e": 5.0}),
        (11, {"g": 7.0}),
    ]
    assert sent[0]["training_run_id"] == "run-1"
    assert sent[0]["source"] == "host:1"
    assert sent[0]["final"] is False


def test_repeated_key_within_a_step_is_last_write_wins():
    mirror, sent, _clock = _mirror(flush_interval=3600)
    mirror.log({"loss": 1.0}, step=5)
    mirror.log({"loss": 0.5, "lr": 0.1}, step=5)
    mirror.flush()
    assert sent[0]["points"] == [
        {"step": 5, "ts": 1000.0, "metrics": {"loss": 0.5, "lr": 0.1}}
    ]


def test_invalid_steps_and_non_numeric_payloads_are_dropped():
    mirror, sent, _clock = _mirror(flush_interval=3600)
    assert mirror.log({"a": 1}, step=-1) is None
    assert mirror.log({"a": 1}, step="x") is None  # type: ignore[arg-type]
    assert mirror.log({"img": object()}, step=2) == 2
    assert mirror.log(None, step=3) == 3
    assert mirror.flush() == 0
    assert sent == []


def test_batches_flush_on_size_and_interval_not_per_call():
    mirror, sent, clock = _mirror(flush_interval=2.0, max_pending_steps=3)
    mirror.log({"a": 1}, step=0)
    mirror.log({"a": 1}, step=1)
    assert sent == []
    mirror.log({"a": 1}, step=2)
    assert len(sent) == 1 and len(sent[0]["points"]) == 3
    clock["now"] += 1.0
    mirror.log({"a": 1}, step=3)
    assert len(sent) == 1
    clock["now"] += 1.5
    mirror.log({"a": 1}, step=4)
    assert len(sent) == 2 and [p["step"] for p in sent[1]["points"]] == [3, 4]


def test_final_flush_is_sent_even_when_empty():
    mirror, sent, _clock = _mirror(flush_interval=3600)
    mirror.flush(final=True)
    assert sent == [
        {"training_run_id": "run-1", "points": [], "source": "host:1", "final": True}
    ]


def test_send_failures_never_propagate():
    def boom(payload, final):
        raise RuntimeError("network down")

    mirror = MetricMirror("run-1", send=boom, flush_interval=3600)
    mirror.log({"a": 1}, step=0)
    assert mirror.flush() == 0


def test_enqueue_metric_points_derives_url_and_retries(monkeypatch):
    monkeypatch.setenv(
        "TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dash.test/api/framework-status"
    )
    items = []
    monkeypatch.setattr(reporting._REPORT_QUEUE, "put_nowait", items.append)
    monkeypatch.setattr(reporting, "_ensure_worker", lambda **kw: None)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    reporting._enqueue_metric_points({"training_run_id": "r", "points": []})
    reporting._enqueue_metric_points(
        {"training_run_id": "r", "points": [], "final": True}, final=True
    )
    assert [item["_url"] for item in items] == [
        "https://dash.test/api/metric-points",
        "https://dash.test/api/metric-points",
    ]
    assert [item["_retry_count"] for item in items] == [1, 3]

    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", True)
    reporting._enqueue_metric_points({"training_run_id": "r", "points": []})
    assert len(items) == 2  # non-final batches are dropped while draining
    reporting._enqueue_metric_points({"training_run_id": "r"}, final=True)
    assert len(items) == 3


def test_active_mirror_needs_run_id_and_dashboard_url(monkeypatch):
    metric_mirror.reset_active_mirror()
    monkeypatch.delenv("TRAINING_GYM_TRAINING_RUN_ID", raising=False)
    monkeypatch.delenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", raising=False)
    monkeypatch.delenv("SLIME_PHASE_REPORT_URL", raising=False)
    assert metric_mirror.active_mirror() is None

    metric_mirror.reset_active_mirror()
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-env")
    monkeypatch.setenv(
        "TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dash.test/api/framework-status"
    )
    hooks = []
    monkeypatch.setattr(reporting, "register_pre_drain_hook", hooks.append)
    mirror = metric_mirror.active_mirror()
    assert mirror is not None and mirror.training_run_id == "run-env"
    assert metric_mirror.active_mirror() is mirror
    assert len(hooks) == 1
    metric_mirror.reset_active_mirror()


# ── config wiring ─────


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


def test_dashboard_config_is_a_sink_without_secrets_or_preflight():
    config = DashboardMetricConfig(project="p", group="g", exp_name="e")
    assert config.provider == "dashboard"
    assert mirror_mode(config) == "sink"
    assert metric_runtime_env(config, run_id="r") == {
        "TRAINING_GYM_METRIC_PROVIDER": "dashboard",
        MIRROR_MODE_ENV: "sink",
    }
    assert config.metadata(run_id="r") == {
        "provider": "dashboard",
        "project": "p",
        "group": "g",
        "entity": "",
        "run_id": "r",
    }
    assert metric_secrets(config) == []
    assert preflight_metric(config) == ""
    image = apply_metric_image(_FakeImage(), config)
    assert image.packages == []
    assert len(image.commands) == 1
    assert "_training_gym_metric_mirror.pth" in image.commands[0]
    assert "TRAINING_GYM_METRIC_MIRROR" in image.commands[0]


def test_wandb_and_trackio_tee_by_default_and_can_opt_out():
    assert (
        metric_runtime_env(WandbConfig(project="p"), run_id="r")[MIRROR_MODE_ENV]
        == "tee"
    )
    assert MIRROR_MODE_ENV not in metric_runtime_env(
        WandbConfig(project="p", mirror_to_dashboard=False), run_id="r"
    )
    trackio = TrackioConfig(project="p")
    assert metric_runtime_env(trackio, run_id="r")[MIRROR_MODE_ENV] == "tee"
    image = apply_metric_image(_FakeImage(), trackio)
    assert image.packages == [f"trackio=={trackio.TRACKIO_PACKAGE_VERSION}"]
    assert len(image.commands) == 1
    assert "_training_gym_trackio.pth" in image.commands[0]
    assert "_training_gym_metric_mirror.pth" in image.commands[0]
    assert mirror_mode(None) == ""


# ── sink shim ─────


@pytest.fixture
def isolated_wandb(monkeypatch):
    saved = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name.split(".")[0] == "wandb"
    }
    for name in saved:
        monkeypatch.delitem(sys.modules, name)
    metric_mirror.reset_active_mirror()
    yield
    for name in list(sys.modules):
        if name.split(".")[0] == "wandb":
            monkeypatch.delitem(sys.modules, name, raising=False)
    metric_mirror.reset_active_mirror()


def test_sink_shim_routes_wandb_calls_to_the_mirror(isolated_wandb, monkeypatch):
    sent: list[dict] = []
    mirror = MetricMirror(
        "run-shim",
        source="s",
        flush_interval=3600,
        send=lambda payload, final: sent.append(payload),
    )
    monkeypatch.setattr(metric_mirror, "_ACTIVE", mirror)
    install_wandb_shim()
    import wandb  # noqa: PLC0415

    assert wandb._training_gym_metric_mirror is True
    run = wandb.init(project="p", name="exp", config={"lr": 0.1})
    assert wandb.run is run
    assert run.name == "exp"
    assert run.config.lr == 0.1
    run.config.update({"steps": 3})
    assert wandb.config["steps"] == 3
    assert wandb.util.generate_id()
    assert wandb.Image("x") is not None
    with pytest.raises(AttributeError):
        wandb.not_a_thing  # noqa: B018

    wandb.log({"train/loss": 0.9, "img": wandb.Image("x")}, step=1)
    wandb.log({"train/loss": 0.4}, step=2)
    run.log({"reward": 1.5}, step=2)
    assert run.summary == {"train/loss": 0.4, "reward": 1.5}
    wandb.define_metric("train/loss", summary="min")
    wandb.save("file.txt")
    assert wandb.login() is True
    wandb.finish()
    assert wandb.run is None
    assert [(p["step"], p["metrics"]) for p in sent[0]["points"]] == [
        (1, {"train/loss": 0.9}),
        (2, {"train/loss": 0.4, "reward": 1.5}),
    ]


# ── tee over real wandb ─────


def _fake_wandb_package() -> types.ModuleType:
    wandb = types.ModuleType("wandb")
    sdk = types.ModuleType("wandb.sdk")
    wandb_run = types.ModuleType("wandb.sdk.wandb_run")

    class Run:
        def __init__(self) -> None:
            self._step = 0
            self.logged: list[tuple] = []

        @property
        def step(self) -> int:
            return self._step

        def log(self, data, step=None, commit=None, sync=None):
            self.logged.append((dict(data), step, commit))
            if step is None:
                if commit is not False:
                    self._step += 1
            elif step > self._step:
                self._step = step

        def finish(self, exit_code=None, quiet=None):
            self.finished = True

    wandb_run.Run = Run  # type: ignore[attr-defined]
    sdk.wandb_run = wandb_run  # type: ignore[attr-defined]
    wandb.sdk = sdk  # type: ignore[attr-defined]
    sys.modules["wandb"] = wandb
    sys.modules["wandb.sdk"] = sdk
    sys.modules["wandb.sdk.wandb_run"] = wandb_run
    return wandb


def test_tee_patches_run_log_and_mirrors_at_wandb_step(isolated_wandb, monkeypatch):
    sent: list[dict] = []
    mirror = MetricMirror(
        "run-tee",
        source="s",
        flush_interval=3600,
        send=lambda payload, final: sent.append(payload),
    )
    monkeypatch.setattr(metric_mirror, "_ACTIVE", mirror)
    wandb = _fake_wandb_package()
    assert patch_wandb_module(wandb) is True
    assert patch_wandb_module(wandb) is True  # idempotent

    run = wandb.sdk.wandb_run.Run()
    run.log({"a": 1})  # implicit step 0, then wandb advances to 1
    run.log({"b": 2}, commit=False)  # stays on step 1
    run.log({"c": 3})  # step 1
    run.log({"d": 4}, step=7)
    run.log({"e": "text", "f": 5.5})  # step 7 (wandb's current step)
    run.finish()
    assert run.finished is True
    assert len(run.logged) == 5  # every call still reached wandb

    assert len(sent) == 1  # finish() flushed
    assert [(p["step"], p["metrics"]) for p in sent[0]["points"]] == [
        (0, {"a": 1.0}),
        (1, {"b": 2.0, "c": 3.0}),
        (7, {"d": 4.0, "f": 5.5}),
    ]


def test_tee_leaves_shims_alone(isolated_wandb):
    shim = types.ModuleType("wandb")
    shim._training_gym_trackio_adapter = True  # type: ignore[attr-defined]
    assert patch_wandb_module(shim) is False


def test_bootstrap_selects_mode_from_env(isolated_wandb, monkeypatch):
    calls = []
    monkeypatch.setattr(
        metric_mirror, "install_wandb_shim", lambda: calls.append("sink")
    )
    monkeypatch.setattr(metric_mirror, "install_wandb_tee", lambda: calls.append("tee"))
    monkeypatch.delenv(MIRROR_MODE_ENV, raising=False)
    metric_mirror.bootstrap()
    monkeypatch.setenv(MIRROR_MODE_ENV, "sink")
    metric_mirror.bootstrap()
    monkeypatch.setenv(MIRROR_MODE_ENV, "tee")
    metric_mirror.bootstrap()
    assert calls == ["sink", "tee"]


def test_trackio_shim_tees_when_mirror_mode_is_set(isolated_wandb, monkeypatch):
    from modal_training_gym.common import trackio as trackio_module

    sent: list[dict] = []
    mirror = MetricMirror(
        "run-trackio",
        source="s",
        flush_interval=3600,
        send=lambda payload, final: sent.append(payload),
    )
    monkeypatch.setattr(metric_mirror, "_ACTIVE", mirror)
    monkeypatch.setenv(MIRROR_MODE_ENV, "tee")
    monkeypatch.setenv("TRAINING_GYM_METRIC_PROVIDER", "trackio")

    fake_trackio = types.ModuleType("trackio")
    logged = []

    class _Run:
        def log(self, metrics, step=None):
            logged.append((metrics, step))

    fake_trackio.init = lambda **kwargs: _Run()  # type: ignore[attr-defined]
    fake_trackio.log = lambda data, step=None: logged.append((data, step))  # type: ignore[attr-defined]
    fake_trackio.finish = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    monkeypatch.setattr(trackio_module, "import_module", lambda name: fake_trackio)

    trackio_module.install_wandb_shim()
    import wandb  # noqa: PLC0415

    wandb.log({"x": 1.0}, step=4)
    mirror.flush()
    assert logged and logged[-1][1] == 4
    assert sent and sent[0]["points"] == [
        {"step": 4, "ts": sent[0]["points"][0]["ts"], "metrics": {"x": 1.0}}
    ]
