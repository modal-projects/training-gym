from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import py_compile
from types import ModuleType, SimpleNamespace

import pytest

from modal_training_gym import TrackioConfig
from modal_training_gym.common.trackio import trackio_project_url, trackio_run_name
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.slime_recipe import Qwen3_0_6b_Recipe


def test_trackio_recipe_emits_native_slime_flags() -> None:
    recipe = Qwen3_0_6b_Recipe(
        trackio=TrackioConfig(project="slime-tests", run_name="run-one")
    )

    args = recipe.cli_args()

    assert "--use-trackio" in args
    assert args[args.index("--trackio-project") + 1] == "slime-tests"
    assert "--use-wandb" not in args


def test_tracking_backends_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Qwen3_0_6b_Recipe(
            trackio=TrackioConfig(project="slime-tests"),
            wandb=WandbConfig(project="slime-tests"),
        )


def test_trackio_project_url_filters_permanent_dashboard() -> None:
    assert (
        trackio_project_url("https://trackio.example/", "slime tests")
        == "https://trackio.example/?project=slime+tests"
    )
    assert trackio_project_url(None, "slime-tests") is None


def test_trackio_run_name_defaults_to_training_run_id() -> None:
    assert (
        trackio_run_name(TrackioConfig(project="slime-tests"), "cyan-flask-123")
        == "cyan-flask-123"
    )
    assert (
        trackio_run_name(
            TrackioConfig(project="slime-tests", run_name="display-name"),
            "cyan-flask-123",
        )
        == "display-name"
    )


def _load_trackio_patch():
    patch_path = (
        Path(__file__).parents[1]
        / "modal_training_gym/frameworks/slime/modal_helpers/patches/patch_trackio.py"
    )
    spec = importlib.util.spec_from_file_location("test_patch_trackio", patch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generated_trackio_utils(
    monkeypatch: pytest.MonkeyPatch,
    *,
    training_run_id: str = "cyan-flask-123",
) -> ModuleType:
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", training_run_id)
    monkeypatch.setenv("TRACKIO_RUN_NAME", training_run_id)
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://trackio.example/")
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "secret-token")
    module = ModuleType(f"trackio_utils_{training_run_id}")
    exec(_load_trackio_patch().TRACKIO_UTILS, module.__dict__)
    module.init_trackio_secondary(
        SimpleNamespace(use_trackio=True, trackio_project="slime-tests")
    )
    return module


def test_distributed_trackio_payload_uses_one_stable_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _load_generated_trackio_utils(monkeypatch)
    second = _load_generated_trackio_utils(monkeypatch)

    first_payload = first._bulk_log_payload({"reward": 0.5, "loss": 0.1}, step=7)
    second_payload = second._bulk_log_payload({"reward": 0.5}, step=7)

    assert {entry["run_id"] for entry in first_payload["logs"]} == {"cyan-flask-123"}
    assert first_payload["logs"][0]["log_id"] == second_payload["logs"][0]["log_id"]
    assert {next(iter(entry["metrics"])) for entry in first_payload["logs"]} == {
        "reward",
        "loss",
    }


def test_trackio_preserves_native_train_and_eval_metric_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trackio_utils = _load_generated_trackio_utils(monkeypatch)
    payloads = []
    monkeypatch.setattr(trackio_utils, "_post", payloads.append)

    trackio_utils.log({"rollout/average_last_reward": 0.625}, step=0)
    trackio_utils.log(
        {"eval/train-2-smoke": 0.5, "eval/eval-2-smoke": 0.25},
        step=1,
    )

    assert len(payloads) == 2
    entries = [entry for payload in payloads for entry in payload["logs"]]
    assert {entry["run_id"] for entry in entries} == {"cyan-flask-123"}
    assert {next(iter(entry["metrics"])) for entry in entries} == {
        "rollout/average_last_reward",
        "eval/train-2-smoke",
        "eval/eval-2-smoke",
    }


def test_trackio_logging_is_authenticated_and_non_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trackio_utils = _load_generated_trackio_utils(monkeypatch)
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"{}"

    def capture(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(trackio_utils.urllib.request, "urlopen", capture)
    trackio_utils.log({"reward": 0.5}, step=7)
    assert requests[0][0].get_header("X-trackio-write-token") == "secret-token"
    assert requests[0][0].full_url == "https://trackio.example/api/bulk_log"

    def unavailable(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(trackio_utils.urllib.request, "urlopen", unavailable)
    with caplog.at_level(logging.ERROR):
        trackio_utils.log({"reward": 0.6}, step=8)
    assert "Trackio bulk_log" in caplog.text


def test_trackio_init_requires_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAINING_GYM_TRAINING_RUN_ID", raising=False)
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://trackio.example")
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "secret-token")
    module = ModuleType("trackio_utils_missing_identity")
    exec(_load_trackio_patch().TRACKIO_UTILS, module.__dict__)

    with pytest.raises(RuntimeError, match="TRAINING_GYM_TRAINING_RUN_ID"):
        module.init_trackio_primary(
            SimpleNamespace(use_trackio=True, trackio_project="slime-tests")
        )


def test_recipe_environment_cannot_override_trackio_identity() -> None:
    launcher = (
        Path(__file__).parents[1] / "modal_training_gym/frameworks/slime/launcher.py"
    ).read_text()
    runtime_env = launcher[launcher.index("runtime_env = {") :]

    assert runtime_env.index("**slime.environment") < runtime_env.index("**trackio_env")
    assert runtime_env.index("**trackio_env") < runtime_env.index(
        '"TRAINING_GYM_TRAINING_RUN_ID": training_run_id'
    )


def test_trackio_source_patch_is_idempotent(tmp_path: Path) -> None:
    utils_dir = tmp_path / "slime/utils"
    utils_dir.mkdir(parents=True)
    arguments = utils_dir / "arguments.py"
    arguments.write_text(
        """
def get_extra_args_provider():
    # wandb
    def add_wandb_arguments(parser):
        return parser

    parser = add_wandb_arguments(parser)
    return add_wandb_arguments
"""
    )
    logging_utils = utils_dir / "logging_utils.py"
    logging_utils.write_text(
        """import logging
import wandb

from . import wandb_utils

def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")

def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        pass
"""
    )

    patch = _load_trackio_patch()
    patch.SLIME_ROOT = tmp_path
    patch.main()
    patch.main()

    assert "--use-trackio" in arguments.read_text()
    assert "trackio_utils.init_trackio_secondary(args)" in logging_utils.read_text()
    assert "step=metrics.get(step_key)" in logging_utils.read_text()
    py_compile.compile(str(arguments), doraise=True)
    py_compile.compile(str(logging_utils), doraise=True)
    py_compile.compile(str(utils_dir / "trackio_utils.py"), doraise=True)
