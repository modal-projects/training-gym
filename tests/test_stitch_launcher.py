import inspect
import json
import os
import runpy
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import modal
import pytest

from modal_training_gym import Qwen3_30B, TrainConfig, WandbConfig
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.frameworks.stitch import launcher
from modal_training_gym.train_recipes import base
from modal_training_gym.train_recipes.stitch_recipe import (
    Qwen3_30B_A3B_Stitch_Recipe,
    Qwen3_30B_A3B_Stitch_Serve,
    Qwen3_30B_A3B_Stitch_Train,
)


class RowsDataset(DatasetConfig):
    def __init__(self, value):
        self.value = value
        self.writes = []

    def cache_key(self):
        return None

    def input_key(self):
        return "prompt"

    def label_key(self):
        return "label"

    def rows(self):
        yield {"prompt": self.value, "label": self.value}

    def write(self, path):
        self.writes.append(path)
        super().write(path)


@pytest.fixture
def build_app(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "DATA_PATH", tmp_path / "data")
    monkeypatch.setattr(launcher, "hf_secrets", lambda: [])
    monkeypatch.setattr(launcher, "proxy_auth_secrets", lambda: [])
    monkeypatch.setattr(
        launcher, "_stitch_trainer_image", lambda _: modal.Image.debian_slim()
    )
    monkeypatch.setattr(
        launcher.serving_image,
        "build_serving_image",
        lambda **_: modal.Image.debian_slim(),
    )
    monkeypatch.setattr(modal.Volume, "reload", Mock())
    monkeypatch.setattr(modal.Volume, "commit", Mock())

    def build(dataset=None, eval_dataset=None, recipe=None, **kwargs):
        return launcher.build_stitch_app(
            model=Qwen3_30B(),
            dataset=dataset or RowsDataset("training"),
            eval_dataset=eval_dataset,
            recipe=recipe or Qwen3_30B_A3B_Stitch_Recipe(),
            **kwargs,
        )

    return build


@pytest.mark.parametrize("prepare_first", [False, True])
@pytest.mark.parametrize("with_eval", [False, True])
def test_preparation_and_cold_trainer_share_dataset_paths(
    build_app, monkeypatch, prepare_first, with_eval
):
    dataset = RowsDataset("training")
    evaluation = RowsDataset("evaluation") if with_eval else None
    app = build_app(dataset, evaluation)
    train = inspect.unwrap(app.train.get_raw_f())
    captured = inspect.getclosurevars(train).nonlocals
    if prepare_first:
        inspect.unwrap(app.prepare_dataset.get_raw_f())()

    monkeypatch.setitem(
        sys.modules,
        "cookbook.common",
        SimpleNamespace(
            hooks=Mock(),
            launch=Mock(),
            process=Mock(),
            ray_cluster=SimpleNamespace(
                get_modal_cluster_context=lambda _: (0, "127.0.0.1", "127.0.0.1")
            ),
        ),
    )
    monkeypatch.setattr(launcher, "start_ray_head", Mock())
    monkeypatch.setattr(
        Qwen3_30B, "download", Mock(side_effect=RuntimeError("stop after datasets"))
    )
    with (
        patch.dict(os.environ),
        pytest.raises(RuntimeError, match="stop after datasets"),
    ):
        train()

    inspect.unwrap(app.prepare_dataset.get_raw_f())()
    assert dataset.writes == [captured["dataset_path"]]
    assert json.loads(Path(dataset.writes[0]).read_text())["prompt"] == "training"
    payload = captured["recipe"].to_payload(
        model=captured["model"],
        dataset=dataset,
        eval_dataset=evaluation,
        dataset_path=captured["dataset_path"],
        eval_dataset_path=captured["eval_dataset_path"],
    )
    assert payload.fields["prompt_data"] == dataset.writes[0]
    if evaluation is not None:
        assert evaluation.writes == [captured["eval_dataset_path"]]
        assert evaluation.writes != dataset.writes
        assert (
            json.loads(Path(evaluation.writes[0]).read_text())["prompt"] == "evaluation"
        )
        assert payload.fields["eval_prompt_data"] == ["eval", evaluation.writes[0]]
    else:
        assert payload.fields.get("eval_prompt_data") is None
    assert (
        inspect.getclosurevars(
            inspect.unwrap(build_app(dataset, evaluation).prepare_dataset.get_raw_f())
        ).nonlocals["dataset_path"]
        != dataset.writes[0]
    )


def test_stitch_rejects_incompatible_eval_dataset(build_app):
    class OtherInputDataset(RowsDataset):
        def input_key(self):
            return "messages"

    with pytest.raises(TrainingGymConfigError, match="same input_key"):
        build_app(RowsDataset("training"), OtherInputDataset("evaluation"))


def test_train_config_forwards_eval_dataset(monkeypatch):
    evaluation = RowsDataset("evaluation")
    build = Mock(return_value=object())
    monkeypatch.setattr("modal_training_gym.frameworks.stitch.build_stitch_app", build)
    config = TrainConfig(
        model=Qwen3_30B(),
        dataset=RowsDataset("training"),
        eval_dataset=evaluation,
        recipe=Qwen3_30B_A3B_Stitch_Recipe(),
    )
    with pytest.warns(UserWarning, match="TrainConfig._build_app"):
        assert config._build_app("test-run") is build.return_value
    assert build.call_args.kwargs["eval_dataset"] is evaluation


def test_server_budget_and_draft_checkpoint_volume(build_app, monkeypatch):
    captured = {}
    cls = modal.App.cls

    def capture_cls(self, **kwargs):
        captured.update(kwargs)
        return cls(self, **kwargs)

    monkeypatch.setattr(modal.App, "cls", capture_cls)
    volume = modal.Volume.from_name("draft-checkpoints")
    serve = Qwen3_30B_A3B_Stitch_Serve(startup_timeout=1234, volumes={"/draft": volume})
    serve.sglang.extra_server_args["--speculative-draft-model-path"] = "/draft/dflash"
    app = build_app(recipe=Qwen3_30B_A3B_Stitch_Recipe(serve=serve))
    startup = inspect.unwrap(
        app.registered_classes["Server"]._get_user_cls().startup._get_raw_f()
    )
    closure = inspect.getclosurevars(startup).nonlocals
    baseline_wait = inspect.getclosurevars(closure["local_checkpoint"]).nonlocals[
        "baseline_wait_timeout"
    ]
    assert baseline_wait == (
        launcher.DOWNLOAD_TIMEOUT
        + launcher.DATASET_PREPARATION_TIMEOUT
        + launcher.CHECKPOINT_PREPARATION_TIMEOUT
    )
    assert captured["startup_timeout"] == baseline_wait + serve.startup_timeout
    assert captured["volumes"]["/draft"] is volume
    assert (
        closure["sglang_server_args"]["--speculative-draft-model-path"]
        == "/draft/dflash"
    )


@pytest.mark.parametrize(
    "mount",
    ["/checkpoints", "/checkpoints/draft", "/root", "draft", "/draft/../checkpoints"],
)
def test_serving_volumes_cannot_shadow_required_mounts(build_app, mount):
    serve = Qwen3_30B_A3B_Stitch_Serve(
        volumes={mount: modal.Volume.from_name("draft-checkpoints")}
    )
    with pytest.raises(TrainingGymConfigError, match="must be absolute"):
        build_app(recipe=Qwen3_30B_A3B_Stitch_Recipe(serve=serve))


def test_tutorial_constructs_recipe_with_metrics(monkeypatch):
    train = Mock(return_value=SimpleNamespace(checkpoint_dir="/checkpoints/test"))
    monkeypatch.setattr(TrainConfig, "train", train)
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "tutorials/disaggregated_rl.py")
    )
    recipe = namespace["recipe"]
    assert recipe.metrics.project == "training-gym"
    assert recipe.train.metrics is recipe.metrics
    train.assert_called_once()


@pytest.mark.parametrize("source_kind", ["local", "hub", "missing", "file"])
def test_checkpoint_preparation_resolves_local_and_hub_sources(
    build_app, monkeypatch, tmp_path, source_kind
):
    source = tmp_path / "source"
    if source_kind == "file":
        source.touch()
    reference = "org/model" if source_kind == "hub" else str(source)
    recipe = Qwen3_30B_A3B_Stitch_Recipe(
        train=Qwen3_30B_A3B_Stitch_Train(source_hf_checkpoint=reference)
    )
    app = build_app(recipe=recipe)
    prep = Mock()
    monkeypatch.setitem(
        sys.modules, "cookbook.miles_disagg", SimpleNamespace(prep=prep)
    )
    download = Mock(return_value=str(source))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)

    def reload_volume():
        if source_kind in {"local", "hub"}:
            source.mkdir(exist_ok=True)

    monkeypatch.setattr(modal.Volume, "reload", Mock(side_effect=reload_volume))
    prepare = inspect.unwrap(app.prepare_checkpoints.get_raw_f())
    if source_kind in {"missing", "file"}:
        with pytest.raises(FileNotFoundError, match=str(source)):
            prepare()
        prep.prepare_checkpoints.assert_not_called()
    else:
        prepare()
        assert prep.prepare_checkpoints.call_args.kwargs["source_snapshot"] == str(
            source
        )
    if source_kind == "hub":
        download.assert_called_once_with("org/model", local_files_only=False)
    else:
        download.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("preflight_fails", [False, True])
def test_tracker_identity_is_set_before_every_ray_node(
    build_app, monkeypatch, rank, preflight_fails
):
    recipe = Qwen3_30B_A3B_Stitch_Recipe(
        metrics=WandbConfig(project="project", entity="configured-team")
    )
    monkeypatch.setattr(launcher, "metric_secrets", lambda _: [])
    app = build_app(recipe=recipe, training_run_id="test-run")
    preflight = Mock(
        return_value="resolved-team",
        side_effect=RuntimeError("offline") if preflight_fails else None,
    )
    monkeypatch.setattr(launcher, "preflight_metric", preflight)
    monkeypatch.setitem(
        sys.modules,
        "cookbook.common",
        SimpleNamespace(
            hooks=Mock(),
            launch=Mock(),
            process=Mock(),
            ray_cluster=SimpleNamespace(
                get_modal_cluster_context=lambda _: (rank, "127.0.0.1", "127.0.0.1")
            ),
        ),
    )

    def start_ray(*args, **kwargs):
        assert os.environ["WANDB_RUN_ID"] == "test-run"
        assert os.environ["WANDB_ENTITY"] == (
            "configured-team" if preflight_fails else "resolved-team"
        )
        raise RuntimeError("stop at Ray startup")

    monkeypatch.setattr(launcher, "start_ray_head", start_ray)
    monkeypatch.setattr(launcher, "start_ray_worker", start_ray)
    with (
        patch.dict(os.environ),
        pytest.raises(RuntimeError, match="stop at Ray startup"),
    ):
        inspect.unwrap(app.train.get_raw_f())()
    preflight.assert_called_once()


@pytest.mark.parametrize(
    "save_interval,fails", [(10, False), (None, False), (10, True)]
)
def test_trainer_persists_result_and_metric_identity(
    build_app, monkeypatch, tmp_path, save_interval, fails
):
    from modal_training_gym.frameworks.stitch import trainer_helpers

    checkpoints = tmp_path / "checkpoints"
    served = tmp_path / "served"
    masters = tmp_path / "masters"
    served.mkdir()
    masters.mkdir()
    monkeypatch.setattr(launcher, "CHECKPOINTS_PATH", checkpoints)
    monkeypatch.setattr(launcher, "metric_secrets", lambda _: [])
    recipe = Qwen3_30B_A3B_Stitch_Recipe(
        train=Qwen3_30B_A3B_Stitch_Train(
            save=str(checkpoints), save_interval=save_interval
        ),
        served_checkpoint_path=str(served),
        bf16_checkpoint_path=str(masters),
        metrics=WandbConfig(project="project"),
    )
    app = build_app(recipe=recipe, training_run_id="test-run")
    monkeypatch.setitem(
        sys.modules,
        "cookbook.common",
        SimpleNamespace(
            hooks=Mock(),
            launch=Mock(),
            process=Mock(),
            ray_cluster=SimpleNamespace(
                get_modal_cluster_context=lambda _: (0, "127.0.0.1", "127.0.0.1")
            ),
        ),
    )
    stored = {}

    def save(run):
        stored[run.training_run_id] = run.model_dump()

    monkeypatch.setattr(TrainingRun, "save", save)
    monkeypatch.setattr(
        TrainingRun,
        "from_id",
        lambda run_id: TrainingRun.from_stored_data(stored[run_id]),
    )
    monkeypatch.setattr(launcher, "start_ray_head", Mock())
    monkeypatch.setattr(Qwen3_30B, "download", Mock())
    monkeypatch.setattr(Qwen3_30B_A3B_Stitch_Train, "download_model", Mock())
    monkeypatch.setattr(trainer_helpers, "await_gateway_ready", Mock())
    monkeypatch.setattr(launcher, "_build_train_cmd", Mock(return_value="miles"))
    monkeypatch.setattr(launcher, "save_train_result_blob", Mock())
    preflight = Mock(return_value="resolved-team")
    monkeypatch.setattr(launcher, "preflight_metric", preflight)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        Mock(side_effect=subprocess.CalledProcessError(1, "miles") if fails else None),
    )
    train = inspect.unwrap(app.train.get_raw_f())
    with patch.dict(os.environ):
        if fails:
            with pytest.raises(RuntimeError, match="miles exited 1"):
                train(modal_app_id="ap-test", rollout_endpoint_url="https://pool")
        else:
            result = train(modal_app_id="ap-test", rollout_endpoint_url="https://pool")

    reloaded = TrainingRun.from_id("test-run")
    assert reloaded.status == (
        TrainingRunStatus.FAILED if fails else TrainingRunStatus.COMPLETED
    )
    assert reloaded.metrics["entity"] == "resolved-team"
    assert reloaded.metrics["run_id"] == "test-run"
    preflight.assert_called_once()
    if fails:
        assert reloaded.source_model is None
        launcher.save_train_result_blob.assert_not_called()
    else:
        assert reloaded.checkpoint_dir == result["checkpoint_dir"]
        assert reloaded.checkpoint_dir == (
            str(checkpoints / "test-run") if save_interval else ""
        )
        assert reloaded.source_model == result["model_config"]
        assert reloaded.metrics == result["metrics"]
        if save_interval:
            assert (
                reloaded.metadata["checkpoint_location"]["checkpoints_volume_name"]
                == result["checkpoints_volume_name"]
            )
        monkeypatch.setattr(
            "modal_training_gym.common.checkpoint._list_checkpoints",
            lambda *args, **kwargs: [],
        )
        assert reloaded.model.model_name == "Qwen/Qwen3-30B-A3B"
        if save_interval:
            assert reloaded.model.model_path == result["checkpoint_dir"]
