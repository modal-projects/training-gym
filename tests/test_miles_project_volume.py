from unittest.mock import Mock

import pytest

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.launcher_helpers import run_prepare_dataset
from modal_training_gym.common.models import Qwen3_5_4B
from modal_training_gym.frameworks.miles import launcher
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


class ProjectRecipe(MilesRecipe):
    @staticmethod
    def _resolve_data_paths(dataset):
        return "/checkpoints/data/train.jsonl", {"eval": "/checkpoints/data/eval.jsonl"}


@pytest.fixture
def build_app(monkeypatch):
    for name in ("hf_secrets", "proxy_auth_secrets", "metric_secrets"):
        monkeypatch.setattr(launcher, name, lambda *a, **kw: [])
    monkeypatch.setattr(launcher, "resolve_caller_context", lambda: (None, None))
    dataset = HuggingFaceDataset(
        hf_repo="org/data", input_column="prompt", output_column="answer"
    )

    def build(recipe):
        return launcher.build_miles_app(
            training_run_id="test", miles=recipe, model=Qwen3_5_4B(), dataset=dataset
        )

    return build


def test_project_volume_is_mounted_once_for_training_and_preparation(
    build_app, monkeypatch
):
    app = build_app(ProjectRecipe(project_volume_name="shared-project"))
    train = app.registered_functions["train"]
    prepare = app.registered_functions["prepare_dataset"]
    assert set(train.spec.volumes) == {"/root/.cache/huggingface", "/checkpoints"}
    assert set(prepare.spec.volumes) == {"/root/.cache/huggingface", "/checkpoints"}
    assert prepare.spec.volumes["/checkpoints"] is train.spec.volumes["/checkpoints"]
    prepare_data = Mock()
    monkeypatch.setattr(launcher, "run_prepare_dataset", prepare_data)
    prepare.get_raw_f()()
    assert prepare_data.call_args.args[2] is ProjectRecipe._resolve_data_paths
    assert prepare_data.call_args.kwargs == {"clear_data_dir": False}


def test_default_recipe_keeps_separate_data_and_checkpoint_mounts(build_app):
    app = build_app(MilesRecipe())
    assert set(app.registered_functions["train"].spec.volumes) == {
        "/root/.cache/huggingface",
        "/data",
        "/checkpoints",
    }


def test_forced_preparation_preserves_other_files_in_shared_directory(tmp_path):
    prompt = tmp_path / "train.jsonl"
    prompt.write_text("old data")
    checkpoint = tmp_path / "model" / "release" / "__0_0.distcp"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("weights")
    dataset = Mock(always_prepare=True)
    dataset.prepare.side_effect = lambda path, evals: prompt.write_text("new data")

    run_prepare_dataset(
        dataset, Mock(), lambda _: (str(prompt), {}), clear_data_dir=False
    )

    assert prompt.read_text() == "new data"
    assert checkpoint.read_text() == "weights"


@pytest.mark.parametrize(
    "path", ["/data/train.jsonl", "/checkpoints/../outside/train.jsonl"]
)
def test_project_data_must_be_inside_its_mount(build_app, monkeypatch, path):
    monkeypatch.setattr(
        ProjectRecipe, "_resolve_data_paths", staticmethod(lambda ds: (path, {}))
    )
    with pytest.raises(ValueError, match="dataset paths must be below"):
        build_app(ProjectRecipe(project_volume_name="shared-project"))
