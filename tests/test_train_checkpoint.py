from __future__ import annotations

import inspect

from modal_training_gym.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    apply_train_checkpoint,
)
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_5_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.frameworks.miles.launcher import build_miles_app
from modal_training_gym.frameworks.slime.launcher import build_slime_app
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

_RECIPE_KW = dict(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    rollout_num_gpus_per_engine=1,
    num_rollout=1,
    rollout_batch_size=16,
    rollout_max_response_len=4096,
    rollout_temperature=1.0,
    save_interval=10,
)


def _checkpoint(
    *,
    checkpoint_type: CheckpointType,
    name: str,
    path: str,
) -> Checkpoint:
    return Checkpoint(
        checkpoint_type=checkpoint_type,
        name=name,
        path=path,
        timestamp=1.0,
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )


def _config(checkpoint: Checkpoint, recipe=None) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_5_4B(),
        dataset=HuggingFaceDataset(
            hf_repo="some/dataset",
            input_column="prompt",
            output_column="answer",
        ),
        recipe=recipe or SlimeRecipe(**_RECIPE_KW),
        checkpoint=checkpoint,
        merge_model_recipe=False,
    )


def test_megatron_checkpoint_sets_load_and_keeps_hub_hf_checkpoint() -> None:
    checkpoint = _checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_0000009",
        path="/checkpoints/run/iter_0000009",
    )
    config = _config(checkpoint)
    recipe = config._prepare_recipe()
    fields = recipe._fields(model=config.model)

    assert recipe.load == "/checkpoints/run"
    assert fields["load"] == "/checkpoints/run"
    assert fields["hf_checkpoint"] == config.model.model_name == "Qwen/Qwen3.5-4B"
    assert config.model.model_path is None

    release = SlimeRecipe(**_RECIPE_KW)
    apply_train_checkpoint(
        release,
        Qwen3_5_4B(),
        _checkpoint(
            checkpoint_type=CheckpointType.megatron,
            name="release",
            path="/checkpoints/run/release",
        ),
    )
    assert release.load == "/checkpoints/run"


def test_hf_checkpoint_sets_sibling_load_root() -> None:
    checkpoint = _checkpoint(
        checkpoint_type=CheckpointType.hf,
        name="iter_0000009_hf",
        path="/checkpoints/run/iter_0000009_hf",
    )
    config = _config(checkpoint)
    recipe = config._prepare_recipe()
    fields = recipe._fields(model=config.model)

    assert recipe.load == "/checkpoints/run"
    assert fields["load"] == "/checkpoints/run"
    assert fields["hf_checkpoint"] == checkpoint.path
    assert config.model.model_path is None


def test_standalone_hf_checkpoint_leaves_load_unset() -> None:
    checkpoint = _checkpoint(
        checkpoint_type=CheckpointType.hf,
        name="Qwen3.5-4B",
        path="Qwen/Qwen3.5-4B",
    )
    recipe = SlimeRecipe(**_RECIPE_KW)
    apply_train_checkpoint(recipe, Qwen3_5_4B(), checkpoint)
    assert recipe.load == ""
    assert recipe.hf_checkpoint == checkpoint.path


def test_explicit_load_and_hf_checkpoint_are_preserved() -> None:
    checkpoint = _checkpoint(
        checkpoint_type=CheckpointType.hf,
        name="iter_0000009_hf",
        path="/checkpoints/run/iter_0000009_hf",
    )
    recipe = SlimeRecipe(
        **_RECIPE_KW, load="/checkpoints/other", hf_checkpoint="other/model"
    )
    apply_train_checkpoint(recipe, Qwen3_5_4B(), checkpoint)
    assert recipe.load == "/checkpoints/other"
    assert recipe.hf_checkpoint == "other/model"
    assert recipe._fields(model=Qwen3_5_4B())["hf_checkpoint"] == "other/model"


def test_miles_megatron_checkpoint_sets_load_keeps_hub_hf_checkpoint() -> None:
    checkpoint = _checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_0000009",
        path="/checkpoints/run/iter_0000009",
    )
    config = _config(checkpoint, recipe=MilesRecipe())
    recipe = config._prepare_recipe()
    fields = recipe._fields(model=config.model)

    assert recipe.load == "/checkpoints/run"
    assert fields["load"] == "/checkpoints/run"
    assert fields["hf_checkpoint"] == "Qwen/Qwen3.5-4B"
    assert config.model.model_path is None


def test_builders_leave_hub_hf_checkpoint_and_unset_model_path() -> None:
    checkpoint = _checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_0000009",
        path="/checkpoints/run/iter_0000009",
    )
    config = _config(checkpoint)
    recipe = config._prepare_recipe()
    fields = recipe._fields(model=config.model)

    assert config.model.model_path is None
    assert fields["hf_checkpoint"] == "Qwen/Qwen3.5-4B"
    assert "model.model_path = checkpoint.path" not in inspect.getsource(
        build_slime_app
    )
    assert "model.model_path = checkpoint.path" not in inspect.getsource(
        build_miles_app
    )
