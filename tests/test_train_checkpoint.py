from __future__ import annotations

import inspect

import pytest

from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.errors import TrainingGymConfigError
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


def _checkpoint(checkpoint_type: CheckpointType) -> Checkpoint:
    suffix = "_hf" if checkpoint_type == CheckpointType.hf else ""
    return Checkpoint(
        checkpoint_type=checkpoint_type,
        name=f"iter_0000009{suffix}",
        path=f"/checkpoints/run/iter_0000009{suffix}",
        timestamp=1.0,
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )


def _config(recipe, checkpoint_type: CheckpointType) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_5_4B(),
        dataset=HuggingFaceDataset(
            hf_repo="some/dataset",
            input_column="prompt",
            output_column="answer",
        ),
        recipe=recipe,
        checkpoint=_checkpoint(checkpoint_type),
    )


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(SlimeRecipe(**_RECIPE_KW), id="slime"),
        pytest.param(MilesRecipe(), id="miles"),
    ],
)
def test_megatron_checkpoint_only_sets_load(recipe) -> None:
    config = _config(recipe, CheckpointType.megatron)
    prepared = config._prepare_recipe()
    fields = prepared._fields(model=config.model)

    assert prepared is not recipe
    assert prepared.load == "/checkpoints/run"
    assert fields["hf_checkpoint"] == "Qwen/Qwen3.5-4B"
    assert config.model.model_path is None
    assert recipe.load == ""


def test_checkpoint_wins_over_recipe_load() -> None:
    config = _config(
        SlimeRecipe(**_RECIPE_KW, load="/checkpoints/other"),
        CheckpointType.megatron,
    )

    assert config._prepare_recipe().load == "/checkpoints/run"


def test_config_summary_records_resume_without_mutating_recipe() -> None:
    config = _config(SlimeRecipe(**_RECIPE_KW), CheckpointType.megatron)

    summary = config._build_config_summary("run-id")

    assert summary["recipe"]["load"] == "/checkpoints/run"
    assert summary["recipe"]["hf_checkpoint"] == "Qwen/Qwen3.5-4B"
    assert config.recipe.load == ""
    assert config.model.model_path is None


def test_hf_export_is_not_a_training_resume_checkpoint() -> None:
    config = _config(SlimeRecipe(**_RECIPE_KW), CheckpointType.hf)

    with pytest.raises(
        TrainingGymConfigError,
        match="Hugging Face exports are serving artifacts",
    ):
        config._prepare_recipe()


def test_launchers_do_not_replace_model_path_with_checkpoint() -> None:
    assert "model.model_path = checkpoint.path" not in inspect.getsource(
        build_slime_app
    )
    assert "model.model_path = checkpoint.path" not in inspect.getsource(
        build_miles_app
    )


def test_slime_conversion_uses_wrapper_with_expected_environment() -> None:
    source = inspect.getsource(build_slime_app)

    assert (
        "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
        in source
    )
    assert (
        'convert_script = f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"'
        not in source
    )
    assert (
        'if any(arg.startswith("--pipeline-model-parallel-size ") '
        "for arg in extra_args):\n"
        '            env["SKIP_PP_AUTOINFLATE"] = "1"'
    ) in source
    assert 'if num_nodes > 1:\n            env["SKIP_RELEASE_RENAME"] = "1"' in source


def test_miles_conversion_uses_wrapper_with_expected_environment() -> None:
    source = inspect.getsource(build_miles_app)

    assert (
        "modal_training_gym.frameworks.miles.modal_helpers.convert_hf_to_torch_dist"
        in source
    )
    assert (
        'convert_script = f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"'
        not in source
    )
    assert (
        'if any(arg.startswith("--pipeline-model-parallel-size ") '
        "for arg in extra_args):\n"
        '            env["CONVERT_KEEP_PP1"] = "1"'
    ) in source
    assert 'if num_nodes > 1:\n            env["SKIP_RELEASE_RENAME"] = "1"' in source
