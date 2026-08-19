"""Projector-only training: the path from a recipe field to miles' CLI and data.

The container-side halves (provider, forward merge, checkpoint hook) need torch
and a GPU, so what is pinned here is the wiring the launcher owns: the spec
reaching the containers, the flags that make the run supervised and engine-free,
the guards against configurations that silently train nothing, and the dataset
columns miles reads.
"""

import json

import pytest
from pydantic import ValidationError

from modal_training_gym import (
    GLM_5_2,
    GLM_5_2_5Layer,
    EmbeddingProjectorDataset,
    GLM_5_2_5Layer_Projector_Recipe,
    GLM_5_2_Projector_Recipe,
    ProjectorSpec,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.frameworks.miles.projector_config import (
    ARGS_KEY,
    PROVIDER_PATH,
    ROLLOUT_PATH,
    SAVE_HOOK_PATH,
    from_miles_args,
    should_save_projector,
)
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def _flags(args: list[str]) -> dict[str, str]:
    return {
        args[i]: (
            args[i + 1]
            if i + 1 < len(args) and not args[i + 1].startswith("--")
            else ""
        )
        for i in range(len(args))
        if args[i].startswith("--")
    }


def _dataset() -> EmbeddingProjectorDataset:
    return EmbeddingProjectorDataset(
        rows=[
            {
                "messages": [
                    {"role": "user", "content": "protein?"},
                    {"role": "assistant", "content": "kinase"},
                ],
                "embeddings": [[0.0] * 4, [1.0] * 4],
                "positions": [2, 3],
            }
        ]
    )


def test_spec_reaches_containers_through_extra_config():
    recipe = GLM_5_2_5Layer_Projector_Recipe(
        projector=ProjectorSpec(input_dim=4, hidden_dim=8, num_layers=1)
    )
    payload = recipe.extra_config[ARGS_KEY]
    assert payload["input_dim"] == 4 and payload["num_layers"] == 1
    # miles sets every extra_config key on args, which is how the provider and
    # the checkpoint hook read the spec back without new CLI flags.
    spec = from_miles_args(_FakeArgs(payload))
    assert spec.input_dim == 4
    assert spec.embeddings_key == "projector_embeddings"


class _FakeArgs:
    def __init__(self, payload: dict) -> None:
        setattr(self, ARGS_KEY, payload)


def test_missing_spec_is_a_clear_error():
    with pytest.raises(ValueError, match=ARGS_KEY):
        from_miles_args(_FakeArgs.__new__(_FakeArgs))


def test_recipe_emits_supervised_engine_free_flags():
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    flags = _flags(recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer()))

    assert flags["--custom-model-provider-path"] == PROVIDER_PATH
    assert flags["--rollout-function-path"] == ROLLOUT_PATH
    assert flags["--loss-type"] == "sft_loss"
    assert "--debug-train-only" in flags
    assert "--disable-compute-advantages-and-returns" in flags
    # No LoRA anywhere: the whole point of freezing the base instead.
    assert "--lora-rank" not in flags
    # The projector spec travels in the YAML config, never as a flag.
    assert "--projector" not in flags


def test_save_hook_runs_through_the_gyms_phase_reporting_wrapper():
    """Dashboard phase/substep timing must survive the projector's own hook."""
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    flags = _flags(recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer()))
    assert flags["--custom-megatron-before-train-step-hook-path"].startswith(
        "modal_training_gym.frameworks.miles.phase_reporting"
    )
    assert (
        recipe.extra_config["training_gym_custom_megatron_before_train_step_hook_path"]
        == SAVE_HOOK_PATH
    )


def test_lora_is_rejected():
    # Pydantic wraps the validator's error; the message is what a user reads.
    with pytest.raises(ValidationError, match="lora_rank must be unset"):
        GLM_5_2_Projector_Recipe(lora_rank=16)


def test_pipeline_parallelism_is_rejected():
    """PP>1 would give later stages an optimizer over an empty parameter set."""
    recipe = GLM_5_2_Projector_Recipe(pipeline_model_parallel_size=2)
    with pytest.raises(TrainingGymConfigError, match="pipeline_model_parallel_size=1"):
        recipe.validate_model_parallelism(GLM_5_2())


def test_context_parallelism_is_rejected():
    """CP shards the sequence its own way; the merge rebases only TP/SP."""
    recipe = GLM_5_2_Projector_Recipe(context_parallel_size=2)
    with pytest.raises(TrainingGymConfigError, match="context_parallel_size=1"):
        recipe.validate_model_parallelism(GLM_5_2())


def test_megatron_checkpoint_loading_is_rejected():
    """The projector is a submodule, so base state dicts lack its keys."""
    with pytest.raises(ValidationError, match="resumes through projector.load"):
        GLM_5_2_Projector_Recipe(load="/checkpoints/glm")
    with pytest.raises(ValidationError, match="resumes through projector.load"):
        GLM_5_2_Projector_Recipe(ref_load="/checkpoints/glm")


def test_a_finished_run_always_leaves_a_projector_checkpoint():
    """The run's last optimizer step saves even when the interval misses it."""
    assert [
        s for s in range(1, 11) if should_save_projector(s, 10, save_interval=4)
    ] == [4, 8, 10]
    # The shipped defaults: one save, at the end of the run.
    recipe = GLM_5_2_Projector_Recipe()
    total = (
        recipe.num_rollout
        * recipe.rollout_batch_size
        * recipe.n_samples_per_prompt
        // recipe.global_batch_size
    )
    assert [
        s
        for s in range(1, total + 1)
        if should_save_projector(s, total, recipe.projector.save_interval)
    ] == [total]
    # An interval of 0 turns periodic saves off, but not the final one.
    assert not should_save_projector(3, total, 0)
    assert should_save_projector(total, total, 0)


def test_synthetic_validation_data_is_regenerated_per_run():
    """The on-volume path is class-derived, so stale rows would be reused."""
    assert EmbeddingProjectorDataset.synthetic(n_rows=2, input_dim=4).always_prepare


def test_disk_reservation_survives_caller_supplied_train_kwargs():
    recipe = GLM_5_2_Projector_Recipe(train_function_kwargs={"timeout": 60})
    assert recipe.train_function_kwargs["timeout"] == 60
    assert recipe.train_function_kwargs["ephemeral_disk"] > 0


def test_base_recipe_lookup():
    assert isinstance(MilesRecipe.get_base_recipe(GLM_5_2()), GLM_5_2_Projector_Recipe)
    assert isinstance(
        MilesRecipe.get_base_recipe(GLM_5_2_5Layer()),
        GLM_5_2_5Layer_Projector_Recipe,
    )


def test_dataset_writes_embeddings_into_the_metadata_column(tmp_path):
    dataset = _dataset()
    out = str(tmp_path / "train.jsonl")
    dataset.prepare(out)
    dataset.validate_prepared(out)
    row = json.loads(open(out).readline())
    assert row["metadata"]["projector_positions"] == [2, 3]
    assert len(row["metadata"]["projector_embeddings"]) == 2
    # The conversation stays a message list: the SFT loss mask is built by
    # splitting it, which a rendered string would not allow.
    assert isinstance(row["messages"], list)


def test_dataset_rejects_mismatched_embeddings_and_positions():
    with pytest.raises(TrainingGymConfigError, match="position"):
        EmbeddingProjectorDataset(
            rows=[
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "embeddings": [[0.0], [1.0]],
                    "positions": [1],
                }
            ]
        )
