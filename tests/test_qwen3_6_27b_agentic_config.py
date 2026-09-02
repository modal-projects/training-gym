from modal_training_gym import (
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe,
    Qwen3_6_27B_Recipe_Agentic,
)
from modal_training_gym.common.trackio import TrackioConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def test_qwen3_6_27b_agentic_inherits_the_preset_and_pins_the_fork() -> None:
    recipe = Qwen3_6_27B_Recipe_Agentic()

    assert isinstance(recipe, Qwen3_6_27B_Recipe)
    assert recipe.slime_git_repository == "https://github.com/modal-projects/slime.git"
    assert recipe.slime_git_revision == "ba324bebdd3a3cbfc1946b58404a012ad607f38b"
    assert recipe.data_volume_name == "slime-data"
    assert recipe.slime_model_script == "scripts/models/qwen3.5-27B.sh"
    assert recipe.sglang_speculative_algorithm == "EAGLE"


def test_qwen3_6_27b_agentic_does_not_displace_the_base_recipe() -> None:
    recipe = SlimeRecipe.get_base_recipe(Qwen3_6_27B())

    assert type(recipe) is Qwen3_6_27B_Recipe


def test_qwen3_6_27b_agentic_defaults_to_a_trackio_project() -> None:
    recipe = Qwen3_6_27B_Recipe_Agentic()

    assert isinstance(recipe.metrics, TrackioConfig)
    assert recipe.metrics.project == "agentic-harbor"
    assert recipe.metrics.provider == "trackio"

    # Trackio reaches slime through its native W&B flags; the image-level
    # adapter is what routes them to Trackio.
    fields = recipe._fields(model=Qwen3_6_27B())
    assert fields["use_wandb"] is True
    assert fields["wandb_project"] == "agentic-harbor"
    assert "metrics" not in fields


def test_qwen3_6_27b_agentic_emits_fork_rollout_configuration() -> None:
    recipe = Qwen3_6_27B_Recipe_Agentic()

    assert recipe.total_nodes == 6
    assert recipe.gpu_allocation.total_gpus == 48
    assert recipe.colocate is False
    assert recipe.capture_trace is True
    assert recipe.pipeline_model_parallel_size == 2
    assert recipe.conversion_pipeline_model_parallel_size == 2
    assert recipe.decoder_last_pipeline_num_layers == 30
    assert recipe.context_parallel_size == 2
    assert recipe.sglang_server_concurrency == 32
    assert recipe.rollout_batch_size == 32
    assert recipe.global_batch_size == (
        recipe.rollout_batch_size * recipe.n_samples_per_prompt
    )
    assert recipe.eval_interval is None
    assert recipe.max_tokens_per_gpu == 16384
    assert recipe.log_probs_chunk_size == 128
    assert recipe.environment["NCCL_RAS_ENABLE"] == "0"
    assert recipe.environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert recipe.extra_config["custom_generate_function_path"] == (
        "agentic_rl.generate.generate"
    )
    assert recipe.extra_config["training_gym_custom_rollout_log_function_path"] == (
        "agentic_rl.metrics.log_rollout_data"
    )
    fields = recipe._fields(model=Qwen3_6_27B())
    assert fields["sglang_server_concurrency"] == 32
    assert fields["max_tokens_per_gpu"] == 16384
    assert fields["log_probs_chunk_size"] == 128
    assert fields["save_debug_rollout_data"].endswith("/rollout_{rollout_id}.pt")
    assert "data_volume_name" not in fields
