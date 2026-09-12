from modal_training_gym import (
    Qwen3_8_27B,
    Qwen3_8_27B_Recipe,
)
from modal_training_gym.frameworks.slime.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def test_qwen3_8_27b_adapts_current_slime_qwen3_5_recipe() -> None:
    model = Qwen3_8_27B()
    recipe = Qwen3_8_27B_Recipe()

    assert recipe.slime_model_script == "scripts/models/qwen3.5-27B.sh"
    assert recipe.hf_checkpoint == model.model_name == "Qwen/Qwen3.8-27B"
    assert recipe.ref_load == "/checkpoints/Qwen3.8-27B_torch_dist_tp2pp1"
    assert recipe.actor_num_nodes == 1
    assert recipe.actor_num_gpus_per_node == 2
    assert recipe.tensor_model_parallel_size == 2
    assert getattr(recipe, "pipeline_model_parallel_size", 1) == 1
    assert recipe.context_parallel_size == 1
    assert getattr(recipe, "decoder_last_pipeline_num_layers", None) is None
    assert recipe.rollout_num_gpus_per_engine == 2
    assert recipe.sglang_mem_fraction_static == 0.75
    assert recipe.sglang_speculative_algorithm == "EAGLE"
    assert recipe.use_kl_loss is False
    assert recipe.calculate_per_token_loss is True
    assert recipe.eval_interval is None
    assert recipe.max_tokens_per_gpu == 8192
    assert recipe.memory == (128, 2_097_152)

    fields = recipe._fields(model=model)
    assert fields["hf_checkpoint"] == model.model_name
    assert "num_layers" not in fields
    assert "spec" not in fields
    cli_args = recipe.cli_args(model=model)
    prefill_backend = cli_args.index("--sglang-cuda-graph-backend-prefill")
    assert cli_args[prefill_backend + 1] == "disabled"

    nodes, processes, conversion_args = get_checkpoint_conversion_policy(
        recipe, model=model
    )
    assert (nodes, processes) == (1, 2)
    assert "--tensor-model-parallel-size 2" in conversion_args
    assert "--pipeline-model-parallel-size 1" in conversion_args


def test_qwen3_8_27b_is_the_registered_base_recipe() -> None:
    recipe = SlimeRecipe.get_base_recipe(Qwen3_8_27B())

    assert isinstance(recipe, Qwen3_8_27B_Recipe)
    assert recipe.gpu_allocation.total_gpus == 2
