from modal_training_gym import (
    Qwen3_6_35B,
    Qwen3_6_35B_Recipe,
)
from modal_training_gym.frameworks.slime.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


def test_qwen3_6_35b_uses_disagg_two_plus_two_b200_ep2() -> None:
    model = Qwen3_6_35B()
    recipe = Qwen3_6_35B_Recipe()

    assert isinstance(SlimeRecipe.get_base_recipe(model), Qwen3_6_35B_Recipe)
    assert recipe.gpu_type == "B200"
    assert recipe.colocate is False
    assert recipe.actor_num_nodes == 1
    assert recipe.actor_num_gpus_per_node == 2
    assert recipe.rollout_num_gpus == 2
    assert recipe.rollout_num_gpus_per_engine == 2
    assert recipe.expert_model_parallel_size == 2
    assert recipe.sglang_ep_size == 2
    assert recipe.tensor_model_parallel_size == 1
    assert recipe.gpu_allocation.actor_gpus == 2
    assert recipe.gpu_allocation.rollout_gpus == 2
    assert recipe.gpu_allocation.total_gpus == 4
    assert recipe.gpu_allocation.gpus_per_node == 4
    assert recipe.gpu_allocation.total_nodes == 1
    assert recipe.gpu_allocation.rollout_engines == 1
    assert recipe.ref_load == "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp1pp1"

    cli_args = recipe.cli_args(model=model)
    dispatcher = cli_args.index("--moe-token-dispatcher-type")
    assert cli_args[dispatcher + 1] == "alltoall"
    assert "--moe-enable-deepep" not in cli_args
    assert cli_args[cli_args.index("--sglang-ep-size") + 1] == "2"
    assert cli_args[cli_args.index("--expert-model-parallel-size") + 1] == "2"
    assert cli_args[cli_args.index("--rollout-num-gpus-per-engine") + 1] == "2"
    assert cli_args[cli_args.index("--actor-num-gpus-per-node") + 1] == "2"
    assert cli_args[cli_args.index("--rollout-num-gpus") + 1] == "2"
    attn = cli_args.index("--sglang-attention-backend")
    assert cli_args[attn + 1] == "triton"
    moe = cli_args.index("--sglang-moe-runner-backend")
    assert cli_args[moe + 1] == "triton"

    nodes, processes, conversion_args = get_checkpoint_conversion_policy(
        recipe, model=model
    )
    assert (nodes, processes) == (1, 1)
    assert "--tensor-model-parallel-size 1" in conversion_args
    assert "--pipeline-model-parallel-size 1" in conversion_args
    assert all("expert-model-parallel-size" not in arg for arg in conversion_args)
