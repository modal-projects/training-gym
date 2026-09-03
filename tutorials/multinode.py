# ---
# order: 10
# ---
#
# > **Multi-node workspace required:** This is a multi-node example. To run it,
# > your Modal workspace must have multi-node enabled. Contact
# > [support@modal.com](mailto:support@modal.com) to enable multi-node.

# # Frontier-scale training
#
# When you decide you need a frontier-scale model for your workload, you want
# the biggest hardware you can get, and runs that stay up. This tutorial trains
# [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) across 8 nodes with 8 H200s
# each (64 GPUs) using full-weight
# [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) (GSPO)
# on
# [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k).

from modal_training_gym import (
    Endpoint,
    GLM_4_7,
    HuggingFaceDataset,
    TrainConfig,
)
from modal_training_gym.train_recipes.slime_recipe import GLM_4_7_Recipe

# ## Set up
#
# Model and recipe classes carry the settings that make this run correct and
# fast. A few parameters are printed below so you can see the cluster shape.

model = GLM_4_7()


train_dataset = HuggingFaceDataset(
    "zhuzilin/dapo-math-17k",
    hf_split="train[:2000]",
    input_column="prompt",
    output_column="label",
    input_format="messages",
    always_download=True,
)
recipe = GLM_4_7_Recipe(
    eval_interval=None,
    rm_type="deepscaler",
)

print(f"training and rollout gpus colocated: {recipe.colocate}")
print(f"training nodes: {recipe.actor_num_nodes}, gpus/node: {recipe.actor_num_gpus_per_node}")
print(f"rollout gpus: {recipe.rollout_num_gpus}, rollout gpus/engine: {recipe.rollout_num_gpus_per_engine}")
print(f"parallelism: tp={recipe.tensor_model_parallel_size}, pp={recipe.pipeline_model_parallel_size}, "
      f"cp={recipe.context_parallel_size}, ep={recipe.expert_model_parallel_size}")
print(f"optimizer cpu offload: {recipe.optimizer_cpu_offload}")

# ## Kick off training

config = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=recipe,
)

run = config.launch()
print(f"run id: {run.training_run_id}")

# ## Test out the trained model
#
# Spin up an [Endpoint](https://modal.com/docs/guide/endpoints) and try a prompt.

result = run.result()
checkpoint = result.checkpoints()[-1]
print(f"checkpoint: {checkpoint.path}")

trained_deployment = Endpoint.launch(
    model, checkpoint, unauthenticated=True, recreate_if_existing=True
)
trained_deployment.wait_until_ready(timeout=45 * 60)
print(f"checkpoint deployed to {trained_deployment.url}")

msg = trained_deployment.chat(
    [
        {
            "role": "user",
            "content": (
                "Let $p$ be a prime number. Find the number of integers $n$ "
                "with $1 \\le n \\le p^2$ such that $n^{p-1} \\equiv 1 \\pmod{p^2}$."
            ),
        }
    ],
)
print(msg.get("content") or msg.get("reasoning_content") or "")
