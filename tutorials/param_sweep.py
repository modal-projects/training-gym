# ---
# order: 7
# ---
#
# # Massively-parallel hyperparameter sweeps
#
# When tuning RL runs, finding the optimal set of hyperparameters is time-consuming
# and error-prone if not properly guided or documented. This is made a first-class
# operation in the Gym so you can move faster and spend less.

from modal_training_gym import (
    HuggingFaceDataset,
    Qwen3_5_4B,
    SlimeRecipe,
    TrainConfig,
    TrainingGroup,
)

# ## Define the training base
#
# We'll start by creating the shared model, dataset, and base training recipe
# all sweep runs will share. To stay focused on how to do sweeps, we'll keep
# this code minimal and only tune a few parameters, but the sky's the limit!

model = Qwen3_5_4B()

train_dataset = HuggingFaceDataset(
    "zhuzilin/dapo-math-17k",
    hf_split="train[:2000]",
    input_column="prompt",
    output_column="label",
    apply_chat_template=True,
    always_download=True,
)

base = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=SlimeRecipe(
        gpu_type="H100",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        colocate=True,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        num_rollout=15,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        rollout_max_response_len=8192,
        rollout_temperature=1.0,
        global_batch_size=32,
        lr=1e-6,
        advantage_estimator="grpo",
        use_kl_loss=False,
        kl_coef=0.0,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=9216,
        sglang_mem_fraction_static=0.75,
        save_interval=10,
        rm_type="dapo",
    ),
)

# ## Create the sweep
#
# We'll pass in the parameters we wish to test, then preview the set of
# runs the grid search will kick off once we're ready.

group = TrainingGroup(
    base=base,
    grid={
        "recipe.lr": [1e-6, 5e-6],
        "recipe.rollout_temperature": [0.8, 1.0],
    },
)
configs = group.get_train_configs()
print(f"{len(configs)} runs in group {group.group_id}:")
for cfg in configs:
    print(
        f"- lr={cfg.recipe.lr:<8}, temp={cfg.recipe.rollout_temperature}"
    )

# ## Launch it!
#
# Once it all looks good, `.launch()` it!

launches = group.launch(prepare_inputs=True)
print(f"group {group.group_id}: {len(launches)} runs launched")
for launch in launches:
    print(
        f"- {launch.training_run_id}, app={launch.modal_app_id}, group_id={launch.group_id}"
    )
if group.failures:
    for overrides, err in group.failures:
        print(f"- FAILED {overrides}: {err}")

results = []
for launch in launches:
    result = launch.result()
    results.append(result)
    print(f"completed {result.training_run_id} (group_id={result.group_id})")

print(f"group {group.group_id}: {len(results)} runs completed")
