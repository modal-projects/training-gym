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
    Qwen3_5_4B_Recipe,
    TrainConfig,
    TrainingGroup,
)

# ## Define the training base
#
# We'll start by creating the shared model, dataset, and base training recipe
# all sweep runs will share. To stay focused on how to do sweeps, we'll keep
# this code minimal and only tune a few parameters, but the sky's the limit!

model = Qwen3_5_4B()

class MathDataset(HuggingFaceDataset):
    hf_repo = "zhuzilin/dapo-math-17k"
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True

train_dataset = MathDataset(hf_split="train[:2000]")

base = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_5_4B_Recipe(
        eval_interval=None,
        rollout_num_gpus=8,
        num_rollout=15,
        rollout_max_response_len=8192,
        global_batch_size=32,
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
        "recipe.lr": [5e-7, 5e-6],
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
