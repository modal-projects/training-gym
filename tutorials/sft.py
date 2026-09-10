# ---
# order: 11
# ---
#
# # Supervised fine-tuning
#
# SFT teaches a model from complete example conversations. It computes loss on
# assistant responses directly.

from modal_training_gym import Qwen3_4B, Qwen3_4B_Recipe, SFTDataset, TrainConfig

# Use `messages_column` for existing OpenAI-style conversations. For separate
# prompt and answer columns, use `input_column` and `output_column` instead.
dataset = SFTDataset(
    hf_repo="HuggingFaceH4/no_robots",
    messages_column="messages",
    hf_split="train[:1024]",
)

# SFT uses global_batch_size for both the data batch and the optimizer step.
# num_nodes and shuffle can override the model recipe's defaults too.
recipe = Qwen3_4B_Recipe(
    training_type="sft",
    num_gpus_per_node=1,
    num_steps=5,
    global_batch_size=128,
    lr=1e-5,
    save_interval=5,
)

result = TrainConfig(
    model=Qwen3_4B(),
    dataset=dataset,
    recipe=recipe,
).train()

print(result.checkpoints())

# Use `training-gym run get <run-id> --verbose` to inspect the loss history.
# Loss is reported after each optimizer step. Evaluate saved checkpoints on held-out prompts before
# treating a falling training loss as improved model quality.
#
# To continue training, keep the original model and set `recipe.load` to the
# saved `/checkpoints/<run-id>` directory. Set `num_steps` to the desired
# total step count, not the number of additional steps. When extending that
# horizon, also set `extra_config={"override_opt_param_scheduler": True}`
# which loads training state.
