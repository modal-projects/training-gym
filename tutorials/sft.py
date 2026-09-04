# ---
# order: 11
# ---
#
# # Supervised fine-tuning
#
# SFT teaches a model from complete example conversations. It computes loss on
# assistant responses directly, without a reward function or inference server.

from modal_training_gym import Qwen3_4B, Qwen3_4B_Recipe, SFTDataset, TrainConfig

# Use `messages_column` for existing OpenAI-style conversations. For separate
# prompt and answer columns, use `input_column` and `output_column` instead.
dataset = SFTDataset(
    hf_repo="HuggingFaceH4/no_robots",
    messages_column="messages",
    n_rows=8,
)

recipe = Qwen3_4B_Recipe(
    workload_type="sft",
    num_rollout=1,
    rollout_batch_size=8,
    global_batch_size=8,
    lr=1e-5,
    save_interval=1,
)

result = TrainConfig(
    model=Qwen3_4B(),
    dataset=dataset,
    recipe=recipe,
).train()

print(result.checkpoints())
