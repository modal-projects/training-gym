# ---
# order: 2
# ---
#
# # Training multi-turn coding agents
#
# This tutorial trains [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on
# [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).
# During rollouts, the agent inspects repositories, edits code, and runs commands
# in a [Modal Sandbox](https://modal.com/docs/guide/sandboxes) via 
# [Harbor](https://docs.harborframework.com/).

import json

from pathlib import Path
from uuid import uuid4

from modal_training_gym import (
    DatasetConfig,
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe_Agentic,
    TrainConfig,
)

# ## Get the dataset
#
# We must first convert SWE-rebench into Harbor tasks, and split/sample
# the data to create balanced, repository-disjoint train/eval sets.
# Since this is verbose, we have a separate preprocessing script.
# 
# Run with:
#
# ```bash
# uv run -m tutorials.coding_agent.dataset prepare --limit 100
# uv run -m tutorials.coding_agent.dataset prepare
# ```

DATASET_ROOT = "swe_rebench_v2"
DATA_ROOT = Path("/data") / DATASET_ROOT

TRAIN_SUBSET = "train-300"
EVAL_SUBSETS = ("eval",)

class AgentTaskDataset(DatasetConfig):
    def __init__(self, path: Path):
        self.path = path

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def apply_chat_template(self) -> bool:
        return False

    def rows(self):
        with self.path.open() as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

# ## Start training
# 
# With the [Qwen3_6_27B_Recipe_Agentic](https://gym.modal.dev/reference/qwen3_6_27b_recipe_agentic),
# recipe class, it's simply too easy.

RUN_NAME = f"coding-agent-{uuid4().hex}"


config = TrainConfig(
    model=Qwen3_6_27B(),
    dataset=AgentTaskDataset(DATA_ROOT / f"{TRAIN_SUBSET}.jsonl"),
    recipe=Qwen3_6_27B_Recipe_Agentic(
        num_rollout=500,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        rollout_max_response_len=8192,
        eval_max_response_len=8192,
        eval_interval=5,
        n_samples_per_eval_prompt=1,
        save_interval=5,
        eval_config={
            "defaults": {
                "n_samples_per_eval_prompt": 1,
                "temperature": 0.6,
                "top_p": 1.0,
            },
            "datasets": [
                {
                    "name": subset,
                    "path": str(DATA_ROOT / f"{subset}.jsonl"),
                    "metadata_overrides": {"eval_dataset": subset},
                }
                for subset in EVAL_SUBSETS
            ],
        },
        save_debug_rollout_data=(
            f"/checkpoints/agentic_rollout_dumps/{RUN_NAME}/rollout_{{rollout_id}}.pt"
        ),
    )
)

run = config.launch()
print(f"run id: {run.training_run_id}")