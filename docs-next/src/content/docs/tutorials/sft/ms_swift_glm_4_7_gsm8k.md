---
title: "GLM-4.7 LoRA SFT on GSM8K (Megatron)"
description: "GLM-4.7 LoRA SFT on GSM8K (Megatron)"
---

# GLM-4.7 LoRA SFT on GSM8K with ms-swift on Modal

**What ms-swift is.**
[ms-swift](https://github.com/modelscope/ms-swift) is ModelScope's
end-to-end fine-tuning toolkit. It supports PEFT methods (LoRA /
QLoRA / DoRA) and full finetuning across HuggingFace and
**Megatron-LM** backends. The Megatron backend is what makes it
interesting at large scale — it gives you 4-D parallelism (TP / PP /
EP / CP) for models that don't fit on one GPU under HF's data
parallelism alone.

**What this tutorial does.** LoRA SFT of GLM-4.7 (a large MoE model)
on GSM8K, on 4 nodes × 8×H100 (32 GPUs). The interesting piece is
the parallelism split: TP=2, EP=4, PP=4, CP=1 — tensor parallel
across pairs of GPUs, 4-way expert parallel for the MoE layers,
4-stage pipeline parallel for the transformer blocks. Under the
hood this launches `megatron sft` via `torchrun` on each clustered
node. For the shared primitives (`DatasetConfig`, `Model`, 3-stage
pipeline) see [`quickstart`](../../intro/quickstart/quickstart.ipynb).

**What you'll need.**
- Access to Modal's multi-node training preview (4 × 8×H100).
- `wandb` Modal secret.

**What to watch.** W&B project `glm-4-7-sft`. Watch `train/loss`
and `train/grad_norm`; LoRA converges quickly on GSM8K so expect
loss to fall off within the first few hundred iters.

```python
import modal

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import GLM_4_7
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.ms_swift import (
    MsSwiftConfig,
    MsSwiftFrameworkConfig,
)
from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH
```

## Define the dataset

ms-swift reads a JSONL file where each line is a chat-format object:
`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.
`prepare()` converts GSM8K's `(question, answer)` columns into that
shape and writes it under the HF cache volume so both download and
dataset prep share the same mount.

```python
class GSM8KDataset(DatasetConfig):
    def __init__(
        self,
        hf_cache_root,
        data_folder="gsm8k",
        hf_dataset="openai/gsm8k",
        split="train",
        input_col="question",
        output_col="answer",
    ):
        self._hf_cache_root = str(hf_cache_root)
        self._data_folder = data_folder
        self._hf_dataset = hf_dataset
        self._split = split
        self._input_col = input_col
        self._output_col = output_col
        self.prompt_data = (
            f"{self._hf_cache_root}/msswift-data/{data_folder}/training.jsonl"
        )

    def prepare(self):
        import json
        import os

        from datasets import load_dataset

        output_dir = f"{self._hf_cache_root}/msswift-data/{self._data_folder}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            ds = load_dataset(self._hf_dataset, split=self._split)
        except ValueError:
            ds = load_dataset(self._hf_dataset, "main", split=self._split)

        out_path = f"{output_dir}/training.jsonl"
        with open(out_path, "w") as f:
            for row in ds:
                messages = [
                    {"role": "user", "content": row[self._input_col]},
                    {"role": "assistant", "content": row[self._output_col]},
                ]
                f.write(json.dumps({"messages": messages}) + "\n")
        print(f"Wrote {out_path}")
```

## Define the experiment

`MsSwiftFrameworkConfig` holds ms-swift-specific knobs; the launcher
forwards them to `megatron sft` as `--flag value` args. ms-swift
uses underscore-style names (e.g. `--tensor_model_parallel_size 2`)
and expects booleans as strings.

### Parallelism (32 GPUs = 4 nodes × 8 H100)

GLM-4.7 is a large MoE — it needs all four axes:

| Axis | Setting | Why |
|---|---|---|
| Tensor (TP)   | 2 | Shard individual weight matrices across 2 GPUs |
| Expert (EP)   | 4 | Spread MoE experts across 4 GPUs |
| Pipeline (PP) | 4 | 4-stage pipeline over transformer blocks |
| Context (CP)  | 1 | No sequence-dim parallelism at this context length |
| Data (implicit) | 32/(2·4·4·1) = 1 | No data parallelism at this scale |

`sequence_parallel=True` saves activation memory in the TP groups.

### LoRA

- `tuner_type="lora"` + `lora_rank=128`, `lora_alpha=32` — higher
  rank than the usual 8–16; GLM-4.7 is large enough that a bigger
  rank pays for itself without running out of memory.
- `merge_lora=False` — keep the LoRA weights separate at save time
  (smaller checkpoints, easier to A/B against the base).

### Throughput

- `global_batch_size=8`, `max_length=2048` — GSM8K is short so we
  don't need long context; batch is small because GLM-4.7 is big.
- `lr=1e-4` — standard LoRA LR (higher than a full-finetune LR
  because only the adapter params update).
- `train_iters=1`, `num_train_epochs=1` — set for a quick smoke run;
  bump either for real training.

```python
swift_framework_config = MsSwiftFrameworkConfig(
    gpu="H100",
    n_nodes=4,
    gpus_per_node=8,
    global_batch_size=8,
    max_length=2048,
    train_iters=1,
    num_train_epochs=1,
)

my_training_run = MsSwiftConfig(
    dataset=GSM8KDataset(HF_CACHE_PATH),
    model=GLM_4_7(),
    wandb=WandbConfig(project="glm-4-7-sft"),
    framework_config=swift_framework_config,
)
```

## Build and run

`build_app()` returns a Modal app with `download_model`,
`prepare_dataset`, and `train`. See
[`quickstart`](../../intro/quickstart/quickstart.ipynb) for the pattern.

```python
app = my_training_run.build_app()
```

---

## Related API Reference

- [`MsSwiftConfig`](/reference/frameworks/msswiftconfig/)
- [`MsSwiftFrameworkConfig`](/reference/frameworks/msswiftframeworkconfig/)
- [`DatasetConfig`](/reference/core/datasetconfig/)
- [`GLM_4_7`](/reference/models/glm_4_7/)
- [`WandbConfig`](/reference/core/wandbconfig/)

**Source:** [`tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py`](https://github.com/modal-projects/training-gym/blob/main/tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py)
 | [Open in Modal Notebook](https://github.com/modal-projects/training-gym/blob/main/tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.ipynb)
