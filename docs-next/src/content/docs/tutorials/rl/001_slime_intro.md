---
title: "Qwen3-4B GRPO on GSM8K with slime on Modal"
description: "Qwen3-4B GRPO on GSM8K (colocated)"
---

**What slime is.** [slime](https://github.com/THUDM/slime) is an RL
post-training framework that pairs Megatron (training) with SGLang
(rollouts) and orchestrates both with Ray. `modal-training-gym`'s
`slime` launcher wires that stack onto a Modal multi-node cluster.

**What this tutorial does.** GRPO-tunes Qwen3-4B against
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) on 4 nodes ×
8×H100 with actor and rollout **colocated** on the same GPUs. GSM8K
is the canonical target for math-RL: short prompts, short answers,
and a deterministic correctness check. This is the "everything works
end-to-end" reference for the `slime` framework — a medium-scale RL
post-training run with slime's built-in math reward (no custom
reward code). For a custom-reward example see
[`003_slime_with_llm_as_judge`](../../003_slime_with_llm_as_judge/003_slime_with_llm_as_judge.ipynb); for the shared
primitives (DatasetConfig, volumes, the 3-stage pipeline) see
[`001_quickstart`](../../intro/001_quickstart/001_quickstart.ipynb).

**What you'll need.**
- A Modal account with GPU access (1 × 8×H100).
- A `wandb` Modal secret holding your W&B API key (the slime launcher
  mounts it automatically when `WandbConfig` is present).
- Patience: multi-hour run — use `modal run --detach`.

**What to watch.**
- **Weights & Biases** under project `slime-grpo`, group
  `qwen3-4b-gsm8k`. Rollout reward should climb steadily; eval fires
  every 20 training steps (see `eval_interval` below) against
  GSM8K's test split.
- **Modal dashboard** — per-node GPU utilization and live logs. On a
  healthy run you'll see SGLang warm up, then alternating rollout /
  training phases.

```python
import modal

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.slime import (
    ModalConfig,
    SlimeConfig,
)
from modal_training_gym.frameworks.slime.config import DATA_PATH
```

## Define the dataset

The non-obvious choices for GSM8K under slime:

- `input_key="messages"` + `apply_chat_template=True` — the prompt
  column holds a list of chat messages; slime runs the model's chat
  template over them before tokenizing. The upstream
  [`zhuzilin/gsm8k`](https://huggingface.co/datasets/zhuzilin/gsm8k)
  mirror we load below is already in that shape.
- `label_key="label"` — the column slime scores against.
- `rm_type="math"` — selects slime's built-in math correctness
  reward. It parses the boxed numeric answer out of the rollout and
  compares to `label`. No custom reward code needed.
- `rollout_shuffle=True` — matters for GRPO's group-sampling
  stability.

```python
class GSM8KDataset(HuggingFaceDataset):
    hf_repo = "zhuzilin/gsm8k"
    input_key = "messages"
    label_key = "label"

    def __init__(self, data_root):
        super().__init__(data_root)
        test_path = self.prompt_data.replace("train.parquet", "test.parquet")
        self.eval_prompt_data = ["gsm8k", test_path]

    def prepare(self):
        from datasets import load_dataset

        super().prepare()
        test_path = self.prompt_data.replace("train.parquet", "test.parquet")
        ds = load_dataset(self.hf_repo, split="test")
        ds.to_parquet(test_path)
```

## Define the experiment

`SlimeConfig` is a pydantic dataclass — hover over any field in
your IDE to see its type, default, and description. Only non-default
values need to be specified; everything else inherits sensible
defaults.

**Cluster**

- `colocate=True` — actor and rollout share the same GPUs.

**Throughput**
- `use_dynamic_batch_size=True` + `max_tokens_per_gpu=9216` — pack
  prompts up to a per-GPU token budget.
- `recompute_granularity="full"` — activation recomputation for
  memory savings.

**RL**
- `use_kl_loss=True` — KL divergence is computed (but `kl_loss_coef`
  defaults to 0.0, so it's tracked but not penalized).
- `weight_decay=0.1`, `adam_beta2=0.98` — optimizer overrides.

```python
base_model = Qwen3_4B()
my_training_run = SlimeConfig(
    name="qwen3-4b-gsm8k",
    model=base_model,
    dataset=GSM8KDataset(DATA_PATH),
    wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-gsm8k"),
    ref_load=base_model.model_name,
)
```

## Build and run

`build_app()` returns a Modal app with `download_model`,
`prepare_dataset`, and `train`. (Bridge mode means there's no
separate `convert_checkpoint` step to call — see quickstart for the
general pattern.)

```python
app = my_training_run.build_app()
```

## Use the trained model

After `train` completes, `TrainResult` gives you back a **model** —
the trained checkpoint, ready to serve, evaluate, or continue
training from. No need to re-import the training config.

```python
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.serve_vllm import build_vllm_serve_app

result = TrainResult.load("qwen3-4b-gsm8k")
trained_model = result.model
```

Both the base model and the trained checkpoint are HuggingFace-format
weights, so you can serve them the same way. Deploy them side by side
to compare outputs:

```python
# Serve the TRAINED model (checkpoint from training):
trained_app = result.build_serve_app(served_model_name="qwen3-4b-gsm8k-trained")

# Serve the BASE model (original HuggingFace weights):
base_app = build_vllm_serve_app(
    app_name="qwen3-4b-base-serve",
    model_path="Qwen/Qwen3-4B",
    served_model_name="qwen3-4b-base",
)
```

```bash
# Deploy both:
modal deploy eval.py::trained_app
modal deploy eval.py::base_app
```

Now you have two OpenAI-compatible endpoints. Compare them on
GSM8K prompts to measure the improvement:

```python
import openai

base_client = openai.OpenAI(base_url="<base_app_url>/v1", api_key="na")
trained_client = openai.OpenAI(base_url="<trained_app_url>/v1", api_key="na")

prompt = "What is 15% of 80? Show your work step by step."

base_answer = base_client.chat.completions.create(
    model="qwen3-4b-base",
    messages=[{"role": "user", "content": prompt}],
).choices[0].message.content

trained_answer = trained_client.chat.completions.create(
    model="qwen3-4b-gsm8k-trained",
    messages=[{"role": "user", "content": prompt}],
).choices[0].message.content

print("BASE:", base_answer)
print("TRAINED:", trained_answer)
```

The trained model should produce more structured math reasoning
and box its final answer (the format GSM8K rewards).

You can also continue training from the checkpoint:

```python
next_run = SlimeConfig(
    model=trained_model,  # picks up where this run left off
    dataset=...,
)
```

Or pull W&B metrics to see the training curves:

```python
print(result.wandb_url())
metrics = result.wandb_metrics(keys=["train/loss", "train/reward"])
```

---

## Related API Reference

- [`SlimeConfig`](/reference/frameworks/slimeconfig/)
- [`DatasetConfig`](/reference/core/datasetconfig/)
- [`Qwen3_4B`](/reference/models/qwen3_4b/)
- [`WandbConfig`](/reference/core/wandbconfig/)

**Source:** [`tutorials/rl/001_slime_intro/001_slime_intro.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/tutorials/rl/001_slime_intro/001_slime_intro.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/joy/initial-setup/tutorials/rl/001_slime_intro/001_slime_intro.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
