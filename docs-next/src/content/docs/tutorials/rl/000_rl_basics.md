---
title: "RL basics: verifiable rewards, haiku edition"
description: "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare"
---

This tutorial uses Qwen3-4B and haiku poems to introduce the
**verifiable reward** pattern that underpins RL post-training:

1. Serve the base model.
2. Define a scoring function with a verifiable reward (syllable structure).
3. Evaluate the base model against that scorer.
4. GRPO-train the model with slime using the reward function.
5. Serve the trained checkpoint.
6. Evaluate it with the same scorer and compare.

Why haikus? A haiku has two attributes you can score
automatically — whether it follows the 5-7-5 syllable format
(deterministic, cheap) and whether the poem is actually good
(subjective, needs an LLM judge). That split between
*verifiable* and *subjective* rewards is exactly the landscape
RL post-training operates in. This tutorial covers the
verifiable half; see
[`003_slime_with_llm_as_judge`](../003_slime_with_llm_as_judge/)
for the LLM-judge extension.

The training config is intentionally tiny: 1×H100 and one
training iteration. It is a smoke run for the workflow, not a
quality run.

```python
import re
from typing import Any

import modal

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.eval import EvalConfig, EvalRowResult
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.slime import SlimeConfig
from modal_training_gym.frameworks.slime.config import DATA_PATH
```

## Serve the base model

So, how does Qwen3-4B currently fare at writing haikus? We can
serve the base model and find out.

`ModelConfig.serve()` builds and deploys a vLLM app, then
returns a `ModelDeployment` with the concrete endpoint URL.

```python
base_model = Qwen3_4B()
base_deployment = base_model.serve(
    app_name="qwen3-4b-base-serve",
    served_model_name="qwen3-4b-base",
)
print(base_deployment.url)
```

Now that the model has come alive, we can request it to write a haiku about a topic.

```python
response = base_deployment.generate(
    "Write a haiku about cat.",
    chat_template_kwargs={"enable_thinking": False},
)
print(response)
```

Okay, how do we evaluate if that was a good haiku or not?
In addition to the poetic quality, haikus must follow the 5-7-5 syllable format.
We can use a simple heuristic to count the syllables in the response.

Seems straightforward enough, right? How do we run an eval on our base model?
We can transform our scoring function above into an Eval Configuration.

First, to explain, an Eval Configuration is a class that owns the model-calling loop.
The task-specific part is a scoring function passed to `.evaluate(...)`, which must
return `EvalRowResult`.

Before, our score_haiku function only returned a boolean, which gives 0 or 1 score.
But we can make it return more granular result on *how much off* it was from the syllable
target.

```python
def score_haiku(response: str) -> EvalRowResult:
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    has_three_lines = len(lines) == 3

    syllable_score = 0.0
    syllable_counts = []
    if has_three_lines:
        for line, target in zip(lines, [5, 7, 5]):
            count = _count_syllables(line)
            syllable_counts.append(count)
            diff = abs(count - target)
            if diff == 0:
                syllable_score += 1.0
            elif diff == 1:
                syllable_score += 0.5
        syllable_score /= 3.0

    score = (0.25 if has_three_lines else 0.0) + 0.75 * syllable_score
    return EvalRowResult(
        score=score,
        passed=score >= 0.75,
        response=response,
        metadata={
            "lines": len(lines),
            "syllable_counts": syllable_counts,
        },
    )
```

```python
class HaikuEvalConfig(EvalConfig):        
    def build_prompt(self, example: dict[str, Any]) -> str:
        return (
            f"Write a haiku about {example['keywords'].lower()}.\n\n"
            "Output only the three lines of the haiku, nothing else."
        )
```

Let's also define a Haiku dataset.
Here, we use the statworx/haiku dataset from HuggingFace.
Each row has a `keywords` topic and a reference `text` haiku.
We can use this dataset to train our model.

```python
SYSTEM_PROMPT = (
    "You are a haiku poet. Write a haiku about the given topic. "
    "Use the 5-7-5 syllable format across three lines."
)

class HaikuDataset(HuggingFaceDataset):
    hf_repo = "statworx/haiku"
    input_column = "keywords"
    output_column = "text"
    output_format = "jsonl"
    system_prompt = SYSTEM_PROMPT

    def __init__(
        self,
        data_root=DATA_PATH,
        *,
        split: str = "train",
        n_train: int = 500,
        n_eval: int = 50,
    ):
        super().__init__(data_root)
        self.split = split
        self.n_train = n_train
        self.n_eval = n_eval
        self.prompt_data = f"{data_root}/haiku/{split}.jsonl"

    def load(self):
        ds = super().load()
        if self.split == "train":
            return ds.select(range(min(self.n_train, len(ds))))
        else:
            return ds.select(range(len(ds) - self.n_eval, len(ds)))

    def _format_for_training(self, ds):
        in_col = self.input_column
        sys_prompt = self.system_prompt

        def _to_chat(row: dict) -> dict:
            return {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": f"Write a haiku about {row[in_col].lower()}.",
                    },
                    {"role": "assistant", "content": row["text"]},
                ]
            }

        return ds.map(_to_chat, remove_columns=ds.column_names)


train_dataset = HaikuDataset(split="train", n_train=50)
eval_dataset = HaikuDataset(split="eval", n_eval=10)
```

Let's take a look at the eval set. Each row has a `keywords`
topic and a reference `text` haiku.

```python
df = eval_dataset.to_pandas()
print(len(df))
df.head(5)
```

## Evaluate the base model

```python
base_eval = HaikuEvalConfig(
    deployment=base_deployment,
    dataset=eval_dataset,
    eval_fn=(lambda df_row, response: score_haiku({}, response)),
    generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
).evaluate()
print(f"Base haiku score: {base_eval.accuracy:.1%}")
print(f"Passed (score >= 0.75): {base_eval.correct}/{base_eval.total}")
```

## Train with slime

Now, let's actually train the model to write good haikus.
Here, we use the slime framework (https://github.com/THUDM/slime) on Modal.

```python
from modal_training_gym.common.haiku_reward import haiku_rm

my_training_run = SlimeConfig(
    model=base_model,
    dataset=HaikuDataset(DATA_PATH, base_model.model_name),
    wandb=WandbConfig(project="slime-grpo", group="qwen3-0.6b-haiku"),

    custom_rm_function=haiku_rm,

    num_rollout=10,
    apply_chat_template_kwargs='{"enable_thinking": false}',

    eval_interval=5,
    n_samples_per_eval_prompt=8,

    save="/checkpoints/qwen3-0.6b-haiku",
    save_interval=1,

    image_run_commands=[
        "uv pip install --system aiohttp nltk>=3.8.0",
        "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
    ],
)
```

## Build and run

```python
app = my_training_run.build_app()
```

## Serve and evaluate the trained checkpoint

After training completes, `TrainResult.load(...)` reconstructs the
trained model with its checkpoint path and volume metadata attached.
Calling `.serve(...)` deploys that checkpoint behind vLLM.

```python
result = TrainResult.load("qwen3-4b-haiku-sft")
print(result.latest_checkpoint_path())

deployment = result.model.serve(
    app_name="qwen3-4b-haiku-sft-serve",
    served_model_name="qwen3-4b-haiku-sft",
)
print(deployment.url)

trained_eval = HaikuEvalConfig(
    deployment=deployment,
    dataset=eval_dataset,
    eval_fn=(lambda golden_df_row, response: score_haiku(response)),
    generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
).evaluate(debug=True)
print(f"Trained haiku score: {trained_eval.accuracy:.1%}")
print(f"Passed (score >= 0.75): {trained_eval.correct}/{trained_eval.total}")
print(f"Delta: {trained_eval.accuracy - base_eval.accuracy:+.1%}")
```

## What's next: RL with LLM--as-a-judge

---

## Related API Reference

- [`Qwen3_4B`](/reference/models/qwen3_4b/)
- `EvalConfig`
- `EvalRowResult`
- [`SlimeConfig`](/reference/frameworks/slimeconfig/)
- [`WandbConfig`](/reference/core/wandbconfig/)
- [`TrainResult`](/reference/core/trainresult/)

**Source:** [`tutorials/rl/000_rl_basics/000_rl_basics.py`](https://github.com/modal-projects/training-gym/blob/main/tutorials/rl/000_rl_basics/000_rl_basics.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/main/tutorials/rl/000_rl_basics/000_rl_basics.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
