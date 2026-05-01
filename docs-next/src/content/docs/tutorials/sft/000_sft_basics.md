---
title: "SFT basics: serve, evaluate, train, evaluate again"
description: "Qwen3-0.6B SFT on generated arithmetic problems"
---

This tutorial uses Qwen3-0.6B and generated arithmetic problems to
show the basic SFT loop:

1. Serve the base model.
2. Evaluate it on a fresh arithmetic eval set.
3. LoRA SFT the model with ms-swift.
4. Serve the trained checkpoint.
5. Evaluate it with the same scorer.

The training config is intentionally tiny: 1×H100 and one training
iteration. It is a smoke run for the workflow, not a quality run.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any

import modal

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.eval import EvalConfig, EvalRowResult
from modal_training_gym.common.models import Qwen3_0_6B
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.ms_swift import (
    MsSwiftConfig,
    MsSwiftFrameworkConfig,
)
from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH
```

## Serve the base model

`ModelConfig.serve()` builds and deploys a vLLM app, then
returns a `ModelDeployment` with the concrete endpoint URL.

```python
base_model = Qwen3_0_6B()
base_deployment = base_model.serve(
    app_name="qwen3-0.6b-base-serve",
    served_model_name="qwen3-0.6b-base",
)
print(base_deployment.url)
```

## Generate train/eval datasets and scorer

The dataset below generates arithmetic expression problems locally.
The training split and eval split use different seeds, so they are
fresh samples from the same generator rather than copies of each
other.

`prepare()` writes chat-format JSONL for ms-swift:
`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.

`load()` returns raw `{question, answer}` dictionaries for local
inspection and eval.

`EvalConfig` owns the model-calling loop. The task-specific part is a
scoring function passed to `.evaluate(...)`, which must return
`EvalRowResult`.

```python
class Operation(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


PHRASES = [
    "What is",
    "Compute",
    "Derive",
    "Calculate",
    "Find",
    "Solve",
    "Evaluate",
]
OPS = [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV]


@dataclass
class LeafClause:
    value: int

    def evaluate(self) -> float:
        return float(self.value)

    def to_string(self) -> str:
        return str(self.value)


@dataclass
class Clause:
    left: "LeafClause | Clause"
    right: "LeafClause | Clause"
    op: Operation

    def evaluate(self) -> float | None:
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        if left_val is None or right_val is None:
            return None

        if self.op == Operation.ADD:
            return left_val + right_val
        if self.op == Operation.SUB:
            return left_val - right_val
        if self.op == Operation.MUL:
            return left_val * right_val
        if self.op == Operation.DIV:
            if right_val == 0:
                return None
            return left_val / right_val
        return None

    def to_string(self) -> str:
        left_str = self.left.to_string()
        right_str = self.right.to_string()

        if isinstance(self.left, Clause):
            left_str = f"({left_str})"
        if isinstance(self.right, Clause):
            right_str = f"({right_str})"

        return f"{left_str} {self.op.value} {right_str}"


class ArithmeticDataset(DatasetConfig):
    input_column = "question"
    output_column = "answer"

    def __init__(
        self,
        data_root=HF_CACHE_PATH,
        *,
        split: str,
        seed: int,
        num_per_depth: int = 64,
    ):
        self.split = split
        self.seed = seed
        self.num_per_depth = num_per_depth
        self.prompt_data = f"{data_root}/arithmetic/{split}.jsonl"

    def random_leaf(self, rng, max_val: int = 20) -> LeafClause:
        return LeafClause(rng.randint(1, max_val))

    def random_op(self, rng) -> Operation:
        return rng.choice(OPS)

    def generate_tree(self, rng, depth: int, max_val: int = 20, force_op: bool = False):
        if depth <= 0:
            return self.random_leaf(rng, max_val)

        should_branch = force_op or depth == 1 or rng.random() < 0.7
        if not should_branch:
            return self.random_leaf(rng, max_val)

        left = self.generate_tree(rng, depth - 1, max_val)
        right = self.generate_tree(rng, depth - 1, max_val)
        op = self.random_op(rng)

        if op == Operation.DIV and isinstance(right, LeafClause) and right.value != 0:
            if isinstance(left, LeafClause):
                left = LeafClause(right.value * rng.randint(1, 10))

        return Clause(left, right, op)

    def generate_problem(self, rng, problem_id: str, depth: int) -> dict[str, Any] | None:
        tree = self.generate_tree(rng, depth, force_op=True)
        answer = tree.evaluate()
        if answer is None:
            return None

        if abs(answer - round(answer)) < 0.0001:
            answer = int(round(answer))
        else:
            answer = round(answer, 2)

        question = (
            f"{rng.choice(PHRASES)} {tree.to_string()}"
            f"{rng.choice(['?', '.', '', '!'])}"
        )
        return {"id": problem_id, "question": question, "answer": str(answer)}

    def load(self) -> list[dict[str, Any]]:
        import random

        rng = random.Random(self.seed)
        records = []
        problem_counter = {"d1": 1, "d2": 1, "d3": 1}
        configs = [
            ("d1", 1, self.num_per_depth),
            ("d2", 2, self.num_per_depth * 2),
            ("d3", 3, self.num_per_depth * 3),
        ]

        for prefix, depth, count in configs:
            generated = 0
            attempts = 0
            while generated < count and attempts < count * 10:
                attempts += 1
                pid = f"{prefix}_{str(problem_counter[prefix]).zfill(3)}"
                problem = self.generate_problem(rng, pid, depth)
                if problem is not None:
                    records.append(problem)
                    problem_counter[prefix] += 1
                    generated += 1

        rng.shuffle(records)
        return records

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.load())

    def prepare(self) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(self.prompt_data), exist_ok=True)
        with open(self.prompt_data, "w") as f:
            for record in self.load():
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": record["question"]},
                                {"role": "assistant", "content": record["answer"]},
                            ]
                        }
                    )
                    + "\n"
                )


def score_arithmetic(example: dict[str, Any], response: str) -> EvalRowResult:
    prediction = response.replace(",", "").strip()
    gold = str(example["answer"]).replace(",", "").strip()
    try:
        correct = abs(float(prediction) - float(gold)) < 1e-6
    except ValueError:
        correct = prediction == gold
    return EvalRowResult(
        score=1.0 if correct else 0.0,
        passed=correct,
        response=response,
        metadata={"prediction": prediction, "gold": gold},
    )


class ArithmeticPrompts(EvalConfig):
    def build_prompt(self, example: dict[str, Any]) -> str:
        return (
            f"{example['question']}\n\n"
            "Only output the final numeric answer. Do not include units or explanation."
        )


train_dataset = ArithmeticDataset(split="train", seed=0, num_per_depth=64)
eval_dataset = ArithmeticDataset(split="eval", seed=1, num_per_depth=8)
```

Let's take a look at the generated eval set. Each row has a question
and a numeric answer; there is no hidden chain-of-thought target.

```python
pd = eval_dataset.to_pandas()
print(len(pd))
pd.head(5)
```

## Evaluate the base model

```python
base_eval = ArithmeticPrompts(
    deployment=base_deployment,
    dataset=eval_dataset,
).evaluate(score_arithmetic)
print(f"Base arithmetic accuracy: {base_eval.accuracy:.1%}")
```

## Train with ms-swift

This run uses a single H100 and one iteration. Increase
`train_iters` and `num_per_depth` for a real experiment.

```python
swift_framework_config = MsSwiftFrameworkConfig(
    n_nodes=1,
    gpus_per_node=1,
    global_batch_size=1,
    max_length=2048,
    train_iters=1,
    num_train_epochs=1,
    save_interval=1,
    eval_iters=0,
)

my_training_run = MsSwiftConfig(
    name="qwen3-0.6b-arithmetic-sft",
    dataset=train_dataset,
    model=base_model,
    wandb=WandbConfig(project="qwen3-0.6b-arithmetic-sft"),
    framework_config=swift_framework_config,
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
result = TrainResult.load("qwen3-0.6b-arithmetic-sft")
print(result.latest_checkpoint_path())

deployment = result.model.serve(
    app_name="qwen3-0.6b-arithmetic-sft-serve",
    served_model_name="qwen3-0.6b-arithmetic-sft",
)
print(deployment.url)

trained_eval = ArithmeticPrompts(
    deployment=deployment,
    dataset=eval_dataset,
).evaluate(score_arithmetic)
print(f"Trained arithmetic accuracy: {trained_eval.accuracy:.1%}")
print(f"Delta: {trained_eval.accuracy - base_eval.accuracy:+.1%}")
```

---

## Related API Reference

- [`Qwen3_0_6B`](/reference/models/qwen3_0_6b/)
- `EvalConfig`
- `EvalRowResult`
- [`MsSwiftConfig`](/reference/frameworks/msswiftconfig/)
- [`MsSwiftFrameworkConfig`](/reference/frameworks/msswiftframeworkconfig/)
- [`WandbConfig`](/reference/core/wandbconfig/)
- [`TrainResult`](/reference/core/trainresult/)

**Source:** [`tutorials/sft/000_sft_basics/000_sft_basics.py`](https://github.com/modal-projects/training-gym/blob/main/tutorials/sft/000_sft_basics/000_sft_basics.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/main/tutorials/sft/000_sft_basics/000_sft_basics.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
