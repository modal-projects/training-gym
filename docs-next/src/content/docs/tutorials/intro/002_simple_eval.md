---
title: "Writing an Eval on Training Gym"
description: "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare"
---

Now that we have a basic model deployed, we can write an eval to see how it performs.

Evals are important before you train a model, because they give you a baseline that your trained model can be compared to.
They also help you figure out if your reward function is too lenient or too strict.

```python
from modal_training_gym import (
    DeploymentConfig,
    EvalConfig,
    EvalRowResult,
    HuggingFaceDataset,
    Qwen3_4B,
)
```

## Serve the base model

As always, let's start by serving the base model.

```python
base_model = Qwen3_4B()
base_model_deployment = DeploymentConfig(
    model=base_model,
).serve()
print(f"Base model deployed to {base_model_deployment.url}")
```

In this example, we are going to ask our model to solve math problems, and evaluate how well it does on that.

To first define an eval, we should define a dataset that contains the problems we want to solve.

We will use `Onlydrinkwater/int_division` from HuggingFace as our dataset.
Each row has a `question` and a `label`.
We can use this dataset to evaluate our model.

```python
class IntDivisionDataset(HuggingFaceDataset):
    hf_repo = "Onlydrinkwater/int_division"
    input_column = "question"
    output_column = "label"
    output_format = "jsonl"
    apply_chat_template = True
    system_prompt = (
        "You are a math problem solver. Solve the given math problem."
    )
eval_dataset = IntDivisionDataset(n_rows=20)
```

Let's now specify a scoring function, which is that the model's response should be the same as the label.

Here, _example is a dictionary that contains the dataset row, and response is the model's response.

```python
def eval_response_fn(_example: dict, response: str) -> EvalRowResult:
    if _example["label"] == response:
        return EvalRowResult(score=1.0, response=response)
    else:
        return EvalRowResult(score=0.0, response=response)

eval_config = EvalConfig(
    dataset=eval_dataset,
    eval_response_fn=eval_response_fn,
)
```

```python
print("——— Running base model evaluation... ———")
base_eval = eval_config.evaluate(base_model_deployment, debug=True)
print(f"Average haiku score: {base_eval.mean:.1f}")
print("——— Base model evaluation complete ———")
```

Why did we get a score of 0.0?
Well, the model's response is not the same as the label. What is it then? We can check the dashboard url to debug this or we can just print the response.

```python
print(base_eval.rows[0])
```

There are several issues with our eval:
- We did not disable thinking, so the model is thinking about the problem and then answering.
- The model did not output an exact number, so it would never be perfectly equivalent to the label.

If we immediately trained on this dataset with this reward function above, it would be extremely difficult for our model
to learn to solve the problems correctly.

Let's fix both of these issues.
- First, we will disable thinking using the `generate_kwargs` parameter.
- Second, we will update the system prompt to instruct the model to output only the answer.

Let's see how this works.

```python
class FixedIntDivisionDataset(HuggingFaceDataset):
    hf_repo = "Onlydrinkwater/int_division"
    input_column = "question"
    output_column = "label"
    output_format = "jsonl"
    apply_chat_template = True
    # This is new!
    system_prompt = (
        "You are a math problem solver. Output the answer to the given math problem, and nothing else. Your response must be a valid integer."
    )

def new_eval_response_fn(_example: dict, response: str) -> EvalRowResult:
    if _example["label"] == response:
        return EvalRowResult(score=1.0, response=response)
    else:
        return EvalRowResult(score=0.0, response=response)

new_eval_config = EvalConfig(
    dataset=FixedIntDivisionDataset(),
    eval_response_fn=new_eval_response_fn,
    generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
)
```

```python
print("——— Running fixed eval... ———")
fixed_eval = new_eval_config.evaluate(base_model_deployment, debug=True)
print(f"Average fixed eval score: {fixed_eval.mean:.1f}")
print("——— Fixed eval complete ———")
```

---

## Related API Reference

- [`Qwen3_4B`](/reference/models/qwen3_4b/)
- [`DeploymentConfig`](/reference/deployment/deploymentconfig/)
- [`EvalConfig`](/reference/evaluation/evalconfig/)
- [`EvalRowResult`](/reference/evaluation/evalrowresult/)

**Source:** [`tutorials/intro/002_simple_eval/002_simple_eval.py`](https://github.com/modal-projects/training-gym/blob/main/tutorials/intro/002_simple_eval/002_simple_eval.py)
 | <a href="https://modal.com/notebooks/new/https://github.com/modal-projects/training-gym/blob/main/tutorials/intro/002_simple_eval/002_simple_eval.ipynb" target="_blank" rel="noopener noreferrer">Open in Modal Notebook</a>
