---
order: 3
---

# Training

Once you have your model, dataset, and training recipe, you're ready to start training:

```python
config = TrainConfig(
    model=model,
    dataset=dataset,
    recipe=recipe,
)
run = config.launch()

print(f"Run ID: {run.training_run_id}")
print(f"Modal App: {run.modal_app_id}")
print(f"Modal App URL: {run.modal_app_url}")
```

The training run will continue in the background entirely on Modal-managed infrastructure, so sit back and relax! At any time, you can easily observe its progress on the [Gym dashboard](https://gym.modal.dev/guides/observability-dashboard) or with the [CLI](https://gym.modal.dev/reference/cli).

## Handling checkpoints

After the run has completed, you'll likely want to get its saved checkpoints (which are stored on a [Modal Volume](https://modal.com/docs/guide/volumes)) to either deploy them as an [Endpoint](https://modal.com/docs/guide/endpoints) or continue training with a new configuration.

Note that you can always access runs by their ID:

```python
run = TrainingRun.from_id("bristled-pine-a7c3e91d4b2")
```

From there, we get the run's result after it has completed:

```python
result = run.result()
```

Then, get the latest checkpoint:

```python
checkpoint = result.checkpoints()[-1]
```

Now, we could do offline evals:

```python
deployment = Endpoint.launch(
    model, checkpoint, unauthenticated=True, recreate_if_existing=True
)
deployment.wait_until_ready(timeout=15 * 60)
print(f"trained model deployed to {deployment.url}")


def score(response, label):
    return 1 if response==label else 0


def run_eval(deployment, max_concurrency: int = 2) -> float:
    from concurrent.futures import ThreadPoolExecutor

    def _score_one(example):
        prompt = example["prompt"][0]["content"]
        msg = deployment.chat(
            [{"role": "user", "content": prompt}],
            chat_template_kwargs={"enable_thinking": True},
        )
        response = msg.get("content") or msg.get("reasoning_content") or ""
        return score(response, example["label"])

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        scores = list(executor.map(_score_one, eval_dataset.load()))
    percent_correct = (
        len([s for s in scores if s == 1]) / len(scores) if scores else float("nan")
    )
    return percent_correct

print("running trained model evaluation...")
correct = run_eval(deployment)
print(f"percent correct: {correct:.1%}")
```

Or train the model further with the existing checkpoint as a starting point:

```python
config = TrainConfig(
    model=model,
    dataset=dataset,
    checkpoint=checkpoint,
    recipe=recipe,
)
```

Why might you want to continue training the model? As an example, you can implement curriculum learning by increasing the difficulty of the data and reward function over time:

```python
simple_config = TrainConfig(
    model=model,
    dataset=simple_dataset,
    recipe=simple_recipe,
)

simple_run = simple_config.launch()
simple_result = simple_run.result()
simple_checkpoint = simple_result.latest_checkpoint()

complex_config = TrainConfig(
    model=model,
    checkpoint=simple_checkpoint,
    dataset=complex_dataset,
    recipe=complex_recipe,
)

complex_run = complex_config.launch()
complex_result = complex_run.result()
complex_checkpoint = complex_result.latest_checkpoint()
```
