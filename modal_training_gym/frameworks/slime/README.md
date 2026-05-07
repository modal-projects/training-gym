# slime — Modal launcher for slime RL training

Thin Modal launcher that runs [slime](https://github.com/THUDM/slime) GRPO training on GPU clusters.

## Quick start

Define a `TrainConfig` with your model, dataset, and a `SlimeRecipe`, then call `train()`:

```python
from modal_training_gym import SlimeRecipe, TrainConfig, WandbConfig
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig

config = TrainConfig(
    model=my_model,
    dataset=my_dataset,
    recipe=SlimeRecipe(
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        gpu_type="H100",
        wandb=WandbConfig(project="my-project"),
    ),
)
result = config.train()
```

Then run: `uv run modal run my_tutorial.py::train`

## Recipe building blocks

`SlimeRecipe` can be composed with reusable blocks:

```python
from modal_training_gym import MultiTurn, SlimeRecipe

recipe = SlimeRecipe(
    rollout_batch_size=4,
).with_blocks(
    MultiTurn(
        generate_function=number_guess_generate,
        reward_function=number_guess_rm,
        max_turns=6,
        log_multi_turn=True,
    )
)
```

Each block gets the current recipe and returns a new one, so users can share
"lego" blocks across tasks.

## SlimeRecipe

`SlimeRecipe` is a Pydantic dataclass that holds all configuration for a slime training run:
launcher instructions, cluster topology, RL hyperparameters, and checkpointing.

Every field on `SlimeRecipe` (except internal fields like `environment`, `async_mode`,
`wandb`, `image_overlay`, etc.) is forwarded to slime as a CLI argument:
`field_name` → `--field-name`.

See `modal_training_gym/train_recipes/slime_recipe/recipe.py` for the full field list.

### Key field groups

- **Cluster**: `gpu_type`, `actor_num_nodes`, `actor_num_gpus_per_node`, `colocate`, `tensor_model_parallel_size`
- **RL algorithm**: `advantage_estimator`, `n_samples_per_prompt`, `eps_clip`, `kl_loss_coef`
- **Rollout**: `rollout_batch_size`, `rollout_max_response_len`, `rollout_temperature`
- **Training**: `global_batch_size`, `lr`, `lr_decay_style`, `weight_decay`, `optimizer`
- **Checkpointing**: `save`, `save_interval`, `megatron_to_hf_mode`
- **Eval**: `eval_interval`, `eval_config`
- **Reward**: `rm_type`, `custom_rm_path`, `custom_rm_function`

## YAML config fields

`eval_config`, `custom_config_path`, and `sglang_config` normally take file paths in slime.
In `SlimeRecipe` you can write them as inline dicts — the launcher materializes them to temp YAML files automatically:

```python
recipe = SlimeRecipe(
    eval_config={
        "eval": {
            "defaults": {"max_response_len": 16384},
            "datasets": [
                {"name": "aime", "path": "/data/aime.jsonl", "rm_type": "deepscaler"},
            ],
        }
    },
    custom_config_path={
        "max_turns": 3,
        "rollout_interaction_env_path": "examples.my_env.rollout",
    },
)
```

## Image overlay

To customize the container image (e.g. install extra packages), pass `image_overlay`:

```python
recipe = SlimeRecipe(
    image_overlay=lambda img: img.pip_install("my-package"),
)
```

## Dev overlay

To test local slime changes without rebuilding the image, set `local_slime`:

```python
recipe = SlimeRecipe(
    local_slime="/path/to/your/slime",
)
```
