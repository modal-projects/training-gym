---
title: SlimeRecipe
description: API reference for SlimeRecipe
---

# SlimeRecipe

```python
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe
```

Recipe dataclass for configuring slime GRPO training on Modal.

**Inherits from:** `BaseTrainRecipe`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recipe_type` | `<enum 'RecipeType` | `slime` |  |
| `name` | `str` | `""` |  |
| `app_tags` | `dict` | `{}` |  |
| `environment` | `dict` | `{'PYTHONPATH': '/root/Megatron-LM/', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '1'}` |  |
| `async_mode` | `bool` | `False` |  |
| `wandb` | `WandbConfig | None` | `None` |  |
| `image_run_commands` | `collections.abc.Callable[[modal.image.Image], modal.image.Image] | None` | `None` |  |
| `local_python_sources` | `list[str]` | `[]` |  |
| `local_slime` | `str | None` | `None` |  |
| `gpu_type` | `str` | `"H100"` |  |
| `actor_num_nodes` | `int` | `1` |  |
| `actor_num_gpus_per_node` | `int` | `8` |  |
| `colocate` | `bool` | `True` |  |
| `tensor_model_parallel_size` | `int` | `1` |  |
| `sequence_parallel` | `bool` | `False` |  |
| `rollout_num_gpus` | `int | None` | `None` |  |
| `rollout_num_gpus_per_engine` | `int` | `1` |  |
| `use_critic` | `bool` | `False` |  |
| `critic_num_nodes` | `int | None` | `None` |  |
| `critic_num_gpus_per_node` | `int | None` | `None` |  |
| `advantage_estimator` | `str` | `"grpo"` |  |
| `n_samples_per_prompt` | `int` | `2` |  |
| `eps_clip` | `float` | `0.2` |  |
| `eps_clip_high` | `float` | `0.28` |  |
| `use_kl_loss` | `bool` | `False` |  |
| `kl_loss_type` | `str` | `"low_var_kl"` |  |
| `kl_loss_coef` | `float` | `0.0` |  |
| `entropy_coef` | `float` | `0.0` |  |
| `ref_load` | `str` | `""` |  |
| `num_rollout` | `int` | `1` |  |
| `rollout_batch_size` | `int` | `8` |  |
| `rollout_max_response_len` | `int` | `8192` |  |
| `rollout_temperature` | `float` | `1.0` |  |
| `sglang_mem_fraction_static` | `float` | `0.7` |  |
| `rollout_shuffle` | `bool` | `True` |  |
| `global_batch_size` | `int` | `16` |  |
| `lr` | `float` | `1e-06` |  |
| `lr_decay_style` | `str` | `"constant"` |  |
| `weight_decay` | `float` | `0.1` |  |
| `adam_beta1` | `float` | `0.9` |  |
| `adam_beta2` | `float` | `0.98` |  |
| `optimizer` | `str` | `"adam"` |  |
| `attention_dropout` | `float` | `0.0` |  |
| `hidden_dropout` | `float` | `0.0` |  |
| `attention_softmax_in_fp32` | `bool` | `True` |  |
| `accumulate_allreduce_grads_in_fp32` | `bool` | `True` |  |
| `recompute_granularity` | `str` | `"full"` |  |
| `recompute_method` | `str` | `"uniform"` |  |
| `recompute_num_layers` | `int` | `1` |  |
| `use_dynamic_batch_size` | `bool` | `True` |  |
| `max_tokens_per_gpu` | `int` | `9216` |  |
| `eval_interval` | `int` | `0` |  |
| `n_samples_per_eval_prompt` | `int` | `4` |  |
| `eval_max_response_len` | `int` | `16384` |  |
| `eval_top_p` | `float` | `1.0` |  |
| `eval_config` | `dict | None` | `None` |  |
| `save` | `str` | `"/checkpoints"` |  |
| `save_interval` | `int` | `1000` |  |
| `megatron_to_hf_mode` | `str` | `"bridge"` |  |
| `use_fault_tolerance` | `bool` | `True` |  |
| `checkpoint` | `modal_training_gym.common.checkpoint.CheckpointConfig` | `<modal_training_gym.common.checkpoint.CheckpointConfig object at 0x7f2bcae6c140>` |  |
| `rm_type` | `str | None` | `None` |  |
| `custom_rm_path` | `str` | `""` |  |
| `custom_rm_function` | `collections.abc.Callable | None` | `None` |  |
| `sglang_config` | `dict | None` | `None` |  |
| `custom_config_path` | `dict | None` | `None` |  |
| `apply_chat_template_kwargs` | `str` | `""` |  |

## Methods

### `cli_args(self, dataset: 'DatasetConfig | None' = None, model: 'ModelConfig | None' = None) -> list[str]`

### `get_base_recipe(model_config: modal_training_gym.common.models.base.ModelConfig) -> 'SlimeRecipe'`

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)

**Source:** [`modal_training_gym/train_recipes/slime_recipe/recipe.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/train_recipes/slime_recipe/recipe.py)
