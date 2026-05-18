---
title: SlimeRecipe
description: API reference for SlimeRecipe
---

```python
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe
```

Recipe dataclass for configuring slime GRPO training on Modal.

**Inherits from:** `BaseTrainRecipe`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_type` | `str` |  |  |
| `colocate` | `bool` |  |  |
| `tensor_model_parallel_size` | `int` |  |  |
| `sequence_parallel` | `bool` |  |  |
| `rollout_num_gpus_per_engine` | `int` |  |  |
| `num_rollout` | `int` |  |  |
| `rollout_batch_size` | `int` |  |  |
| `rollout_max_response_len` | `int` |  |  |
| `rollout_temperature` | `float` |  |  |
| `save_interval` | `int` |  |  |
| `recipe_type` | `RecipeType` | `slime` |  |
| `name` | `str` | `""` |  |
| `app_tags` | `dict` | `{}` |  |
| `environment` | `dict` | `{'PYTHONPATH': '/root/Megatron-LM/', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '1'}` |  |
| `async_mode` | `bool` | `False` |  |
| `wandb` | `WandbConfig \| None` | `None` |  |
| `image_overlay` | `collections.abc.Callable[[modal.image.Image], modal.image.Image] \| None` | `None` |  |
| `local_slime` | `str \| None` | `None` |  |
| `actor_num_nodes` | `int` | `1` |  |
| `actor_num_gpus_per_node` | `int` | `8` |  |
| `rollout_num_gpus` | `int \| None` | `None` |  |
| `use_critic` | `bool` | `False` |  |
| `critic_num_nodes` | `int \| None` | `None` |  |
| `critic_num_gpus_per_node` | `int \| None` | `None` |  |
| `advantage_estimator` | `str` | `"grpo"` |  |
| `n_samples_per_prompt` | `int` | `2` |  |
| `eps_clip` | `float` | `0.2` |  |
| `eps_clip_high` | `float` | `0.28` |  |
| `use_kl_loss` | `bool` | `False` |  |
| `kl_loss_type` | `str` | `"low_var_kl"` |  |
| `kl_loss_coef` | `float` | `0.0` |  |
| `entropy_coef` | `float` | `0.0` |  |
| `ref_load` | `str` | `""` |  |
| `rollout_shuffle` | `bool` | `True` |  |
| `sglang_mem_fraction_static` | `float` | `0.75` |  |
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
| `eval_interval` | `int \| None` | `None` |  |
| `n_samples_per_eval_prompt` | `int` | `4` |  |
| `eval_max_response_len` | `int` | `16384` |  |
| `eval_top_p` | `float` | `1.0` |  |
| `eval_config` | `dict \| None` | `None` |  |
| `save` | `str` | `"/checkpoints"` |  |
| `megatron_to_hf_mode` | `str` | `"bridge"` |  |
| `use_fault_tolerance` | `bool` | `True` |  |
| `rm_type` | `str \| None` | `None` |  |
| `custom_rm_function` | `collections.abc.Callable \| None` | `None` |  |
| `custom_generate_function` | `collections.abc.Callable \| None` | `None` |  |
| `rollout_function` | `collections.abc.Callable \| str \| None` | `None` |  |
| `custom_megatron_before_train_step_hook` | `collections.abc.Callable \| str \| None` | `None` |  |
| `extra_config` | `dict \| None` | `None` |  |
| `sglang_config` | `dict \| None` | `None` |  |
| `apply_chat_template_kwargs` | `str` | `""` |  |

## Methods

### `cli_args(self, dataset: 'DatasetConfig | None' = None, model: 'ModelConfig | None' = None) -> list[str]`

### `get_base_recipe(model_config: modal_training_gym.common.models.base.ModelConfig) -> 'SlimeRecipe | None'`

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn/)
- [On-policy distillation on math — Qwen3-8B teacher, Qwen3-4B student](/tutorials/rl/003_on_policy_distillation/)

**Source:** [`modal_training_gym/train_recipes/slime_recipe/recipe.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/train_recipes/slime_recipe/recipe.py)
