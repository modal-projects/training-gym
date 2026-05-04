from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

_QWEN3_4B_DEFAULTS = {
    # ── Cluster ─────────────────────────────────────────────────────────
    "gpu_type": "H100",
    "actor_num_nodes": 1,
    "actor_num_gpus_per_node": 8,
    "colocate": True,
    "tensor_model_parallel_size": 1,
    "sequence_parallel": False,
    # ── RL / rollout ────────────────────────────────────────────────────
    "n_samples_per_prompt": 8,
    "rollout_batch_size": 16,
    "rollout_max_response_len": 4096,
    "rollout_temperature": 1.0,
    "sglang_mem_fraction_static": 0.75,
    # ── Training ────────────────────────────────────────────────────────
    "global_batch_size": 16,
    "lr": 5e-7,
    "max_tokens_per_gpu": 8192,
    # ── Eval ────────────────────────────────────────────────────────────
    "eval_interval": 10,
    "n_samples_per_eval_prompt": 4,
    "eval_max_response_len": 4096,
    # ── Checkpointing ───────────────────────────────────────────────────
    "save_interval": 10,
    "megatron_to_hf_mode": "bridge",
}


_SLIME_DEFAULTS = SlimeRecipe()


class Qwen3_4b_Recipe(SlimeRecipe):
    """Qwen3-4B on 1×8×H100, colocated GRPO."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_4B_DEFAULTS.items():
            if getattr(self, key) == getattr(_SLIME_DEFAULTS, key):
                object.__setattr__(self, key, val)
        super().__post_init__()
