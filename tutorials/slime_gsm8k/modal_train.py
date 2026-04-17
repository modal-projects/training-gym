"""Qwen3-4B GRPO on GSM8K — 4 nodes, colocated.

Run with:
    uv run modal run tutorials/train_slime_qwen_4b_gsm8k.py::app.download_model
    uv run modal run tutorials/train_slime_qwen_4b_gsm8k.py::app.prepare_dataset
    uv run modal run tutorials/train_slime_qwen_4b_gsm8k.py::app.train
"""

from modal_training_gym.frameworks.slime import (
    ModalConfig,
    SlimeConfig,
    build_slime_app,
)
from modal_training_gym.frameworks.slime.config import DATA_PATH


class _Slime(SlimeConfig):
    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3-4B"
    ref_load = hf_checkpoint  # reference model = same base checkpoint in bridge mode
    megatron_to_hf_mode = "bridge"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 4
    actor_num_gpus_per_node = 8
    colocate = True

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/gsm8k/train.parquet"
    eval_prompt_data = ["gsm8k", f"{DATA_PATH}/gsm8k/test.parquet"]
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    # ── Rollout ───────────────────────────────────────────────────────────────
    num_rollout = 3000
    rollout_batch_size = 64
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    rollout_num_gpus_per_engine = 1
    sglang_mem_fraction_static = 0.7
    n_samples_per_prompt = 8
    global_batch_size = 128
    use_fault_tolerance = True

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_interval = 20
    n_samples_per_eval_prompt = 4
    eval_max_response_len = 16384
    eval_top_p = 1.0

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 1
    sequence_parallel = False
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 9216
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0

    # ── Model architecture (Qwen3-4B) ─────────────────────────────────────────
    num_layers = 36
    hidden_size = 2560
    ffn_hidden_size = 9728
    num_attention_heads = 32
    group_query_attention = True
    num_query_groups = 8
    kv_channels = 128
    vocab_size = 151936
    normalization = "RMSNorm"
    norm_epsilon = 1e-6
    swiglu = True
    disable_bias_linear = True
    qk_layernorm = True
    use_rotary_position_embeddings = True
    rotary_base = 1000000

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = "slime-grpo"
    wandb_group = "qwen3-4b-gsm8k"
    disable_wandb_random_suffix = True

    def prepare_data(self) -> None:
        """Download GSM8K from HuggingFace and save as parquet to the data volume."""
        from datasets import load_dataset

        ds = load_dataset("zhuzilin/gsm8k")
        ds["train"].to_parquet(f"{DATA_PATH}/gsm8k/train.parquet")
        ds["test"].to_parquet(f"{DATA_PATH}/gsm8k/test.parquet")


app = build_slime_app(modal=ModalConfig(gpu="H200"), slime=_Slime())

