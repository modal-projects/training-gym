"""Megatron-bridge training script, spawned via torchrun from the launcher.

This module only runs inside the NeMo image — it imports `megatron.bridge`
which isn't available locally. The launcher calls it by file path via
`torchrun …`; it's never imported from our launcher or tutorial code.

Reads a JSON config (written by the launcher) for all tunable knobs plus
CLI flags for run-specific paths (`--preprocessed_dir`, `--megatron_checkpoint`,
`--checkpoints_dir`, `--config`).
"""

import argparse
import json
import os
from functools import wraps


def parse_args():
    p = argparse.ArgumentParser(description="GLM-4.7 LoRA Training (Megatron bridge)")
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--megatron_checkpoint", required=True)
    p.add_argument("--checkpoints_dir", required=True)
    p.add_argument("--config", required=True, help="Path to JSON config from MegatronConfig.write_script_config")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Imports deferred so `--help` works without megatron installed.
    import torch
    import wandb
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.recipes.utils.optimizer_utils import (
        distributed_fused_adam_with_cosine_annealing,
    )
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        DistributedDataParallelConfig,
        FinetuningDatasetConfig,
        LoggerConfig,
        RNGConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.core.transformer.transformer_block import TransformerBlock

    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and cfg.get("wandb_project"):
        run = wandb.init(project=cfg["wandb_project"])
        print(f"WandB initialized: {run.name} ({run.id})")

    print("=" * 70)
    print(f"Megatron LoRA Training — {cfg['hf_checkpoint']}")
    print("=" * 70)

    # Gradient-flow patch for LoRA + full-recompute compatibility.
    # Without this, recomputed activations arrive without `requires_grad=True`
    # and the LoRA adapters receive zero gradients.
    _original_transformer_forward = TransformerBlock.forward

    @wraps(_original_transformer_forward)
    def _patched_transformer_forward(self, hidden_states, *args_, **kwargs):
        if (
            torch.is_tensor(hidden_states)
            and not hidden_states.requires_grad
            and hidden_states.is_floating_point()
        ):
            hidden_states = hidden_states.detach().requires_grad_(True)
        return _original_transformer_forward(self, hidden_states, *args_, **kwargs)

    TransformerBlock.forward = _patched_transformer_forward
    print("[PEFT+Recompute FIX] Patched TransformerBlock.forward for gradient flow")

    lora_config = LoRA(
        dim=cfg["lora_dim"],
        alpha=cfg["lora_alpha"],
        dropout=cfg["lora_dropout"],
    )

    print(f"Creating config from HF model: {cfg['hf_checkpoint']}")
    bridge = AutoBridge.from_hf_pretrained(cfg["hf_checkpoint"], trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    print(
        f"Model: num_layers={model_cfg.num_layers}, "
        f"num_moe_experts={getattr(model_cfg, 'num_moe_experts', 'N/A')}"
    )

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=cfg["lr_warmup_iters"],
        lr_decay_iters=None,
        max_lr=cfg["lr"],
        min_lr=cfg["min_lr"],
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        adam_eps=cfg["adam_eps"],
        weight_decay=cfg["weight_decay"],
    )

    config = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=cfg["train_iters"],
            eval_interval=9999,
            eval_iters=0,
            global_batch_size=cfg["global_batch_size"],
            micro_batch_size=cfg["micro_batch_size"],
            manual_gc_interval=10,  # reduce DP-sync jitter
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            # Disabled for PEFT: overlap_* expect gradients on every param.
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            bucket_size=100_000_000,
        ),
        dataset=FinetuningDatasetConfig(
            dataset_root=args.preprocessed_dir,
            seq_length=cfg["seq_length"],
            seed=5678,
            dataloader_type="batch",
            num_workers=4,
            do_validation=False,
            do_test=False,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir="/tmp/tensorboard",
            wandb_project=cfg.get("wandb_project") or None,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=cfg["hf_checkpoint"],
        ),
        checkpoint=CheckpointConfig(
            save_interval=cfg["save_interval"],
            save=f"{args.checkpoints_dir}/lora",
            pretrained_checkpoint=args.megatron_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            # Modal blocks pidfd_getfd needed for async CUDA IPC.
            async_save=False,
        ),
        rng=RNGConfig(seed=5678),
        peft=lora_config,
        mixed_precision="bf16_mixed",
    )

    # Parallelism
    config.model.tensor_model_parallel_size = cfg["tensor_model_parallel_size"]
    config.model.pipeline_model_parallel_size = cfg["pipeline_model_parallel_size"]
    config.model.expert_model_parallel_size = cfg["expert_model_parallel_size"]
    config.model.context_parallel_size = cfg["context_parallel_size"]
    config.model.calculate_per_token_loss = True
    config.model.sequence_parallel = True
    config.model.attention_backend = "flash"
    config.model.moe_grouped_gemm = True

    # Recompute
    if not cfg["no_recompute"]:
        config.model.recompute_granularity = "full"
        config.model.recompute_method = "uniform"
        config.model.recompute_num_layers = cfg["recompute_num_layers"]

    config.model.seq_length = cfg["seq_length"]

    # Operator fusion
    config.model.masked_softmax_fusion = True
    config.model.bias_activation_fusion = True
    config.model.bias_dropout_fusion = True
    config.model.apply_rope_fusion = True

    accum_steps = cfg["global_batch_size"] // cfg["micro_batch_size"]
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"  Model: {cfg['hf_checkpoint']}")
    print(f"  seq_length: {config.model.seq_length}")
    print(f"  micro_batch_size: {config.train.micro_batch_size}")
    print(f"  global_batch_size: {config.train.global_batch_size}")
    print(f"  gradient_accumulation_steps: {accum_steps}")
    print(
        f"  TP={config.model.tensor_model_parallel_size}  "
        f"PP={config.model.pipeline_model_parallel_size}  "
        f"EP={config.model.expert_model_parallel_size}  "
        f"CP={config.model.context_parallel_size}"
    )
    if not cfg["no_recompute"]:
        print(f"  recompute_granularity: full  recompute_num_layers: {cfg['recompute_num_layers']}")
    print("=" * 70 + "\n")

    finetune(config=config, forward_step_func=forward_step)
    print("Training complete!")


if __name__ == "__main__":
    main()
