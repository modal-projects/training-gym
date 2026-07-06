"""GDPO (Group Dual-clip Policy Optimization) advantage computation for slime.

Implements per-dimension GRPO advantage normalization for multi-reward training.
Each reward dimension gets its own group-normalized advantage; the combined
advantage (sum of per-dimension advantages) drives the policy update. When
paired with slime's ``--eps-clip-c`` dual-clip flag, this reproduces GDPO
(arxiv:2601.05242).

Reference: https://arxiv.org/abs/2601.05242

Usage (in a SlimeRecipe subclass or extra_config)::

    extra_config = {
        "custom_advantage_function_path":
            "modal_training_gym.frameworks.slime.gdpo.gdpo_compute_advantages",
        # Length penalty params (optional — defaults match toolathlon tuning):
        "gdpo_length_penalty_free_tokens": 4000,
        "gdpo_length_penalty_max_tokens": 16000,
        "gdpo_length_penalty_max_cost": 0.25,
    }
    eps_clip_c: float = 3.0  # GDPO dual-clip on the recipe

The advantage function reads length-penalty params from ``args`` (injected by
``extra_config`` at runtime) and falls back to the defaults above.
"""

from __future__ import annotations


def _length_reward(
    assistant_tokens: int,
    free_tokens: int,
    max_tokens: int,
    max_cost: float,
) -> float:
    """Compute the length penalty reward for a single sample.

    Returns 0.0 when assistant_tokens <= free_tokens, and decreases linearly
    to -max_cost at max_tokens (clamped beyond).
    """
    if assistant_tokens <= free_tokens:
        return 0.0
    span = max_tokens - free_tokens
    if span <= 0:
        return -max_cost
    frac = min(1.0, (assistant_tokens - free_tokens) / span)
    return -max_cost * frac


def _grpo_group_normalize(rewards, group_indices, eps: float = 1e-8):
    """GRPO-style per-group mean/std normalization.

    Args:
        rewards: 1-D tensor of per-sample rewards.
        group_indices: 1-D tensor of integer group labels (same length).
        eps: small constant for numerical stability.

    Returns:
        1-D tensor of per-sample advantages, normalized within each group.
    """
    import torch

    advantages = torch.zeros_like(rewards)
    unique_groups = group_indices.unique()
    for g in unique_groups:
        mask = group_indices == g
        group_r = rewards[mask]
        mean = group_r.mean()
        std = group_r.std(correction=0)
        if std != std or std < eps:  # std != std catches NaN
            advantages[mask] = 0.0
        else:
            advantages[mask] = (group_r - mean) / (std + eps)
    return advantages


def gdpo_compute_advantages(args, rollout_data):
    """GDPO advantage: per-dimension GRPO normalization + summation.

    Two reward dimensions:
      1. **Task reward** — taken from ``rollout_data["rewards"]`` (0/1 pass/fail
         from the Toolathlon evaluator).
      2. **Length reward** — computed on-the-fly from ``rollout_data["loss_masks"]``
         (sum of loss_mask per sample = assistant-generated tokens), then mapped
         through a linear penalty curve parameterized by three constants from
         ``args`` / ``extra_config``.

    Each dimension is group-normalized (GRPO-style) independently, then summed
    to form the final ``advantages`` tensor. This decouples the reward scales so
    neither dimension dominates regardless of raw magnitude.

    Called by slime when ``--custom-advantage-function-path`` points here.
    """
    import torch

    # ── Extract standard training data from rollout_data ──────────────────
    task_rewards = rollout_data["rewards"]  # [num_samples]
    loss_masks = rollout_data["loss_masks"]  # [num_samples, seq_len] or list
    response_lengths = rollout_data["response_lengths"]  # [num_samples]

    # Determine target device from loss_masks (always on the correct CUDA device)
    if isinstance(loss_masks, list) and len(loss_masks) > 0:
        _dev = (
            loss_masks[0].device
            if isinstance(loss_masks[0], torch.Tensor)
            else torch.device("cuda")
        )
    elif isinstance(loss_masks, torch.Tensor):
        _dev = loss_masks.device
    else:
        _dev = torch.device("cuda")

    # Ensure task_rewards and response_lengths are tensors on the right device
    if isinstance(task_rewards, list):
        task_rewards = torch.tensor(task_rewards, dtype=torch.float32, device=_dev)
    else:
        task_rewards = task_rewards.to(_dev)
    if isinstance(response_lengths, list):
        response_lengths = torch.tensor(response_lengths, dtype=torch.long, device=_dev)
    else:
        response_lengths = response_lengths.to(_dev)

    # Group indices for GRPO normalization — samples from the same prompt share
    # a group index.  sample_indices are sequential per-sample; divide by
    # n_samples_per_prompt to get the prompt-level group.
    group_indices = rollout_data.get("sample_indices")
    n_per = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))
    if group_indices is None:
        group_indices = torch.zeros_like(task_rewards, dtype=torch.long)
    elif isinstance(group_indices, list):
        group_indices = (
            torch.tensor(group_indices, dtype=torch.long, device=_dev) // n_per
        )
    else:
        group_indices = group_indices // n_per

    # ── Length penalty params from args (injected via extra_config) ────────
    free_tokens = int(getattr(args, "gdpo_length_penalty_free_tokens", 4000))
    max_tokens = int(getattr(args, "gdpo_length_penalty_max_tokens", 16000))
    max_cost = float(getattr(args, "gdpo_length_penalty_max_cost", 0.25))

    # ── Compute per-sample length rewards ─────────────────────────────────
    # Assistant tokens = sum of loss_mask (trainable tokens the policy wrote).
    if isinstance(loss_masks, torch.Tensor) and loss_masks.dim() == 2:
        assistant_tokens_per_sample = loss_masks.sum(dim=1)
    elif isinstance(loss_masks, list):
        assistant_tokens_per_sample = torch.tensor(
            [
                sum(m) if not isinstance(m, torch.Tensor) else m.sum().item()
                for m in loss_masks
            ],
            dtype=task_rewards.dtype,
            device=task_rewards.device,
        )
    else:
        # Fallback: use response_lengths as a proxy.
        assistant_tokens_per_sample = response_lengths.float()

    length_rewards = torch.tensor(
        [
            _length_reward(int(at.item()), free_tokens, max_tokens, max_cost)
            for at in assistant_tokens_per_sample
        ],
        dtype=task_rewards.dtype,
        device=task_rewards.device,
    )

    # ── Per-dimension GRPO advantages ─────────────────────────────────────
    task_advantages = _grpo_group_normalize(task_rewards, group_indices)
    length_advantages = _grpo_group_normalize(length_rewards, group_indices)

    # ── GDPO: combine per-dimension advantages ───────────────────────────
    # Sum of per-dimension advantages (each already group-normalized to
    # zero-mean, unit-variance). When fed to the dual-clip PPO loss
    # (--eps-clip-c), this gives the GDPO update rule from the paper.
    combined_advantages = task_advantages + length_advantages

    # ── Expand per-sample advantages to CP-local per-token tensors ────────
    # With context_parallel_size > 1, the loss function works on CP-local
    # chunks.  We expand each scalar advantage to response_length tokens,
    # then slice to the CP-local chunk via slice_log_prob_with_cp (the same
    # helper used by the KL-zero fallback in patch_advantages.py).
    total_lengths = rollout_data["total_lengths"]
    if isinstance(total_lengths, list):
        total_lengths = torch.tensor(total_lengths, dtype=torch.long)
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    per_token_advantages = [
        slice_log_prob_with_cp(
            combined_advantages[i].expand(int(response_lengths[i].item())),
            int(total_lengths[i].item()),
            int(response_lengths[i].item()),
        )
        for i in range(len(combined_advantages))
    ]
    rollout_data["advantages"] = per_token_advantages
    rollout_data["returns"] = per_token_advantages
