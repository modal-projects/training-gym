"""Slime custom advantage function for transition-level reward training."""

from __future__ import annotations

from typing import Any


def compute_token_reward_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    """Compute discounted response-token returns from ``token_rewards``.

    Event samples use return-to-go so an action receives credit for rewards it
    caused later in the episode.  Samples without events use Slime's normal
    scalar GRPO broadcast, which keeps mixed-source batches compatible.
    """
    import torch

    estimator = getattr(args, "advantage_estimator", "grpo")
    if estimator not in {"grpo", "gspo", "cispo"}:
        raise ValueError(
            "Training Gym transition rewards currently support only GRPO/GSPO/CISPO; "
            f"got advantage_estimator={estimator!r}"
        )

    kl = rollout_data["kl"]
    scalar_rewards = rollout_data["rewards"]
    token_rewards = rollout_data.get("token_rewards")
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    loss_masks = rollout_data["loss_masks"]
    gamma = float(getattr(args, "training_gym_token_reward_gamma", 1.0))

    advantages = []
    returns = []
    for index, kl_chunk in enumerate(kl):
        vector = token_rewards[index] if token_rewards is not None else None
        if vector is None:
            value = torch.as_tensor(
                scalar_rewards[index], dtype=torch.float32, device=kl_chunk.device
            )
            result = torch.ones_like(kl_chunk, dtype=torch.float32) * value
            advantages.append(result)
            returns.append(result)
            continue

        full_rewards = torch.as_tensor(
            vector, dtype=torch.float32, device=kl_chunk.device
        )
        full_mask = torch.as_tensor(
            loss_masks[index], dtype=torch.float32, device=kl_chunk.device
        )
        if (
            full_rewards.numel() != int(response_lengths[index])
            or full_mask.numel() != full_rewards.numel()
        ):
            raise ValueError(
                "Transition reward vector must match response/loss-mask length: "
                f"sample={index}, rewards={full_rewards.numel()}, "
                f"response={response_lengths[index]}, mask={full_mask.numel()}"
            )

        full_returns = torch.zeros_like(full_rewards)
        running = torch.zeros((), dtype=torch.float32, device=kl_chunk.device)
        for position in range(full_rewards.numel() - 1, -1, -1):
            if full_mask[position] != 0:
                running = full_rewards[position] + gamma * running
                full_returns[position] = running

        if len(kl_chunk) == len(full_returns):
            local_returns = full_returns
        else:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns = slice_log_prob_with_cp(
                full_returns,
                int(total_lengths[index]),
                int(response_lengths[index]),
            )
        advantages.append(local_returns)
        returns.append(local_returns)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
