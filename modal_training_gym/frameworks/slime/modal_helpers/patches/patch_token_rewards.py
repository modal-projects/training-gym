"""Teach Slime's rollout transport about optional token reward vectors."""

from __future__ import annotations

from pathlib import Path


ROLLOUT = Path("/root/slime/slime/ray/rollout.py")
MARKER = "# training-gym: token rewards"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count == 0:
        raise RuntimeError(f"{label}: anchor not found")
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_rollout() -> None:
    if not ROLLOUT.is_file():
        print(f"[patch_token_rewards] WARNING: {ROLLOUT} not found; skipped")
        return
    source = ROLLOUT.read_text()
    if MARKER in source:
        print("[patch_token_rewards] rollout.py already patched")
        return
    source = _replace_once(
        source,
        '        train_data["loss_masks"] = loss_masks\n',
        '        train_data["loss_masks"] = loss_masks\n\n'
        f"        {MARKER}\n"
        "        from modal_training_gym.common.token_rewards import build_token_reward_vectors\n"
        "\n"
        "        token_rewards = build_token_reward_vectors(samples, rewards)\n"
        "        if token_rewards is not None:\n"
        '            train_data["token_rewards"] = token_rewards\n',
        "rollout conversion",
    )
    source = _replace_once(
        source,
        '                "loss_masks",\n                "round_number",\n',
        '                "loss_masks",\n                "token_rewards",\n                "round_number",\n',
        "rollout partition",
    )
    ROLLOUT.write_text(source)
    print("[patch_token_rewards] patched rollout.py")


if __name__ == "__main__":
    patch_rollout()
