from __future__ import annotations

from modal_training_gym.frameworks.miles.modal_helpers.patches.patch_gemma4_vl_rollout_text import (
    gemma4_collapse_image_token_ids,
)


def test_gemma4_collapse_keeps_one_image_token_per_run() -> None:
    image_id = 7
    prompt_ids = [1, 2] + [image_id] * 264 + [3, 4]
    collapsed = gemma4_collapse_image_token_ids(prompt_ids, image_id)
    assert collapsed == [1, 2, image_id, 3, 4]
    assert gemma4_collapse_image_token_ids([1, 2, 3], image_id) == [1, 2, 3]
