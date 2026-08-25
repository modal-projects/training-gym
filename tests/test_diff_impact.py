from __future__ import annotations

from scripts.diff_impact import analyze_diff


def test_model_file_diff_does_not_infer_tutorial_relationships() -> None:
    diff = (
        "diff --git a/modal_training_gym/common/models/qwen3_5_9b.py "
        "b/modal_training_gym/common/models/qwen3_5_9b.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/modal_training_gym/common/models/qwen3_5_9b.py\n"
        "+++ b/modal_training_gym/common/models/qwen3_5_9b.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "Qwen3_5_9B" in report.affected_classes
    assert report.affected_tutorials == ()


def test_flat_tutorial_diff_maps_to_tutorial() -> None:
    diff = (
        "diff --git a/tutorials/on_policy_distillation.py "
        "b/tutorials/on_policy_distillation.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/tutorials/on_policy_distillation.py\n"
        "+++ b/tutorials/on_policy_distillation.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "on_policy_distillation" in {
        slug for slug, _, _ in report.affected_tutorials
    }
