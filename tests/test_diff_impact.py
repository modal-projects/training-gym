from __future__ import annotations

from scripts.diff_impact import analyze_diff


def test_model_file_diff_maps_to_related_tutorials() -> None:
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
    tutorial_slugs = {slug for slug, _, _ in report.affected_tutorials}
    assert "agent/000_agent_sandbox" in tutorial_slugs
    assert "rl/003_on_policy_distillation" in tutorial_slugs


def test_generated_tutorial_diff_maps_back_to_source() -> None:
    diff = (
        "diff --git a/tutorials/rl/003_on_policy_distillation/"
        "003_on_policy_distillation.py "
        "b/tutorials/rl/003_on_policy_distillation/003_on_policy_distillation.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/tutorials/rl/003_on_policy_distillation/"
        "003_on_policy_distillation.py\n"
        "+++ b/tutorials/rl/003_on_policy_distillation/"
        "003_on_policy_distillation.py\n"
        "@@ -1,3 +1,3 @@\n"
    )

    report = analyze_diff(diff)

    assert "rl/003_on_policy_distillation" in {
        slug for slug, _, _ in report.affected_tutorials
    }
