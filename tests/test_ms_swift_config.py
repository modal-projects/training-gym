"""Targeted tests for ms-swift model-driven launcher defaults.

Run from the repo root:

    uv run python tests/test_ms_swift_config.py
"""

from __future__ import annotations

import sys


def test_qwen3_0_6b_builtin_model_defaults() -> None:
    from modal_training_gym.common.models import Qwen3_0_6B

    model = Qwen3_0_6B()
    assert model.model_name == "Qwen/Qwen3-0.6B"
    assert model.training is not None
    assert model.training.n_nodes == 1
    assert model.training.gpus_per_node == 1
    assert model.architecture is not None
    assert model.architecture.hidden_size == 1024


def test_ms_swift_resolves_model_topology_defaults() -> None:
    from modal_training_gym.common.models import Qwen3_0_6B
    from modal_training_gym.frameworks.ms_swift import MsSwiftConfig

    cfg = MsSwiftConfig(model=Qwen3_0_6B())
    framework = cfg.resolved_framework_config()
    assert framework.n_nodes == 1
    assert framework.gpus_per_node == 1


def test_explicit_framework_shape_beats_model_defaults() -> None:
    from modal_training_gym.common.models import Qwen3_0_6B
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig,
        MsSwiftFrameworkConfig,
    )

    cfg = MsSwiftConfig(
        model=Qwen3_0_6B(),
        framework_config=MsSwiftFrameworkConfig(n_nodes=1, gpus_per_node=4),
    )
    framework = cfg.resolved_framework_config()
    assert framework.n_nodes == 1
    assert framework.gpus_per_node == 4


def test_multi_node_requires_eight_gpus_per_node() -> None:
    from modal_training_gym.frameworks.ms_swift import MsSwiftFrameworkConfig

    try:
        MsSwiftFrameworkConfig(n_nodes=2, gpus_per_node=4)
    except ValueError as exc:
        assert "gpus_per_node=8" in str(exc)
    else:
        raise AssertionError("Expected multi-node config validation to fail")


def test_single_node_train_path_skips_clustered_wrapper() -> None:
    from modal_training_gym.frameworks.ms_swift.launcher import (
        _train_ms_swift_worker,
        _use_clustered_launch,
    )

    assert _use_clustered_launch(1) is False
    assert _use_clustered_launch(2) is True
    assert callable(_train_ms_swift_worker)


def main() -> int:
    tests = [
        test_qwen3_0_6b_builtin_model_defaults,
        test_ms_swift_resolves_model_topology_defaults,
        test_explicit_framework_shape_beats_model_defaults,
        test_multi_node_requires_eight_gpus_per_node,
        test_single_node_train_path_skips_clustered_wrapper,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures.append((test.__name__, str(exc)))
            print(f"  FAIL  {test.__name__}: {exc}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
