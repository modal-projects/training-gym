"""Smoke test for the ``ModelConfiguration`` additive introduction.

Verifies:
  1. ``ModelConfiguration`` and the ``Kimi_K2_5`` subclass both import.
  2. ``Kimi_K2_5`` constructs and exposes ``model_name`` / ``model_path``.
  3. ``download_model`` is callable (it is not actually invoked here).
  4. Attaching a ``Kimi_K2_5`` to ``MilesConfig`` does not leak a
     ``--model-configuration`` or subclass repr into ``cli_args()``.
  5. The model instance lives on the outer ``MilesConfig.model`` slot, not on
     ``MilesFrameworkConfig`` — structural guarantee that it never renders
     as a Miles CLI flag.

Run from the repo root:

    uv run python tests/test_model_configuration.py
"""

from __future__ import annotations

import sys


def test_model_configuration_imports_and_constructs() -> None:
    from modal_training_gym.common.models import (
        Kimi_K2_5,
        ModelConfiguration,
    )

    assert ModelConfiguration.__doc__ and "model identity" in ModelConfiguration.__doc__.lower()

    kimi = Kimi_K2_5()
    assert kimi.model_name == "moonshotai/Kimi-K2.5"
    assert isinstance(kimi.model_path, str) and kimi.model_path.endswith(
        "/Kimi-K2.5-bf16"
    )
    assert callable(kimi.download_model)


def test_base_download_model_is_not_implemented() -> None:
    from modal_training_gym.common.models import ModelConfiguration

    try:
        ModelConfiguration().download_model()
    except NotImplementedError:
        pass
    else:
        raise AssertionError(
            "ModelConfiguration.download_model() should raise NotImplementedError"
        )


def test_model_does_not_leak_into_recipe_args() -> None:
    """Miles takes CLI via `recipe_args` + `extra_args` strings (no auto
    field→flag rendering). Attaching a ModelConfiguration must not cause
    the subclass or its instance to bleed into the derived argv list.
    """
    from modal_training_gym.common.models import Kimi_K2_5
    from modal_training_gym.frameworks.miles.config import (
        MilesConfig,
        MilesFrameworkConfig,
    )

    probe = MilesConfig(
        model=Kimi_K2_5(),
        framework_config=MilesFrameworkConfig(),
    )
    argv = probe.framework_config.parsed_recipe_args()
    for token in argv:
        assert "Kimi_K2_5" not in str(token), (
            f"Kimi_K2_5 repr leaked into parsed_recipe_args: {argv}"
        )
        assert "--model-configuration" not in str(token), (
            f"--model-configuration leaked into parsed_recipe_args: {argv}"
        )


def test_model_lives_on_wrapper_not_framework_config() -> None:
    """The custom-model instance belongs on MilesConfig.model, not on
    MilesFrameworkConfig — otherwise it would render as a CLI arg.
    """
    from modal_training_gym.common.models import Kimi_K2_5
    from modal_training_gym.frameworks.miles.config import (
        MilesConfig,
        MilesFrameworkConfig,
    )

    probe = MilesConfig(
        model=Kimi_K2_5(),
        framework_config=MilesFrameworkConfig(),
    )
    assert isinstance(probe.model, Kimi_K2_5)
    assert (
        not hasattr(probe.framework_config, "model")
        or probe.framework_config.model is None
    )


def main() -> int:
    tests = [
        test_model_configuration_imports_and_constructs,
        test_base_download_model_is_not_implemented,
        test_model_does_not_leak_into_recipe_args,
        test_model_lives_on_wrapper_not_framework_config,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
