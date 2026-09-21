"""Disaggregated miles training on Modal via stitch.

``build_stitch_app`` is exported lazily: the serving image imports this package
when Modal deserializes the pool's ``Server`` class, and it has no trainer stack
— an eager launcher import would make every recipe/launcher dependency a
serving-image requirement.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal_training_gym.frameworks.stitch.launcher import build_stitch_app

__all__ = ["build_stitch_app"]


def __getattr__(name: str) -> Any:
    if name == "build_stitch_app":
        from modal_training_gym.frameworks.stitch.launcher import build_stitch_app

        return build_stitch_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
