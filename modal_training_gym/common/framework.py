from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any


def collect_config_class_attrs(
    cls: type,
    *,
    skip: set[str] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    blocked = skip or set()
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        attrs.update(
            {
                key: val
                for key, val in vars(base).items()
                if key not in blocked
                and not key.startswith("_")
                and not callable(val)
            }
        )
    return attrs


def resolve_caller_module(
    skip_prefix: str = "modal_training_gym",
) -> ModuleType | None:
    """Return the first stack frame's module whose name is outside this package.

    Framework configs (e.g. `SlimeConfig.build_app`) delegate to launcher
    factories (e.g. `build_slime_app`) on behalf of user tutorials, so the
    immediate caller of the factory is a framework module — not the tutorial.
    `cloudpickle.register_pickle_by_value(module)` needs the tutorial module
    to inline the user's inline classes; walking past framework frames
    recovers it.
    """
    for frame_info in inspect.stack()[1:]:
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            continue
        name = getattr(module, "__name__", "")
        if not name.startswith(skip_prefix):
            return module
    return None
