from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal import Image


# Canonical mount point for `modal_training_gym/tools/` inside every framework
# image. Framework-agnostic helpers (e.g. `Kimi_K2_5.download`) shell out
# to scripts at this path, so it must exist on every launcher's image.
TOOLS_REMOTE_PATH = "/opt/training-gym/tools"

# Local path to the tools directory. Shipped with the package so
# `mount_tools_dir` works from an editable install or a `pip install`ed copy.
TOOLS_LOCAL_PATH = Path(__file__).resolve().parent.parent / "tools"


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
                if key not in blocked and not key.startswith("_") and not callable(val)
            }
        )
    return attrs


def mount_tools_dir(image: "Image") -> "Image":
    """Attach `modal_training_gym/tools/` to a Modal image at `TOOLS_REMOTE_PATH`.

    Every framework launcher should call this on its training image(s) so
    cross-framework scripts (e.g. `convert_kimi_int4_to_bf16.py`) resolve at
    a predictable remote path regardless of the framework image's layout.
    """
    return image.add_local_dir(
        TOOLS_LOCAL_PATH,
        remote_path=TOOLS_REMOTE_PATH,
        copy=True,
    )


def resolve_app_name(
    framework: str,
    explicit_name: str,
    model: Any = None,
) -> str:
    """Derive the Modal app name from the user-provided name or the model.

    If ``explicit_name`` is set, uses it directly. Otherwise derives a
    readable default from the framework + model name and logs it.
    """
    if explicit_name:
        return explicit_name
    if model is not None and getattr(model, "model_name", ""):
        slug = model.model_name.split("/")[-1].lower().replace("-", "_")
        name = f"{framework}-{slug}"
    else:
        name = f"{framework}-run"
    print(
        f"[training-gym] App name not set — using '{name}'. "
        f"Set name= on your config to customize."
    )
    return name


def resolve_gpu(model: Any) -> str:
    """Resolve the GPU type from a model's preset or gpu_type attribute.

    Checks (in order): model.gpu_type, model.slime.gpu_type,
    model.training.gpu_type.
    """
    if model is None:
        raise ValueError(
            "Cannot resolve GPU: no model attached. Set model= on your config."
        )
    if hasattr(model, "gpu_type") and model.gpu_type:
        return model.gpu_type
    for attr in ("slime", "training"):
        preset = getattr(model, attr, None)
        if preset is not None and hasattr(preset, "gpu_type"):
            return preset.gpu_type
    raise ValueError(
        f"{type(model).__name__} has no gpu_type. "
        f"Set gpu_type on your model class or add a framework preset."
    )


def resolve_caller_module(
    skip_prefix: str = "modal_training_gym",
) -> ModuleType | None:
    """Return the first stack frame's module whose name is outside this package.

    Framework configs (e.g. `TrainConfig.build_app`) delegate to launcher
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
