"""Wire format between a projector-only recipe and the miles containers.

Kept free of torch so the launching side can import it: the recipe serializes a
:class:`~modal_training_gym.common.projector_config.ProjectorSpec` into
``extra_config`` under :data:`ARGS_KEY`, miles sets every ``extra_config`` key on
``args``, and :mod:`modal_training_gym.frameworks.miles.embedding_projector`
reads it back inside the container. No new miles CLI flags are involved.

The spec itself is shared with slime (see
:mod:`modal_training_gym.common.projector_config`); what lives here is only the
miles-side plumbing: the arg name and the dotted paths miles resolves.
"""

from modal_training_gym.common.projector_config import (
    EMBEDDINGS_KEY as EMBEDDINGS_KEY,
    POSITIONS_KEY as POSITIONS_KEY,
    ProjectorSpec as ProjectorSpec,
    should_save_projector as should_save_projector,
    spec_from_args,
)

# Miles arg carrying the serialized ProjectorSpec.
ARGS_KEY = "training_gym_projector"

PROVIDER_PATH = (
    "modal_training_gym.frameworks.miles.embedding_projector.projector_model_provider"
)
SAVE_HOOK_PATH = (
    "modal_training_gym.frameworks.miles.embedding_projector.save_projector_checkpoint"
)
ROLLOUT_PATH = (
    "modal_training_gym.frameworks.miles.embedding_projector.projector_sft_rollout"
)

_STEP_HOOK_KEY = "training_gym_custom_megatron_before_train_step_hook_path"


def _step_hook_path(args) -> str | None:
    """Resolve the hook miles' reporting wrapper will dispatch to.

    Mirrors ``phase_reporting._hook_path_from_args``: the recipe's own hook
    travels in ``extra_config`` (miles sets every key on ``args``) while the
    ``--custom-megatron-before-train-step-hook-path`` flag names the gym's
    reporting wrapper.
    """
    direct = getattr(args, _STEP_HOOK_KEY, None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(_STEP_HOOK_KEY) or container.get(
                _STEP_HOOK_KEY.removeprefix("training_gym_")
            )
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def require_projector_step_hook(args) -> None:
    """Fail at model construction when the projector's step hook is unwired.

    ``save_projector_checkpoint`` is the only thing that installs the
    ``optimizer.step`` wrapper, and that wrapper owns two jobs nothing else
    does: the tensor-parallel all-reduce of the replicated projector's
    gradients, and writing the projector checkpoints. Neither announces its
    absence, so a run whose hook never fires would train on
    tensor-parallel-partial gradients and finish with nothing saved. The wiring
    is a recipe default, but it reaches the container through ``extra_config``,
    so the provider checks it — miles has to call the provider for the run to
    exist at all — rather than trusting it.
    """
    wrapper = getattr(args, "custom_megatron_before_train_step_hook_path", None)
    wrapper = wrapper.strip() if isinstance(wrapper, str) else ""
    wrapped = _step_hook_path(args)
    if SAVE_HOOK_PATH in (wrapper, wrapped):
        return
    raise ValueError(
        "projector-only training needs miles' before-train-step hook to reach "
        f"'{SAVE_HOOK_PATH}', which installs the projector's gradient "
        "all-reduce and checkpoint writer, but this run resolves "
        f"--custom-megatron-before-train-step-hook-path to '{wrapper}' and the "
        f"gym's wrapped hook to '{wrapped}'. Launch through "
        "GLM_5_2_Projector_Recipe (or point "
        "custom_megatron_before_train_step_hook at that path) rather than "
        "training with unreduced gradients and no checkpoints."
    )


def from_miles_args(args: object) -> ProjectorSpec:
    """Rebuild the spec the recipe serialized, from miles' parsed ``args``."""
    return spec_from_args(args, ARGS_KEY, "miles")
