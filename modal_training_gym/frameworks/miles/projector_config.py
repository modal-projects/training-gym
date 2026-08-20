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
    require_step_hook,
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


def require_projector_step_hook(args) -> None:
    """miles-side :func:`require_step_hook`; see it for what the hook owns."""
    require_step_hook(args, SAVE_HOOK_PATH, "GLM_5_2_Projector_Recipe")


def from_miles_args(args: object) -> ProjectorSpec:
    """Rebuild the spec the recipe serialized, from miles' parsed ``args``."""
    return spec_from_args(args, ARGS_KEY, "miles")
