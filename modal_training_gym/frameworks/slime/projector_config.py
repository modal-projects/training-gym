"""Wire format between a projector-only recipe and the slime containers.

Kept free of torch so the launching side can import it: the recipe serializes a
:class:`~modal_training_gym.common.projector_config.ProjectorSpec` into
``extra_config`` under :data:`ARGS_KEY`, slime sets every ``extra_config`` key on
its parsed ``args``, and
:mod:`modal_training_gym.frameworks.slime.embedding_projector` reads it back
inside the container. No new slime CLI flags are involved.

The spec itself is shared with miles (see
:mod:`modal_training_gym.common.projector_config`); what lives here is only the
slime-side plumbing: the arg name and the dotted paths slime resolves.
"""

from modal_training_gym.common.projector_config import (
    EMBEDDINGS_KEY as EMBEDDINGS_KEY,
    POSITIONS_KEY as POSITIONS_KEY,
    ProjectorSpec as ProjectorSpec,
    should_save_projector as should_save_projector,
    spec_from_args,
)

# slime arg carrying the serialized ProjectorSpec.
ARGS_KEY = "training_gym_projector"

# Per-sample row count, alongside the embeddings and their positions, so the
# merge can rebase each sample's positions onto the packed sequence. slime
# concatenates every ``multimodal_train_inputs`` key over the microbatch's
# samples and adds no offsets (unlike miles, which offsets ``*_positions``
# keys), so the row-to-sample mapping has to travel with the data.
ROW_COUNTS_KEY = "projector_row_counts"

# Regexes for slime's ``--only-train-params-name-list``. The provider already
# freezes the base itself, so this is not what makes the run projector-only —
# it states the same intent in slime's own terms, and it is what
# ``wrap_model_provider_with_freeze`` re-applies to the model the provider
# returns (a no-op when neither freeze list is set). Unanchored because the
# names slime matches against carry Megatron's wrapper prefixes; the projector
# is registered as ``embedding_projector``.
TRAINABLE_PARAM_PATTERNS = ["embedding_projector"]

PROVIDER_PATH = (
    "modal_training_gym.frameworks.slime.embedding_projector.projector_model_provider"
)
SAVE_HOOK_PATH = (
    "modal_training_gym.frameworks.slime.embedding_projector.save_projector_checkpoint"
)
ROLLOUT_PATH = (
    "modal_training_gym.frameworks.slime.embedding_projector.projector_sft_rollout"
)


def from_slime_args(args: object) -> ProjectorSpec:
    """Rebuild the spec the recipe serialized, from slime's parsed ``args``."""
    return spec_from_args(args, ARGS_KEY, "slime")
