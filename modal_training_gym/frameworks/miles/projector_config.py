"""Wire format between a projector-only recipe and the miles containers.

Kept free of torch so the launching side can import it: the recipe serializes a
:class:`ProjectorSpec` into ``extra_config`` under :data:`ARGS_KEY`, miles sets
every ``extra_config`` key on ``args``, and
:mod:`modal_training_gym.frameworks.miles.embedding_projector` reads it back
inside the container. No new miles CLI flags are involved.
"""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

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

# Keys of the per-sample tensor dict miles concatenates over a packed batch and
# splats into ``model.forward``. ``_positions`` is miles' own suffix convention:
# it offsets any such key by the sample's start in the packed sequence.
EMBEDDINGS_KEY = "projector_embeddings"
POSITIONS_KEY = "projector_positions"


@dataclass(config=ConfigDict(extra="forbid"))
class ProjectorSpec:
    """The projector's shape, its data keys, and where its checkpoints go.

    input_dim : int
        Width of the embeddings the dataset supplies — the external encoder's
        output dimension.
    hidden_dim : int
        Width of the projector's inner layers.
    num_layers : int
        Linear layers in the projector; 1 makes it a single linear map.
    output_dim : int | None
        Width of the projected vectors; ``None`` uses the model's hidden size,
        which is what writing into the decoder's input embeddings requires.
    embeddings_key : str
        Dataset column (and ``forward`` keyword) holding the per-token
        embeddings, as one ``[num_tokens, input_dim]`` tensor per sample.
    positions_key : str
        Dataset column holding the token positions those embeddings occupy.
    save_dir : str
        Directory for projector checkpoints; empty puts them under
        ``<save>/projector``.
    save_interval : int
        Save the projector every N rollout steps.
    load : str
        Projector checkpoint (file or directory) to resume from.
    """

    input_dim: int = 1536
    hidden_dim: int = 4096
    num_layers: int = 2
    output_dim: int | None = None
    embeddings_key: str = EMBEDDINGS_KEY
    positions_key: str = POSITIONS_KEY
    save_dir: str = ""
    save_interval: int = 10
    load: str = ""

    def to_args_dict(self) -> dict[str, int | str | None]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "embeddings_key": self.embeddings_key,
            "positions_key": self.positions_key,
            "save_dir": self.save_dir,
            "save_interval": self.save_interval,
            "load": self.load,
        }


def from_miles_args(args: object) -> ProjectorSpec:
    """Rebuild the spec the recipe serialized, from miles' parsed ``args``."""
    raw = vars(args).get(ARGS_KEY)
    if not isinstance(raw, dict):
        raise ValueError(
            f"miles arg '{ARGS_KEY}' is missing or not a dict (got {raw!r}). A "
            "projector-only run needs the recipe to serialize a ProjectorSpec "
            "into extra_config."
        )
    return ProjectorSpec(**raw)
