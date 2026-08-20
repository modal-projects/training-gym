"""Projector-only training: the config both trainers share, free of torch.

A projector-only run trains one small adapter over a frozen base model: the
dataset supplies embeddings computed by some external encoder (a protein or DNA
model, an audio tower, a retrieval index), a small MLP maps them into the
decoder's hidden space, and they are written into the decoder's input embeddings
at the token positions the data names. Nothing else trains — no LoRA, no base
parameters — so optimizer state, gradient buffers and checkpoints are
proportional to the adapter rather than to the model.

This module stays importable on the launching machine (no torch), because the
recipe serializes a :class:`ProjectorSpec` into ``extra_config``. The torch and
Megatron parts live in :mod:`modal_training_gym.common.embedding_projector`, and
the framework-specific glue — which arg carries the config, which
provider/rollout/hook paths exist, how a packed batch's per-sample tensors are
laid out — in ``frameworks/<framework>/embedding_projector.py``.
"""

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

LATEST_CHECKPOINT = "projector_latest.pt"

# Keys of the per-sample tensor dict a packed microbatch concatenates and splats
# into ``model.forward``.
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
        Dataset column holding the token positions those embeddings occupy. Has
        to keep the ``_positions`` suffix: it is miles' convention for the keys
        it rebases onto the packed sequence, and slime's projector merge keys its
        own rebase off the same suffix.
    save_dir : str
        Directory for projector checkpoints; empty puts them under
        ``<save>/projector``.
    save_interval : int
        Save the projector every N optimizer steps. The final step of a run is
        always saved regardless, so a finished run leaves a checkpoint even when
        the interval does not divide the step count.
    load : str
        Projector checkpoint (file or directory) to resume from.
    output_scale : float
        Scale the projector's final ``LayerNorm`` starts at, and so the scale of
        the rows it writes into the embedding stream. A ``LayerNorm`` output is
        unit-std, a decoder's token embeddings are ~1e-2 std, so the default
        keeps the injected rows in the base's distribution instead of ~50-100x
        above it. Learnable, so training can grow it.
    init_seed : int
        Seed the projector's weights are initialized from. Every rank holds a
        replica whose gradients are reduced together, so the initialization must
        not depend on the ambient RNG state a rank happens to be in.
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
    output_scale: float = 0.01
    init_seed: int = 0

    @model_validator(mode="after")
    def _reject_silently_wrong_settings(self) -> "ProjectorSpec":
        """Both of these produce a run that trains on garbage without erroring.

        A packed-batch key is rebased onto the sample's start in the packed
        sequence only when the key's name ends in ``_positions``; under any other
        name the offset is skipped and every sample's embeddings land on the
        first sample's tokens.
        """
        if self.output_scale <= 0:
            raise ValueError(
                f"output_scale={self.output_scale} must be positive: it is the "
                "initial gain of the projector's output LayerNorm, and zero "
                "would start training from a projector that writes nothing."
            )
        if not self.positions_key.endswith("_positions"):
            raise ValueError(
                f"positions_key={self.positions_key!r} must end in '_positions': "
                "each sample's offset in the packed batch is added only to keys "
                "with that suffix, and the projector's positions have to end up "
                "as absolute offsets into the packed sequence."
            )
        return self

    def to_args_dict(self) -> dict[str, int | float | str | None]:
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
            "output_scale": self.output_scale,
            "init_seed": self.init_seed,
        }


def spec_from_args(args: object, args_key: str, framework: str) -> ProjectorSpec:
    """Rebuild the spec the recipe serialized, from the trainer's parsed args."""
    raw = vars(args).get(args_key)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{framework} arg '{args_key}' is missing or not a dict (got {raw!r}). "
            "A projector-only run needs the recipe to serialize a ProjectorSpec "
            "into extra_config."
        )
    return ProjectorSpec(**raw)


def should_save_projector(
    steps_applied: int, steps_attempted: int, total_steps: int, save_interval: int
) -> bool:
    """Whether the projector should be written, after an optimizer step.

    The run's last step always saves: a finished run has to leave the adapter it
    spent its GPU budget producing, whether or not the interval happens to
    divide the step count.

    Which step is the last one is decided by ``steps_attempted`` against
    ``total_steps`` (the trainer's ``train_iters``, a count of attempts), while
    the interval counts ``steps_applied`` — so ``save_interval`` means what it
    says and skipped updates do not shift it. Counting only applied steps for
    both would lose the whole artifact on a run whose interval saves nothing and
    where one update was skipped for a gradient overflow: the count would end
    one short of the total, the final-step guarantee would never fire, and the
    run would exit having written nothing.

    Nothing is written before an update has applied, so a projector still at its
    initialization is never mistaken for a trained one.
    """
    if steps_applied <= 0:
        return False
    if total_steps > 0 and steps_attempted >= total_steps:
        return True
    return save_interval > 0 and steps_applied % save_interval == 0
