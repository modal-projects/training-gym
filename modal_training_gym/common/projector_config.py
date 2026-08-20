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
        Keep a numbered projector checkpoint every N optimizer steps.
        ``projector_latest.pt`` is refreshed after every applied step whatever
        this is, so a finished run always leaves the adapter it trained.
    load : str
        Projector checkpoint (file or directory) to resume from. The resumed
        run continues that checkpoint's iteration numbering, so reusing one
        ``save_dir`` across resumes does not overwrite what came before.
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
    """Rebuild the spec the recipe serialized, from the trainer's parsed args.

    The pinned images flatten every ``extra_config`` entry onto ``args``, which
    is where this looks first. The nested containers are read as a fallback for
    the same reason the gym's other in-container readers do
    (:func:`modal_training_gym.common.reporting._arg_value`): a run must not
    lose its projector to a config-plumbing change that keeps the dict nested.
    """
    candidates = [getattr(args, args_key, None)]
    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            candidates.append(container.get(args_key))

    for raw in candidates:
        if isinstance(raw, dict):
            return ProjectorSpec(**raw)

    raise ValueError(
        f"{framework} arg '{args_key}' is missing or not a dict (looked at args."
        f"{args_key}, args.extra_config and args.custom_config, got "
        f"{candidates!r}). A projector-only run needs the recipe to serialize a "
        "ProjectorSpec into extra_config."
    )


def should_save_projector(
    steps_applied: int, steps_attempted: int, total_steps: int, save_interval: int
) -> bool:
    """Whether this optimizer step keeps its own numbered projector checkpoint.

    ``projector_latest.pt`` is refreshed after every applied step regardless, so
    a finished run always leaves the adapter it spent its GPU budget producing;
    this decides only which steps additionally keep a ``projector_iter_*.pt``.
    That split is deliberate: ``total_steps`` is the trainer's ``train_iters``,
    which it derives arithmetically from the rollout counts, and an epoch-driven
    dynamically batched run need not perform exactly that many optimizer steps —
    so nothing that matters may depend on the prediction being right.

    The interval counts ``steps_applied`` (so ``save_interval`` means what it
    says and an overflow-skipped update does not shift it), while the predicted
    last step is matched on ``steps_attempted``, which is what ``train_iters``
    counts. It is an equality, not ``>=``: a run that outlives the prediction
    should not then keep a numbered checkpoint on every step.

    Nothing is written before an update has applied, so a projector still at its
    initialization is never mistaken for a trained one.
    """
    if steps_applied <= 0:
        return False
    if total_steps > 0 and steps_attempted == total_steps:
        return True
    return save_interval > 0 and steps_applied % save_interval == 0


# Both trainers carry the recipe's own hook under this ``extra_config`` key while
# the ``--custom-megatron-before-train-step-hook-path`` flag names the gym's
# phase-reporting wrapper that dispatches to it.
STEP_HOOK_KEY = "training_gym_custom_megatron_before_train_step_hook_path"


def step_hook_path(args) -> str | None:
    """Resolve the hook the gym's reporting wrapper will dispatch to.

    Mirrors ``phase_reporting._hook_path_from_args``: the value travels in
    ``extra_config``, whose keys the trainer sets on ``args``, and is also
    looked for on ``args`` and ``custom_config`` directly.
    """
    direct = getattr(args, STEP_HOOK_KEY, None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for container_name in ("extra_config", "custom_config"):
        container = getattr(args, container_name, None)
        if isinstance(container, dict):
            value = container.get(STEP_HOOK_KEY) or container.get(
                STEP_HOOK_KEY.removeprefix("training_gym_")
            )
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def require_step_hook(args, save_hook_path: str, recipe_name: str) -> None:
    """Fail at model construction when the projector's step hook is unwired.

    ``save_projector_checkpoint`` is the only thing that installs the
    ``optimizer.step`` wrapper, and that wrapper owns two jobs nothing else
    does: the tensor-parallel all-reduce of the replicated projector's
    gradients, and writing the projector checkpoints. Neither announces its
    absence, so a run whose hook never fires would train on
    tensor-parallel-partial gradients and finish with nothing saved. The wiring
    is a recipe default, but the field is assignable and reaches the container
    through ``extra_config``, so the provider checks it — a run only exists if
    the trainer called the provider — rather than trusting it.
    """
    wrapper = getattr(args, "custom_megatron_before_train_step_hook_path", None)
    wrapper = wrapper.strip() if isinstance(wrapper, str) else ""
    wrapped = step_hook_path(args)
    if save_hook_path in (wrapper, wrapped):
        return
    raise ValueError(
        "projector-only training needs the before-train-step hook to reach "
        f"'{save_hook_path}', which installs the projector's gradient "
        "all-reduce and checkpoint writer, but this run resolves "
        f"--custom-megatron-before-train-step-hook-path to '{wrapper}' and the "
        f"gym's wrapped hook to '{wrapped}'. Launch through "
        f"{recipe_name} (or point custom_megatron_before_train_step_hook at "
        "that path) rather than training with unreduced gradients and no "
        "checkpoints."
    )
