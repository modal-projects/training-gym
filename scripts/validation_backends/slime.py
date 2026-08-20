"""Validating a model by running base training on slime."""

from __future__ import annotations

from modal_training_gym.common.dataset import (
    DatasetConfig,
    EmbeddingProjectorDataset,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.qwen3_6_35b import Qwen3_6_35B
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b_projector import (
    Qwen3_6_35b_Projector_Recipe,
)

# Which model each projector-only slime recipe trains a projector against.
# ``get_base_recipe`` cannot answer this: these models also train RL, and that
# is the recipe it returns for them.
PROJECTOR_RECIPES: dict[type[ModelConfig], type[SlimeRecipe]] = {
    Qwen3_6_35B: Qwen3_6_35b_Projector_Recipe,
}

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


class Gsm8kDataset(HuggingFaceDataset):
    hf_repo = "openai/gsm8k"
    hf_config = "main"
    input_column = "question"
    output_column = "answer"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True

    def load(self, split: str = "all"):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds.map(lambda r: {"answer": r["answer"].split("####")[-1].strip()})


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio data-URI + transcript label).

    Mirrors the 006_audio_asr tutorial dataset: audio models can't train on
    gsm8k, so they validate against a handful of LibriSpeech clips instead.
    """

    modality = "audio"
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"
    n_rows = 8
    always_prepare = True
    apply_chat_template = False

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, **kwargs):
        super().__init__(rows=[], **kwargs)

    def _build_rows(self) -> list[dict]:
        import base64 as b64
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        rows = []
        for ex in ds:
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else open(audio["path"], "rb").read()
            )
            arr, sr = sf.read(io.BytesIO(data))
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + b64.b64encode(buf.getvalue()).decode(
                "ascii"
            )
            rows.append(
                {
                    self.input_key: self._INSTRUCTION,
                    self.media_column: [data_uri],
                    self.label_key: ex["text"].lower().strip(),
                }
            )
        return rows

    def load(self, split: str = "all") -> list[dict]:
        return self._build_rows()

    def prepare(self, path, eval_paths=None):
        rows = self._build_rows()
        self._write_jsonl(rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_jsonl(rows, eval_path)


def build_slime_validation(
    model_config: ModelConfig, step_count: int, projector: bool = False
) -> tuple[SlimeRecipe, DatasetConfig]:
    """The model's slime recipe and a dataset matching what it trains.

    Audio models (Qwen3-ASR) need speech clips, so they get LibriSpeech;
    everything else validates against gsm8k, scored by ``deepscaler``.

    A projector-only entry is the exception twice over: the recipe comes from
    ``PROJECTOR_RECIPES`` rather than ``get_base_recipe``, and it trains
    supervised on external embeddings that no prompt dataset carries, so it
    validates on synthetic rows sized to the projector's input dimension.
    """
    if projector:
        recipe_class = PROJECTOR_RECIPES.get(type(model_config))
        if recipe_class is None:
            raise TrainingGymConfigError(
                f"no projector-only slime recipe for model "
                f"{model_config.model_name!r}, which is registered as a "
                "projector validation target"
            )
        projector_recipe = recipe_class()
        projector_recipe.train_function_kwargs = {
            **dict(projector_recipe.train_function_kwargs or {}),
            "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
        }
        return projector_recipe, EmbeddingProjectorDataset.synthetic(
            n_rows=projector_recipe.rollout_batch_size * step_count,
            input_dim=projector_recipe.projector.input_dim,
        )

    recipe = SlimeRecipe.get_base_recipe(model_config)
    recipe.rm_type = "deepscaler"
    recipe.train_function_kwargs = {
        **dict(recipe.train_function_kwargs or {}),
        "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
    }

    if isinstance(model_config, Qwen3_ASR_1_7B):
        return recipe, LibriSpeechASRDataset(n_rows=8)
    return recipe, Gsm8kDataset(n_rows=10)
