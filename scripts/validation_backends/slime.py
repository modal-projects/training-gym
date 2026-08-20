"""Validating a model by running base training on slime."""

from __future__ import annotations

from typing import Literal

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


class Gsm8kDataset(HuggingFaceDataset):
    hf_repo = "openai/gsm8k"
    hf_config = "main"
    input_column = "question"
    output_column = "answer"

    @property
    def output_format(self) -> Literal["jsonl"]:
        return "jsonl"

    @property
    def needs_refresh(self) -> bool:
        return True

    def _load_hf_dataset(self):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.map(lambda r: {"answer": r["answer"].split("####")[-1].strip()})
        return ds.map(self._to_chat, remove_columns=ds.column_names)


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio data-URI + transcript label).

    Mirrors the audio_asr tutorial dataset. Audio models validate on a handful
    of LibriSpeech clips. gsm8k is text-only.
    """

    modality = "audio"
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"
    n_rows = 8

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, *, n_rows=None):
        if n_rows is not None:
            self.n_rows = n_rows
        super().__init__(rows=[])

    @property
    def needs_refresh(self):
        return True

    @property
    def needs_chat_template(self):
        return False

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

    def rows(self):
        return self._build_rows()


def build_slime_validation(
    model_config: ModelConfig, step_count: int
) -> tuple[SlimeRecipe, DatasetConfig]:
    """The model's base slime recipe and a dataset matching its modality.

    Audio models (Qwen3-ASR) need speech clips, so they get LibriSpeech;
    everything else validates against gsm8k, scored by ``deepscaler``.
    """
    recipe = SlimeRecipe.get_base_recipe(model_config)
    recipe.rm_type = "deepscaler"
    recipe.train_function_kwargs = {
        **dict(recipe.train_function_kwargs or {}),
        "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
    }

    if isinstance(model_config, Qwen3_ASR_1_7B):
        return recipe, LibriSpeechASRDataset(n_rows=8)
    return recipe, Gsm8kDataset(n_rows=10)
