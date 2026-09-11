"""Validating a model by running base training on slime."""

from __future__ import annotations

from modal_training_gym.common.dataset import (
    DatasetConfig,
    MultimodalDataset,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


class Gsm8kDataset(DatasetConfig):
    def __init__(self, *, n_rows: int = 10) -> None:
        self.n_rows = n_rows

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.select(range(min(self.n_rows, len(dataset))))
        for row in dataset:
            yield {
                "messages": [{"role": "user", "content": row["question"]}],
                "label": row["answer"].split("####")[-1].strip(),
            }


class LibriSpeechASRDataset(MultimodalDataset):
    """LibriSpeech ASR rows (prompt + audio path + transcript label).

    Mirrors the audio_asr tutorial dataset. Audio models validate on a handful
    of LibriSpeech clips. gsm8k is text-only.
    """

    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, *, n_rows: int = 8):
        self.n_rows = n_rows
        super().__init__(modality="audio")

    def apply_chat_template(self) -> bool:
        return False

    def source_rows(self):
        import io
        import tempfile
        from pathlib import Path

        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        cache = Path(tempfile.gettempdir()) / "training-gym-asr" / self.dataset_id
        cache.mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(ds):
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else open(audio["path"], "rb").read()
            )
            arr, sr = sf.read(io.BytesIO(data))
            wav_path = cache / f"{i:06d}.wav"
            sf.write(wav_path, arr, sr, format="WAV")
            yield {
                "prompt": self._INSTRUCTION,
                "media": str(wav_path),
                "label": ex["text"].lower().strip(),
            }


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
