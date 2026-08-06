"""Validating a model by running base training on slime."""

from __future__ import annotations

from typing import ClassVar

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.qwen3_asr_1_7b import Qwen3_ASR_1_7B
from modal_training_gym.common.models.validation import (
    ValidationFramework,
    ValidationTarget,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

from . import RecipeOverrides, ValidationBackend

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


class SlimeValidationBackend(ValidationBackend):
    framework: ClassVar[ValidationFramework] = ValidationFramework.SLIME

    supported_overrides: ClassVar[frozenset[str]] = frozenset(
        {"eval_interval", "save_interval", "non_colocated"}
    )

    def _build_recipe(
        self,
        target: ValidationTarget,
        model_config: ModelConfig,
        step_count: int,
        overrides: RecipeOverrides,
    ) -> SlimeRecipe:
        recipe = SlimeRecipe.get_base_recipe(model_config)
        if overrides.non_colocated:
            recipe.colocate = False
            if recipe.rollout_num_gpus is None:
                recipe.rollout_num_gpus = (
                    recipe.actor_num_nodes * recipe.actor_num_gpus_per_node
                )
        recipe.rm_type = "deepscaler"
        recipe.train_function_kwargs = {
            **dict(recipe.train_function_kwargs or {}),
            "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
        }
        return recipe

    def pick_dataset(
        self,
        target: ValidationTarget,
        model_config: ModelConfig,
        recipe: SlimeRecipe,
        step_count: int,
    ) -> DatasetConfig:
        """Pick a validation dataset matching the base model's modality.

        Audio models (Qwen3-ASR) need speech clips, so they get LibriSpeech;
        everything else defaults to gsm8k.
        """
        if isinstance(model_config, Qwen3_ASR_1_7B):
            return LibriSpeechASRDataset(n_rows=8)
        return Gsm8kDataset(n_rows=10)
