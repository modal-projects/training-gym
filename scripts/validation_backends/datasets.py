from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HuggingFaceDataset,
    MultimodalDataset,
)
from modal_training_gym.common.errors import TrainingGymConfigError

GSM8K_INSTRUCTION = (
    "You are a helpful assistant. Please put the answer within \\boxed{}."
)


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
                "messages": [
                    {
                        "role": "user",
                        "content": f"{row['question']}\n{GSM8K_INSTRUCTION}",
                    }
                ],
                "label": row["answer"].split("####")[-1].strip(),
            }


LIBRI_SPEECH_INSTRUCTION = (
    "<audio>\nTranscribe the speech to text. Respond with only the transcript."
)


class LibriSpeechASRDataset(MultimodalDataset):
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"

    def __init__(self, *, n_rows: int = 8):
        self.n_rows = n_rows
        super().__init__(modality="audio")

    def apply_chat_template(self) -> bool:
        return False

    def source_rows(self):
        import soundfile as sf
        from datasets import Audio, load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.select(range(min(self.n_rows, len(ds))))
        ds = ds.cast_column("audio", Audio(decode=False))
        cache = Path(tempfile.mkdtemp())
        for i, ex in enumerate(ds):
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else Path(audio["path"]).read_bytes()
            )
            arr, sr = sf.read(io.BytesIO(data))
            wav_path = cache / f"{i:06d}.wav"
            sf.write(wav_path, arr, sr, format="WAV")
            yield {
                "prompt": LIBRI_SPEECH_INSTRUCTION,
                "media": str(wav_path),
                "label": ex["text"].lower().strip(),
            }


SCREENSPOT_INSTRUCTION = (
    "<image>\n"
    "You are a GUI agent. Given the screenshot, click on the element "
    "described below.\n\n"
    "Instruction: {instruction}\n\n"
    "Respond with ONLY the normalized (x, y) coordinates of the click "
    "target, formatted as: (x, y)\n"
    "where x and y are decimals between 0 and 1 representing the "
    "horizontal and vertical position on the screen."
)


class ScreenSpotDataset(MultimodalDataset):
    """GUI grounding dataset from ScreenSpot."""

    hf_repo = "rootsautomation/ScreenSpot"
    hf_split = "test"

    def __init__(self, *, n_rows: int = 10, row_offset: int = 0):
        self.n_rows = n_rows
        self.row_offset = row_offset
        super().__init__(modality="image")

    def source_rows(self):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split=self.hf_split)
        start = min(self.row_offset, len(ds))
        stop = min(start + self.n_rows, len(ds))
        for row in ds.select(range(start, stop)):
            left, top, right, bottom = row["bbox"]
            instruction = row["instruction"]

            buf = io.BytesIO()
            row["image"].save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            data_uri = f"data:image/png;base64,{img_b64}"

            yield {
                "prompt": SCREENSPOT_INSTRUCTION.format(instruction=instruction),
                "media": data_uri,
                "label": f"{left:.4f},{top:.4f},{right:.4f},{bottom:.4f}",
            }


def dapo_math_dataset(*, n_rows: int) -> HuggingFaceDataset:
    return HuggingFaceDataset(
        "zhuzilin/dapo-math-17k",
        hf_split=f"train[:{n_rows}]",
        input_column="prompt",
        output_column="label",
        input_format="messages",
        always_download=True,
    )


def media_dataset(modality: str, n_rows: int | None = None) -> DatasetConfig:
    if modality == "image":
        return ScreenSpotDataset(n_rows=10 if n_rows is None else n_rows)
    if modality == "audio":
        return LibriSpeechASRDataset(n_rows=8 if n_rows is None else n_rows)
    raise TrainingGymConfigError(f"no validation dataset for modality {modality!r}")
