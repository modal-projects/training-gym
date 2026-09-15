from __future__ import annotations

import io
import struct
import tempfile
import zlib
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
                "messages": [{"role": "user", "content": row["question"]}],
                "label": row["answer"].split("####")[-1].strip(),
            }


class LibriSpeechASRDataset(MultimodalDataset):
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"

    _INSTRUCTION = (
        "<audio>\nTranscribe the speech to text. Respond with only the transcript."
    )

    def __init__(self, *, n_rows: int = 8):
        self.n_rows = n_rows
        super().__init__(modality="audio")

    def source_rows(self):
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
                else Path(audio["path"]).read_bytes()
            )
            arr, sr = sf.read(io.BytesIO(data))
            wav_path = cache / f"{i:06d}.wav"
            sf.write(wav_path, arr, sr, format="WAV")
            yield {
                "prompt": self._INSTRUCTION,
                "media": str(wav_path),
                "label": ex["text"].lower().strip(),
            }


def _solid_png(width: int = 224, height: int = 224) -> bytes:
    """RGB PNG large enough for Qwen VL resize (factor 28)."""
    raw = b"".join(b"\x00" + bytes([128, 128, 128]) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class Gsm8kImageDataset(MultimodalDataset):
    def __init__(self, *, n_rows: int = 10) -> None:
        self.n_rows = n_rows
        super().__init__(modality="image")

    def source_rows(self):
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.select(range(min(self.n_rows, len(dataset))))
        cache = Path(tempfile.gettempdir()) / "training-gym-mm" / self.dataset_id
        cache.mkdir(parents=True, exist_ok=True)
        png = _solid_png()
        for i, row in enumerate(dataset):
            img_path = cache / f"{i:06d}.png"
            img_path.write_bytes(png)
            yield {
                "prompt": f"<image>\n{row['question']}\n{GSM8K_INSTRUCTION}",
                "media": str(img_path),
                "label": row["answer"].split("####")[-1].strip(),
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
        return Gsm8kImageDataset(n_rows=10 if n_rows is None else n_rows)
    if modality == "audio":
        return LibriSpeechASRDataset(n_rows=8 if n_rows is None else n_rows)
    raise TrainingGymConfigError(f"no validation dataset for modality {modality!r}")
