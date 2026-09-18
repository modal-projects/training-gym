"""Generic audio helpers shared across audio tasks and frameworks."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def data_uri_to_bytes(data_uri: str) -> bytes:
    if data_uri.startswith("data:"):
        _, _, b64 = data_uri.partition(",")
    else:
        b64 = data_uri
    return base64.b64decode(b64, validate=True)


def coerce_audio_to_bytes(value: Any) -> bytes | None:
    """Coerce one audio reference to raw bytes.

    Accepts ``bytes``, a ``pathlib.Path``, a ``data:`` URI, a raw base64
    string, or a 1-element list/tuple of any of those. Returns ``None``
    when *value* carries no usable audio.
    """
    first = value[0] if isinstance(value, (list, tuple)) and value else value
    if isinstance(first, (bytes, bytearray)):
        return bytes(first)
    if isinstance(first, Path):
        try:
            return first.read_bytes() if first.is_file() else None
        except OSError:
            return None
    if isinstance(first, str) and first:
        try:
            return data_uri_to_bytes(first)
        except binascii.Error:
            return None
    return None


def decode_to_mono(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode encoded audio to a mono float32 waveform at *target_sr* Hz."""
    import io

    import soundfile as sf

    array, sr = sf.read(io.BytesIO(audio_bytes))
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        import librosa

        array = librosa.resample(
            array.astype("float32"), orig_sr=sr, target_sr=target_sr
        )
    return array
