"""Dequantize DeepSeek block-scaled HF checkpoints as mbridge reads them.

DeepSeek's native checkpoints (V4.1 onwards) ship fp8 e4m3 dense weights with
(32, 32)-block e8m0 scales and packed-e2m1 fp4 routed experts with per-row
32-column e8m0 scales, each next to a sibling ``<name>.scale`` tensor. mbridge
maps HF names onto bf16 Megatron parameters and only reshapes: a raw fp8 tensor
is upcast without its scale and a packed fp4 tensor has half the columns the
parameter expects, so the load scatters mismatched shapes. This module wraps the
bridge's safetensor reader so every weight with a sibling scale comes back
dequantized to bf16 and the rest of the bridge sees a plain bf16 checkpoint.
"""

from __future__ import annotations

import torch

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP4_BLOCK = 32


def scale_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight") + ".scale"


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode e8m0 (biased power-of-two exponent) scales without relying on dtype casts."""
    if scale.dtype == torch.float32:
        return scale
    exponent = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.exp2(exponent.to(torch.float32))


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 weight [M, N] with one scale per (M/bm, N/bn) block -> bf16 [M, N]."""
    m, n = weight.shape
    bm, bn = m // scale.shape[0], n // scale.shape[1]
    scale = (
        e8m0_to_float(scale).repeat_interleave(bm, dim=0).repeat_interleave(bn, dim=1)
    )
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


def dequant_fp4_packed(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 weight [M, N/2] (low nibble first) with scale [M, N/32] -> bf16 [M, N]."""
    m, half = weight.shape
    packed = weight.view(torch.uint8)
    nibbles = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(m, half * 2)
    values = FP4_TABLE.to(weight.device)[nibbles.to(torch.long)]
    scale = e8m0_to_float(scale).repeat_interleave(FP4_BLOCK, dim=1)
    return (values * scale).to(torch.bfloat16)


def dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return dequant_fp8_block(weight, scale)
    if weight.dtype in (torch.int8, torch.uint8):
        return dequant_fp4_packed(weight, scale)
    raise TypeError(f"no block dequantization for {weight.dtype}")


QUANTIZED_DTYPES = (torch.float8_e4m3fn, torch.int8, torch.uint8)


def wrap_safetensor_io(io):
    """Make ``io.load_some_hf_weight`` return dequantized bf16 for block-scaled weights.

    Wraps the instance rather than a class: DeepSeek bridges swap in
    ``SafeTensorIO`` subclasses (the V3 ``_scale_inv`` dequantizer passes
    ``.scale`` checkpoints through untouched), so a class patch on the base
    would be shadowed by their overrides.
    """
    original = io.load_some_hf_weight
    available = set(io.index) if io.index else set(io.load_hf_weight_names())

    def load_some_hf_weight(hf_weight_names: list[str]) -> dict:
        scales = {
            name: scale_name(name)
            for name in hf_weight_names
            if scale_name(name) in available and scale_name(name) != name
        }
        loaded = original(list(hf_weight_names) + sorted(set(scales.values())))
        out = {}
        for name in hf_weight_names:
            weight = loaded[name]
            if name in scales and weight.dtype in QUANTIZED_DTYPES:
                weight = dequant(weight, loaded[scales[name]])
            out[name] = weight
        return out

    io.load_some_hf_weight = load_some_hf_weight
    return io


def install() -> None:
    """Hook ``Bridge.load_weights`` so the reader it builds is wrapped.

    ``_get_safetensor_io`` itself is overridden down the DeepSeek bridge MRO, so
    it is resolved through the instance at call time rather than patched on a class.
    """
    from mbridge.core.bridge import Bridge

    original_load_weights = Bridge.load_weights

    def load_weights(self, *args, **kwargs):
        get_io = self._get_safetensor_io
        self._get_safetensor_io = lambda weights_path: wrap_safetensor_io(
            get_io(weights_path)
        )
        print(
            "[hf_block_dequant] dequantizing block-scaled weights on load", flush=True
        )
        return original_load_weights(self, *args, **kwargs)

    Bridge.load_weights = load_weights
