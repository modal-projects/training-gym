"""Gemma-4-26B-A4B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from .base import HFModelConfiguration, parse_gemma4_response


class Gemma4_26B_A4B(HFModelConfiguration):
    """Gemma-4-26B-A4B-it (25.2B total, ~3.8B active) MoE model from Google.

    Mixture-of-Experts with 128 experts, 8 active per token plus 1 shared.
    The checkpoint is a ``Gemma4ForConditionalGeneration``: the MoE decoder
    described by ``architecture`` plus a 27-layer vision tower.
    Downloads from ``google/gemma-4-26B-A4B-it`` on HuggingFace.
    """

    model_name = "google/gemma-4-26B-A4B-it"
    response_parser = staticmethod(parse_gemma4_response)

    # The recipe sets ``miles_model_script``; these overrides drive its
    # expert-parallel validator and preserve the script's model contract.
    architecture_overrides = {
        "qk_layernorm": True,
        "moe_grouped_gemm": True,
        "moe_router_topk": 8,
        "moe_router_score_function": "softmax",
        "moe_router_dtype": "fp32",
        "moe_aux_loss_coeff": 0.0,
        "rotary_base": 1000000,
    }
