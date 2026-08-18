"""Qwen3-VL-8B-Instruct model spec as a concrete HFModelConfiguration subclass.

Qwen3-VL is a vision-language model served by SGLang on ``/v1/chat/completions``
with image content. Its text backbone is a dense Qwen3-8B decoder; the vision
tower (a ViT with patch size 16, depth 27) is loaded by SGLang straight from the
HF checkpoint. The architecture below (from ``config.json`` → ``text_config``)
drives Megatron *training*.

The class holds only the model's specs. Framework wiring (the slime MB→HF
converters that map the vision tower for export) lives in the slime layer instead,
since it's meaningless for other backends.
"""

from __future__ import annotations

from .base import HFModelConfiguration, parse_qwen3_response


class Qwen3_VL_8B(HFModelConfiguration):
    """Qwen3-VL-8B-Instruct (vision-language, 8B parameters) from Alibaba.

    The text-backbone architecture is resolved from ``config.json``. The vision
    tower is frozen during RL training and handled by SGLang for rollouts.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    # Image patches expand prompts into many tokens; padded (bshd) batches avoid
    # the THD packing path that VL models may not support in megatron-bridge.
    requires_bshd = True

    architecture_overrides = {
        "qk_layernorm": True,
        "untie_embeddings_and_output_weights": True,
    }
