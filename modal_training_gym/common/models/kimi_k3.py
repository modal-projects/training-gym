"""Kimi-K3 model spec as a concrete HFModelConfiguration subclass.

Kimi-K3 is Moonshot's 93-layer, 7168-hidden hybrid: KDA (Kimi Delta
Attention, a gated linear-attention layer) and MLA chosen per layer, an
attention-residual snapshot bank, and an 896-expert latent MoE (top-16,
``routed_expert_hidden_size`` 3584, two shared experts). The release
``moonshotai/Kimi-K3`` is multimodal (``KimiK3ForConditionalGeneration``) and
ships its routed experts as ``mxfp4-pack-quantized`` compressed-tensors;
everything else is bf16.

Neither KDA nor the residual bank is representable as a ``ModelArchitecture``,
so ``architecture`` is left ``None`` and the Miles recipe renders upstream's
``scripts/models/kimi-k3.py`` via ``miles_model_name`` — including its
``--spec miles_plugins.models.kimi_k3 get_kimi_k3_spec``. Same arrangement as
``DeepSeek_V4_1_Flash``.

The rollout engines serve the native MXFP4 checkpoint through sglang's Marlin
MoE runner; the trainer side is dequantized to bf16 while the torch_dist
checkpoint is converted (``hf_mxfp4_dequant``).
"""

from __future__ import annotations

from .base import HFModelConfiguration


class Kimi_K3(HFModelConfiguration):
    """Moonshot Kimi-K3 hybrid KDA/MLA MoE model, 93 layers and 896 routed experts."""

    model_name = "moonshotai/Kimi-K3"
