"""Nemotron-3-Ultra-550B-A55B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_6_response


class Nemotron3_Ultra_550B_A55B(HFModelConfiguration):
    """NVIDIA Nemotron-3-Ultra-550B-A55B-BF16 (550 B total / 55 B active).

    The Ultra tier of the ``nemotron_h`` family: 108 layers of hybrid Mamba2 +
    attention + **latent** MoE (512 routed experts at top-22 plus one shared
    expert, with expert input/output bottlenecked through a 2048-dim latent),
    sigmoid routing with aux-free expert bias, no positional embeddings, and an
    MTP head the RL recipes do not train.

    Public, ungated, OpenMDW-1.1 — no ``huggingface-secret`` needed. ~1.12 TB of
    BF16 safetensors, so the first download into the shared ``huggingface-cache``
    volume dominates a cold run.

    ``parse_qwen3_6_response`` is reused rather than duplicated: Nemotron-3's
    chat template emits ``<|im_start|>``/``<|im_end|>`` turns, ``<think>``
    reasoning blocks and Qwen3-Coder-lineage
    ``<tool_call><function=NAME><parameter=KEY>`` tool bodies — the model card
    itself names ``tool_call_parser qwen3_coder`` — which is exactly the wire
    format that parser handles.
    """

    model_name = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    response_parser = staticmethod(parse_qwen3_6_response)

    # Mirrors upstream's scripts/models/nemotron-3-ultra-550b-a55b.py. The miles
    # recipe sets ``miles_model_name``, so none of this is emitted as CLI flags —
    # miles sources that script for ${MODEL_ARGS[@]} instead, and it stays the
    # single source of truth (ModelArchitecture has no field for the hybrid block
    # pattern, the Mamba dimensions, or ``moe_latent_size=2048``). What these do
    # drive is the expert-parallel validator, which needs ``num_experts``; keep
    # them in step with that script.
    architecture = ModelArchitecture(
        num_layers=108,
        hidden_size=8192,
        ffn_hidden_size=5120,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=2,
        kv_channels=128,
        vocab_size=131072,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=False,  # relu2, selected by the layer spec rather than a flag
        disable_bias_linear=True,
        qk_layernorm=False,
        untie_embeddings_and_output_weights=True,
        num_experts=512,
        moe_ffn_hidden_size=5120,
        moe_shared_expert_intermediate_size=10240,
        moe_grouped_gemm=True,
        moe_router_topk=22,
        moe_router_pre_softmax=True,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_bias_update_rate=0,
        # n_group=1 makes group-limited routing a no-op over a single group of
        # 512. The HF config's n_groups=8 is the *Mamba* group count, unrelated.
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_topk_scaling_factor=5.0,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
        # --position-embedding-type none: a hybrid Mamba/attention stack with no
        # rotary embeddings at all.
        use_rotary_position_embeddings=False,
    )
