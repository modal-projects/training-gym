"""Moonlight-16B-A3B-Instruct model configuration."""

from .base import HFModelConfiguration


class Moonlight_16B_A3B_Instruct(HFModelConfiguration):
    """Moonshot AI's 16B-total, 3B-active Moonlight MoE instruct model."""

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    architecture_overrides = {
        "qk_layernorm": True,
        "no_masked_softmax_fusion": True,
        "moe_layer_freq": "[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]",
        "moe_grouped_gemm": True,
        "moe_router_pre_softmax": True,
        "moe_router_score_function": "sigmoid",
        "moe_router_enable_expert_bias": True,
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_router_bias_update_rate": 0,
        "moe_router_group_topk": 1,
        "moe_router_num_groups": 1,
        "moe_router_topk_scaling_factor": 2.446,
        "moe_token_drop_policy": "probs",
        "moe_router_dtype": "fp32",
        "moe_permute_fusion": True,
        "moe_aux_loss_coeff": 0.0,
        "megatron_model_type": "moonlight",
        "rotary_scaling_factor": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "no_rope_fusion": True,
    }
