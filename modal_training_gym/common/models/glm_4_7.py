"""GLM-4.7 (355B-A32B MoE) model spec as a concrete HFModelConfiguration subclass."""

import subprocess

from .base import (
    HFModelConfiguration,
    disable_mtp_in_config,
    parse_glm_response,
)

_TOOLS_PATH = "/opt/training-gym/tools"


class GLM_4_7(HFModelConfiguration):
    """GLM-4.7 (355B total, ~32B active) MoE model from Zhipu AI.

    Mixture-of-Experts with 160 routed experts + 1 shared expert,
    8 active per token. First 3 layers are dense.
    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_glm_response)

    model_name = "zai-org/GLM-4.7"
    architecture_overrides = {
        "untie_embeddings_and_output_weights": True,
        "moe_grouped_gemm": True,
        "moe_shared_expert_gate": True,
        "moe_router_score_function": "softmax",
        "moe_token_drop_policy": "probs",
        "moe_router_dtype": "fp32",
        "moe_permute_fusion": True,
        "moe_aux_loss_coeff": 0,
        "rotary_percent": 1.0,
    }

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(repo_id=self.model_name)
        subprocess.check_call(
            [
                "python3",
                f"{_TOOLS_PATH}/ensure_glm_tokenizer.py",
                "--snapshot-dir",
                snapshot_dir,
            ],
        )
        # The Slime model provider reads num_nextn_predict_layers via the
        # megatron-bridge ``AutoBridge.from_hf_pretrained`` path but does NOT
        # override it from CLI args. With PP > 1 the MTP embedding on the last
        # pipeline stage collides with the main embedding on the first stage
        # during ``broadcast_from_pp_rank``; zeroing the field prevents the
        # bridge from creating MTP layers.
        disable_mtp_in_config(snapshot_dir, "glm_4_7")
