"""GLM-4.7 model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelTrainingConfig


class GLM_4_7(HFModelConfiguration):
    """GLM-4.7 large MoE model from Zhipu AI.

    Training config is tuned for 4×H100 nodes (32 GPUs) with MoE
    parallelism.
    """

    model_name = "zai-org/GLM-4.7"
    training = ModelTrainingConfig(
        gpu_type="H100",
        n_nodes=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_model_parallel_size=4,
        moe_permute_fusion=True,
        moe_grouped_gemm=True,
        moe_shared_expert_overlap=True,
        moe_aux_loss_coeff=1e-3,
        lora_rank=128,
        lora_alpha=32,
    )
