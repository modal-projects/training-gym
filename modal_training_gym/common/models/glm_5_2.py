"""GLM-5.2 (744B-A40B MoE, DSA attention) model specs."""

from .base import HFModelConfiguration, parse_glm_response


class GLM_5_2(HFModelConfiguration):
    """GLM-5.2 (744B total, ~40B active) MoE model from Z.ai.

    256 routed experts + 1 shared expert, 8 active per token, the first 3
    layers dense, and DeepSeek-style sparse attention (``glm_moe_dsa``) with
    a cross-layer index. ``architecture`` is left unset: the Megatron args
    live in miles' own ``scripts/models/glm5.2-744B-A40B.py`` (selected by
    ``miles_model_name`` on the recipe), because the DSA spec, the indexer
    and the MoE routing are not representable in ``ModelArchitecture``.
    """

    response_parser = staticmethod(parse_glm_response)

    model_name = "zai-org/GLM-5.2"


class GLM_5_2_5Layer(HFModelConfiguration):
    """5-layer pruned GLM-5.2, upstream's single-node smoke-test checkpoint.

    Same config family as :class:`GLM_5_2` (``glm_moe_dsa``, 256 experts) with
    the decoder truncated to 5 layers, so a full train step fits on one node.
    Outputs are meaningless — it exists to exercise plumbing, not quality.
    """

    response_parser = staticmethod(parse_glm_response)

    model_name = "Pinaster/GLM-5.2_5layer"
