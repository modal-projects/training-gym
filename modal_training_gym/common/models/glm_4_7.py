"""GLM-4.7 model spec as a concrete HFModelConfiguration subclass.

ms-swift reads the architecture from the HF config at train time, so
`architecture` is left as the default `None`.
"""

from .base import HFModelConfiguration


class GLM_4_7(HFModelConfiguration):
    """GLM-4.7 large MoE model from Zhipu AI.

    No ``architecture`` is set — ms-swift reads the architecture from
    the HF config at train time.
    """

    model_name = "zai-org/GLM-4.7"
