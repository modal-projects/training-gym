"""GLM-4.7 model spec — registered on import.

ms-swift reads the architecture from the HF config at train time, so the
architecture fields aren't used for this family. A default `ModelArchitecture`
is registered so `Model(BaseModelType.GLM_4_7)` still resolves.
"""

from .base import BaseModelType, ModelArchitecture, register_model

GLM_4_7_ARCHITECTURE = ModelArchitecture()

register_model(
    BaseModelType.GLM_4_7,
    hf_checkpoint="zai-org/GLM-4.7",
    architecture=GLM_4_7_ARCHITECTURE,
)
