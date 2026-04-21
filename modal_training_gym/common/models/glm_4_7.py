"""GLM-4.7 model spec as a concrete ModelConfiguration subclass.

ms-swift reads the architecture from the HF config at train time, so
`architecture` is left as the default `None`.
"""

from .base import ModelConfiguration


class GLM_4_7(ModelConfiguration):
    model_name = "zai-org/GLM-4.7"

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name)
