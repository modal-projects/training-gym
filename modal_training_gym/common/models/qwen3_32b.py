"""Qwen3-32B model spec as a concrete ModelConfiguration subclass.

verl reads the architecture from the HF config (via its `converter_hf_to_mcore`
script) at checkpoint-conversion time, so `architecture` is left as the default
`None`. Set `architecture = ModelArchitecture(...)` on a further subclass if
you need to use this model with a framework that consumes
`--num-layers`/`--hidden-size` as CLI flags (e.g. SLIME).
"""

from .base import ModelConfiguration


class Qwen3_32B(ModelConfiguration):
    model_name = "Qwen/Qwen3-32B"

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name)
