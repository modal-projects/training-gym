"""Llama-2-7B model spec as a concrete ModelConfiguration subclass.

The torchrun / accelerate launchers pass `model_name` to a user-supplied
training script (via `transformers.AutoModelForCausalLM.from_pretrained`), so
the script reads architecture from the HF config at runtime. `architecture`
is left as the default `None`.
"""

from .base import ModelConfiguration


class Llama2_7B(ModelConfiguration):
    model_name = "meta-llama/Llama-2-7b-hf"

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name)
