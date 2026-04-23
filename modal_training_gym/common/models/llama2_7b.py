"""Llama-2-7B model spec as a concrete HFModelConfiguration subclass.

The torchrun / accelerate launchers pass `model_name` to a user-supplied
training script (via `transformers.AutoModelForCausalLM.from_pretrained`), so
the script reads architecture from the HF config at runtime. `architecture`
is left as the default `None`.
"""

from .base import HFModelConfiguration


class Llama2_7B(HFModelConfiguration):
    """Llama 2 7B from Meta.

    No ``architecture`` is set — the torchrun/accelerate launchers read
    it from the HF config via ``AutoModelForCausalLM.from_pretrained``.
    """

    model_name = "meta-llama/Llama-2-7b-hf"
