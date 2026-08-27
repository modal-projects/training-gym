from collections.abc import Callable
from dataclasses import field
from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.patches import encode_patch
from modal_training_gym.frameworks.slime.audio_transcription_rollout import (
    transcription_rollout,
)
from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

# Training-time image-build shims for upstream gaps that block Qwen3-ASR on the
# native slime stack (bridge config validate-order, slime processor loading, bridge
# pg_collection). These are slime-specific, so they live on the recipe (not the
# model) and are applied via ``image_run_commands``. The Megatron->HF audio-tower
# converter (``patch_qwen3_asr_export``) instead lives in the slime base image, since
# conversion also runs in the recipe-less deploy/eval path. Each should be reported
# upstream; once fixed, drop the corresponding patch.
_ASR_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "slime"
    / "modal_helpers"
    / "patches"
    / "model_specific_patches"
    / "qwen3_asr"
)
_ASR_PATCHES = (
    "patch_qwen3_asr_bridge_config",
    "patch_qwen3_asr_processor",
    "patch_qwen3_asr_pg_collection",
    "patch_qwen3_asr_packed_seq",
)


def _asr_image_run_commands() -> list[str]:
    # soundfile/librosa decode audio (transcription rollout); jiwer powers the −WER
    # reward. Then apply the upstream-gap shims at image build.
    cmds = ["pip install --no-cache-dir jiwer librosa soundfile"]
    cmds += [
        f"echo {encode_patch(name, _ASR_PATCH_DIR)} | base64 -d | python3"
        for name in _ASR_PATCHES
    ]
    return cmds


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_ASR_1_7b_Recipe(SlimeRecipe):
    """Qwen3-ASR-1.7B audio GRPO on 1×2×H100, colocated.

    Carries the ASR-specific defaults so a user only sets the reward (and W&B):

      recipe = Qwen3_ASR_1_7b_Recipe(custom_rm_function=word_error_rate_reward, metrics=...)

    What's baked in and why:

    * ``custom_generate_function=transcription_rollout`` — Qwen3-ASR is served by
      SGLang on ``/v1/audio/transcriptions`` (never chat completions), so it must
      be driven through the slime audio-transcription rollout.
    * ``use_dynamic_batch_size=False`` + ``qkv_format="bshd"`` + ``micro_batch_size=1``
      — the native megatron-bridge Qwen3-ASR forward doesn't implement THD sequence
      packing. As of nightly-dev-20260701a, slime builds packed_seq_params
      regardless of qkv_format, so a build-time patch
      (``patch_qwen3_asr_packed_seq``) nullifies it in the thinker forward. bshd
      still needs dynamic batching off + an explicit micro_batch_size. The launcher
      enforces this (``model.requires_bshd``).
    * ``sglang_mem_fraction_static=0.45`` — audio conditioning lengthens prompts
      (expanded ``<audio_pad>``) and adds the frozen audio tower, so free SGLang
      memory for the colocated actor (text-only runs use ~0.78).
    * Many samples/prompt + temperature 1.0 — Qwen3-ASR is near-deterministic on
      clean speech, so this is how the GRPO group gets nonzero reward variance.
    * ``image_run_commands`` installs the audio deps and applies the upstream-gap
      shims; ``megatron_to_hf_mode="bridge"`` loads HF directly (no torch_dist
      pre-conversion) since the image's megatron.bridge maps Qwen3-ASR.

    Scale to a full 8×H100 node by passing ``actor_num_gpus_per_node=8`` (and a
    larger ``num_rollout``); everything else holds.
    """

    sequence_parallel: bool = False

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 2

    custom_generate_function: Callable | None = transcription_rollout

    num_rollout: int = 8
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 128
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.45

    global_batch_size: int = 8
    lr: float = 1e-6
    lr_decay_style: str = "cosine"

    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    # Save at the final rollout so the run produces a checkpoint to export to HF.
    save_interval: int = 8
    megatron_to_hf_mode: str = "bridge"

    image_run_commands: list[str] = field(default_factory=_asr_image_run_commands)
