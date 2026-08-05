from __future__ import annotations

import fcntl
import shlex
import shutil
import subprocess
from dataclasses import field
from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.frameworks.miles.modal_helpers.utils import (
    resolve_checkpoint_ref,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


def _valid_safetensors(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from safetensors.torch import safe_open  # pyright: ignore[reportMissingImports]

        with safe_open(str(path), framework="pt") as reader:
            list(reader.keys())
    except Exception as exc:
        print(f"Invalid safetensors file {path}: {exc}")
        return False
    return True


def _valid_hf_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    if not root.is_dir() or not (root / "config.json").is_file():
        return False
    safetensors = sorted(root.glob("*.safetensors"))
    if not safetensors:
        return False
    return all(_valid_safetensors(p) for p in safetensors)


def _remove_if_invalid(path: str | Path) -> bool:
    root = Path(path)
    if not root.exists():
        return False
    if _valid_hf_checkpoint(root):
        return True
    print(f"Removing incomplete or invalid checkpoint directory: {root}")
    shutil.rmtree(root, ignore_errors=True)
    return False


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _KimiK2Recipe(MilesRecipe):
    gpu_type: str = "H200"
    memory: tuple[int, int] = (1024, int(2 * 1024 * 1024))
    image_run_commands: list[str] = field(
        default_factory=lambda: [
            "rm -rf /root/.cache/huggingface 2>/dev/null || true",
            "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
        ]
    )
    image_env: dict[str, str] = field(
        default_factory=lambda: {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
        }
    )
    miles_model_script: str = "scripts/models/kimi-k2-thinking.sh"
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_TIMEOUT": "3600",
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
        }
    )

    actor_num_nodes: int = 16
    actor_num_gpus_per_node: int = 8
    colocate: bool = True
    use_miles_router: bool = True
    skip_eval_before_train: bool = True
    update_weight_buffer_size: int = 4 * 512 * 1024 * 1024

    prompt_data: str = "/data/dapo-math-17k/dapo-math-17k.jsonl"
    input_key: str = "prompt"
    label_key: str = "label"
    apply_chat_template: bool = True
    rollout_shuffle: bool = True
    balance_data: bool = True
    rm_type: str = "deepscaler"

    num_rollout: int = 20
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 16384
    rollout_temperature: float = 1.0
    sglang_cuda_graph_bs: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 129, 8))
    )
    global_batch_size: int = 256

    advantage_estimator: str = "grpo"
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    optimizer: str = "adam"
    lr: float = 1e-5
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    use_distributed_optimizer: bool = True

    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    target_modules: str | None = (
        "q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj"
    )
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True
    sglang_lora_backend: str | None = "triton"
    sglang_lora_use_virtual_experts: bool = True
    use_tis: bool = True

    train_backend: str = "megatron"
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 8
    expert_model_parallel_size: int = 64
    expert_tensor_parallel_size: int = 1
    decoder_last_pipeline_num_layers: int = 30

    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    attention_backend: str = "flash"
    no_check_for_nan_in_loss_and_grad: bool = True

    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.7
    sglang_ep_size: int = 8
    sglang_server_concurrency: int = 1024
    use_rollout_routing_replay: bool = True

    def download_model(self) -> None:
        if self.source_hf_checkpoint:
            resolve_checkpoint_ref(self.source_hf_checkpoint, local_files_only=False)

    def _patch_kimi_source(self, model_dir: Path) -> None:
        model_file = model_dir / "modeling_kimi_k25.py"
        if not model_file.is_file():
            return
        src = model_file.read_text()
        if "use_deterministic_attn: bool = False" in src:
            return
        ctor_old = """    def __init__(
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = 'spatial_temporal') -> None:
"""
        ctor_new = """    def __init__(
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = 'spatial_temporal',
        use_deterministic_attn: bool = False,
) -> None:
"""
        layer_old = """            MoonViTEncoderLayer(
                **block_cfg,
                use_deterministic_attn=self.use_deterministic_attn)
"""
        layer_new = """            MoonViTEncoderLayer(
                **block_cfg,
                use_deterministic_attn=use_deterministic_attn)
"""
        if ctor_old in src and layer_old in src:
            src = src.replace(ctor_old, ctor_new, 1)
            src = src.replace(layer_old, layer_new, 1)
            model_file.write_text(src)

    def post_process_model(self) -> None:
        if not self.source_hf_checkpoint:
            raise TrainingGymConfigError("Kimi recipes require source_hf_checkpoint")

        source_hf_path = Path(
            resolve_checkpoint_ref(self.source_hf_checkpoint, local_files_only=False)
        )
        self._patch_kimi_source(source_hf_path)
        int4_path = Path(str(self.hf_checkpoint))
        bf16_path = Path(str(self.ref_load))
        lock_path = int4_path.parent / f".{int4_path.name}.postprocess.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not _remove_if_invalid(int4_path):
                int4_cmd = [
                    "python",
                    "/root/miles/tools/convert_hf_to_int4_direct.py",
                    "--model-dir",
                    str(source_hf_path),
                    "--save-dir",
                    str(int4_path),
                    "--group-size",
                    self.environment["OPEN_TRAINING_INT4_GROUP_SIZE"],
                ]
                print(
                    "\n=== Converting Kimi HF checkpoint to INT4: "
                    f"{' '.join(shlex.quote(arg) for arg in int4_cmd)} ==="
                )
                subprocess.run(int4_cmd, check=True)

            if not _remove_if_invalid(bf16_path):
                bf16_cmd = [
                    "python",
                    "/opt/training-gym/tools/convert_kimi_int4_to_bf16.py",
                    "--model-dir",
                    str(source_hf_path),
                    "--output-dir",
                    str(bf16_path),
                ]
                print(
                    "\n=== Converting Kimi native INT4 checkpoint to BF16: "
                    f"{' '.join(shlex.quote(arg) for arg in bf16_cmd)} ==="
                )
                subprocess.run(bf16_cmd, check=True)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K2_5_LoRA_Recipe(_KimiK2Recipe):
    source_hf_checkpoint: str = "moonshotai/Kimi-K2.5"
    hf_checkpoint: str = "/checkpoints/Kimi-K2.5-int4"
    ref_load: str = "/checkpoints/Kimi-K2.5-bf16"


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K2_6_LoRA_Recipe(_KimiK2Recipe):
    source_hf_checkpoint: str = "moonshotai/Kimi-K2.6"
    hf_checkpoint: str = "/checkpoints/Kimi-K2.6-int4"
    ref_load: str = "/checkpoints/Kimi-K2.6-bf16"
