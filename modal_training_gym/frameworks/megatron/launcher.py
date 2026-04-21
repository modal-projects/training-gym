"""Factory that builds a Modal app for a NeMo-Megatron (`megatron.bridge`) run.

Usage (from a tutorial file):

    from modal_training_gym.common.models import GLM_4_7
    from modal_training_gym.frameworks.megatron import (
        MegatronConfig, MegatronFrameworkConfig, build_megatron_app,
    )

    megatron = MegatronConfig(
        model=GLM_4_7(),  # or subclass ModelConfiguration for a custom HF model
        dataset=...,  # DatasetConfig subclass with prepare()
        wandb=...,
        framework_config=MegatronFrameworkConfig(gpu="B200"),
    )

    app = build_megatron_app(megatron=megatron)

Produces a `modal.App` with:
  - `download_model` — pull HF model into the cache volume.
  - `convert_to_megatron` — HF → Megatron `torch_dist` checkpoint.
  - `download_and_convert` — orchestrates the two above.
  - `prepare_dataset` — run `megatron.dataset.prepare()`.
  - `train_lora` — launch distributed LoRA training via torchrun.
"""

from __future__ import annotations

import inspect
import os
import subprocess

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered, get_cluster_info

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.framework import resolve_caller_module

from .config import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    HF_CACHE,
    MODELS_DIR,
    MegatronConfig,
)

_CONFIG_PATH = "/tmp/megatron_config.json"


def build_megatron_app(
    *,
    megatron: MegatronConfig,
    name: str | None = None,
) -> App:
    app_name = name or f"megatron-{type(megatron).__name__.lstrip('_').lower()}"
    framework = megatron.framework_config

    # Inline user-defined classes (e.g. `_Megatron`) so the remote container
    # doesn't need to re-import the tutorial module.
    caller_module = resolve_caller_module()
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    # ── Images ───────────────────────────────────────────────────────────────
    download_image = (
        Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "huggingface_hub==0.36.0",
            "transformers==4.57.4",
            "torch==2.9.1",
            "safetensors==0.7.0",
            "sentencepiece==0.2.1",
            # Required by modal_training_gym config modules + cloudpickle
            # deserialization on the remote side.
            "cloudpickle",
            "msgspec",
            "pydantic",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_python_source("modal_training_gym", copy=True)
    )

    nemo_image = (
        Image.from_registry(framework.nemo_image)
        .entrypoint([])
        .apt_install(
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc-dev",
            "libnl-route-3-200",
        )
        .uv_pip_install(
            "wandb==0.23.1",
            "datasets==3.1.0",
            "transformers",
            "cloudpickle",
            "msgspec",
            "pydantic",
        )
        .run_commands(f"rm -Rf {HF_CACHE}")
        .add_local_python_source("modal_training_gym", copy=True)
    )

    # ── Volumes ──────────────────────────────────────────────────────────────
    models_volume = Volume.from_name("big-model-hfcache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True
    )

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "framework": "megatron",
        **framework.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_single = f"{framework.gpu}"
    gpu_multi = f"{framework.gpu}:{framework.gpus_per_node}"

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={str(HF_CACHE): models_volume},
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=4 * 60 * 60,
        serialized=True,
        name="download_model",
    )
    def download_model():
        """Download `megatron.model.model_name` into the shared HF cache."""
        from huggingface_hub import snapshot_download

        assert megatron.model is not None, "megatron.model must be set"
        models_volume.reload()
        path = snapshot_download(
            megatron.model.model_name,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Downloaded to: {path}")
        models_volume.commit()
        return {"path": path}

    # ── convert_to_megatron ──────────────────────────────────────────────────
    @app.function(
        image=nemo_image,
        gpu=gpu_single,
        volumes={
            str(HF_CACHE): models_volume,
            str(CHECKPOINTS_DIR): checkpoints_volume,
        },
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=6 * 60 * 60,
        cpu=16,
        memory=1_048_576,  # 1 TiB
        ephemeral_disk=2_048_000,  # 2 TiB scratch for the checkpoint write
        serialized=True,
        name="convert_to_megatron",
    )
    def convert_to_megatron():
        """Convert HF → Megatron `torch_dist` checkpoint format."""
        import torch
        from megatron.bridge import AutoBridge
        from megatron.bridge.training.model_load_save import save_megatron_model

        assert megatron.model is not None
        models_volume.reload()

        checkpoint_out = f"{CHECKPOINTS_DIR}/{app_name}-megatron"
        print(f"Converting {megatron.model.model_name} → {checkpoint_out}")

        bridge = AutoBridge.from_hf_pretrained(
            megatron.model.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        megatron_model = bridge.to_megatron_model(
            wrap_with_ddp=False, use_cpu_initialization=True
        )
        save_megatron_model(
            megatron_model,
            checkpoint_out,
            ckpt_format="torch_dist",
            hf_tokenizer_path=megatron.model.model_name,
        )

        checkpoints_volume.commit()
        return {"megatron_path": checkpoint_out}

    # ── download_and_convert ─────────────────────────────────────────────────
    @app.function(
        image=Image.debian_slim(),
        timeout=10 * 60 * 60,
        serialized=True,
        name="download_and_convert",
    )
    def download_and_convert():
        """Orchestrate `download_model` (CPU) then `convert_to_megatron` (GPU)."""
        print("Step 1/2: Downloading model (on CPU)...")
        download_result = download_model.remote()
        print(f"Download complete: {download_result}")

        print("Step 2/2: Converting to Megatron format (on GPU)...")
        convert_result = convert_to_megatron.remote()
        print(f"Convert complete: {convert_result}")

        return {"download": download_result, "convert": convert_result}

    # ── prepare_dataset ──────────────────────────────────────────────────────
    @app.function(
        image=nemo_image,
        volumes={str(DATA_DIR): data_volume},
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=60 * 60,
        cpu=32,
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        """Run `megatron.dataset.prepare()` to populate the data volume."""
        assert megatron.dataset is not None, "megatron.dataset must be set"
        data_volume.reload()
        megatron.dataset.prepare()
        data_volume.commit()

    # ── train_lora ───────────────────────────────────────────────────────────
    @app.function(
        image=nemo_image,
        gpu=gpu_multi,
        volumes={
            str(MODELS_DIR): models_volume,
            str(DATA_DIR): data_volume,
            str(CHECKPOINTS_DIR): checkpoints_volume,
        },
        secrets=[
            Secret.from_name("huggingface-secret"),
            Secret.from_name("wandb-secret"),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train_lora",
    )
    @clustered(size=framework.n_nodes, rdma=True)
    def train_lora():
        """Run the LoRA training script on each clustered node via torchrun."""
        assert megatron.model is not None
        assert megatron.dataset is not None

        # Environment optimizations (same as the reference).
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
        os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        cluster_info = get_cluster_info()
        node_rank = cluster_info.rank
        master_addr = (
            cluster_info.container_ips[0]
            if cluster_info.container_ips
            else "localhost"
        )
        print(f"Node {node_rank}/{framework.n_nodes}, Master: {master_addr}:29500")

        # Dataset: DatasetConfig.prompt_data can be a file path; Megatron's
        # FinetuningDatasetConfig wants the directory containing training.jsonl.
        dataset_path = getattr(megatron.dataset, "prompt_data", "")
        preprocessed_dir = (
            os.path.dirname(dataset_path)
            if dataset_path.endswith(".jsonl")
            else dataset_path
        )
        if not preprocessed_dir or not os.path.exists(preprocessed_dir):
            raise RuntimeError(
                f"Preprocessed data not found at {preprocessed_dir!r}. "
                f"Run `prepare_dataset` first."
            )

        # Checkpoint from convert_to_megatron.
        megatron_checkpoint = f"{CHECKPOINTS_DIR}/{app_name}-megatron"
        if not os.path.exists(megatron_checkpoint):
            raise RuntimeError(
                f"Megatron checkpoint not found at {megatron_checkpoint}. "
                f"Run `download_and_convert` (or `convert_to_megatron`) first."
            )

        # Serialize MegatronConfig to JSON for the train script to read.
        megatron.write_script_config(_CONFIG_PATH)

        # Resolve the train script path via the packaged module.
        import importlib.util

        spec = importlib.util.find_spec(
            "modal_training_gym.frameworks.megatron.train_script"
        )
        assert spec is not None and spec.origin is not None, (
            "Could not locate modal_training_gym.frameworks.megatron.train_script"
        )
        script_path = spec.origin

        cmd = [
            "torchrun",
            f"--nnodes={framework.n_nodes}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            "--master_port=29500",
            f"--nproc_per_node={framework.gpus_per_node}",
            script_path,
            "--preprocessed_dir",
            preprocessed_dir,
            "--megatron_checkpoint",
            megatron_checkpoint,
            "--checkpoints_dir",
            str(CHECKPOINTS_DIR),
            "--config",
            _CONFIG_PATH,
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with code {result.returncode}")

        checkpoints_volume.commit()
        return {"status": "complete"}

    # Expose registered functions as attributes (`app.download_model.remote()`).
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
