"""Factory that builds a Modal app for an ms-swift Megatron training run.

Usage (from a tutorial file):

    from modal_training_gym.common.models import BaseModelType, Model
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig, MsSwiftModalConfig, build_ms_swift_app,
    )

    class _MsSwift(MsSwiftConfig):
        model = Model(BaseModelType.GLM_4_7)
        dataset = ...
        wandb = ...
        # ... overrides

    app = build_ms_swift_app(modal=MsSwiftModalConfig(gpu="B200"), swift=_MsSwift())

Then `modal run <file>.py::app.train` (or `::app.download_model`, etc).
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered, get_cluster_info

from .config import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    MsSwiftConfig,
    MsSwiftModalConfig,
)


def build_ms_swift_app(
    *,
    modal: MsSwiftModalConfig,
    swift: MsSwiftConfig,
    name: str | None = None,
) -> App:
    """Return a Modal App with `download_model`, `prepare_dataset`, `train`."""
    app_name = name or f"ms-swift-{type(swift).__name__.lstrip('_').lower()}"

    # Inline user-defined classes (e.g. `_MsSwift`) so the remote container
    # doesn't need to re-import the tutorial module (same trick as slime).
    caller_module = inspect.getmodule(inspect.stack()[1].frame)
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    # ── Images ───────────────────────────────────────────────────────────────
    download_image = (
        Image.debian_slim(python_version="3.11")
        .pip_install(
            "huggingface_hub==0.27.1",
            "transformers>=4.50",
            "torch==2.5.1",
            "safetensors==0.4.5",
            "datasets>=2.14.0",
        )
        .add_local_python_source("modal_training_gym", copy=True)
    )

    train_image = (
        Image.from_registry(modal.image)
        .apt_install(
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc-dev",
            "libnl-route-3-200",
        )
        # Force-remove pre-installed transformers + ms-swift so our pinned
        # versions (which support GLM-4 MoE) take effect.
        .run_commands(
            "pip uninstall -y transformers ms-swift swift 2>/dev/null; true",
        )
        .pip_install(
            f"transformers=={modal.transformers_version}",
            "ms-swift @ git+https://github.com/modelscope/ms-swift.git@main",
            "einops==0.8.2",
            "wandb==0.19.1",
        )
        .env(swift.environment)
        .add_local_python_source("modal_training_gym", copy=True)
    )

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True, version=2
    )
    hf_cache_str = str(HF_CACHE_PATH)
    checkpoints_str = str(CHECKPOINTS_PATH)

    app = App(app_name)
    gpu_spec = f"{modal.gpu}:{swift.gpus_per_node}"

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={hf_cache_str: hf_cache_volume},
        timeout=4 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="download_model",
    )
    def download_model(force: bool = False):
        """Download `swift.model.hf_checkpoint` into the shared HF cache."""
        from huggingface_hub import snapshot_download

        assert swift.model is not None, "swift.model must be set"
        hf_cache_volume.reload()
        path = snapshot_download(
            swift.model.hf_checkpoint,
            token=os.environ.get("HF_TOKEN"),
            force_download=force,
        )
        print(f"Downloaded to: {path}")
        hf_cache_volume.commit()
        return {"path": path}

    # ── prepare_dataset ──────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={hf_cache_str: hf_cache_volume},
        timeout=60 * 60,
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        """Run `swift.dataset.prepare()` to populate the data volume."""
        assert swift.dataset is not None, "swift.dataset must be set"
        hf_cache_volume.reload()
        swift.dataset.prepare()
        hf_cache_volume.commit()

    # ── train ────────────────────────────────────────────────────────────────
    @app.function(
        image=train_image,
        gpu=gpu_spec,
        volumes={
            hf_cache_str: hf_cache_volume,
            checkpoints_str: checkpoints_volume,
        },
        secrets=[
            Secret.from_name("huggingface-secret"),
            Secret.from_name("wandb-secret"),
        ],
        timeout=24 * 60 * 60,
        retries=2,
        memory=1048576,  # 1 TiB
        ephemeral_disk=2_048_000,  # 2 TiB scratch
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train",
    )
    @clustered(size=swift.n_nodes, rdma=True)
    def train(run_id: str | None = None):
        """Run `megatron sft ...` under `torchrun` on each clustered node."""
        if run_id is None:
            run_id = f"train_{int(time.time())}"

        assert swift.model is not None, "swift.model must be set"
        assert swift.dataset is not None, "swift.dataset must be set"

        cluster_info = get_cluster_info()
        node_rank = cluster_info.rank
        n_nodes = (
            len(cluster_info.container_ips) if cluster_info.container_ips else 1
        )
        master_addr = (
            cluster_info.container_ips[0]
            if cluster_info.container_ips
            else "localhost"
        )
        print(f"Node {node_rank}/{n_nodes}, Master: {master_addr}")

        # Resolve the model dir inside the HF cache volume.
        from huggingface_hub import snapshot_download

        try:
            model_dir = snapshot_download(
                swift.model.hf_checkpoint, local_files_only=True
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Model {swift.model.hf_checkpoint} not found in HF cache. "
                f"Run `download_model` first."
            )

        dataset_path = getattr(swift.dataset, "prompt_data", "")
        if not dataset_path or not os.path.exists(dataset_path):
            raise RuntimeError(
                f"No training data at {dataset_path!r}. "
                f"Run `prepare_dataset` first."
            )

        checkpoint_dir = f"{checkpoints_str}/{app_name}_{run_id}"

        # If a prior attempt crashed mid-train, resume from the latest iter_ dir.
        resuming = False
        if os.path.exists(checkpoint_dir):
            iter_dirs = sorted(
                d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")
            )
            if iter_dirs:
                resuming = True
                print(f"Resuming from {iter_dirs[-1]}")

        # Pre-create args.json on every rank. ms-swift writes it from the
        # master rank but reads it from the last rank (different container on
        # Modal), so both need to see a file.
        os.makedirs(checkpoint_dir, exist_ok=True)
        args_json = os.path.join(checkpoint_dir, "args.json")
        if not os.path.exists(args_json):
            with open(args_json, "w") as f:
                json.dump({"run_id": run_id, "placeholder": True}, f)

        # Default the wandb run name to the run_id if the user didn't set one.
        if swift.wandb is not None and not swift.wandb.exp_name:
            swift.wandb.exp_name = run_id

        # The hf_checkpoint on the config might be a bare repo id; the
        # megatron CLI needs the local cache path. Swap in model_dir.
        megatron_flags = swift.cli_args(output_dir=checkpoint_dir)
        for i, a in enumerate(megatron_flags):
            if a == "--model" and i + 1 < len(megatron_flags):
                megatron_flags[i + 1] = model_dir
                break

        if resuming:
            megatron_flags.extend(["--load", checkpoint_dir])

        cmd = [
            "torchrun",
            "--no_python",
            "--nproc_per_node",
            str(swift.gpus_per_node),
            "--nnodes",
            str(n_nodes),
            "--node_rank",
            str(node_rank),
            "--master_addr",
            master_addr,
            "--master_port",
            "29500",
            "megatron",
            "sft",
            *megatron_flags,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ms-swift failed with code {result.returncode}")

        checkpoints_volume.commit()
        print(f"Training {run_id} completed; results in {checkpoint_dir}")
        return {"run_id": run_id, "checkpoint_dir": checkpoint_dir}

    # Expose registered functions as attributes so notebook callers can do
    # `app.download_model.remote()` instead of app.registered_functions[...].
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
