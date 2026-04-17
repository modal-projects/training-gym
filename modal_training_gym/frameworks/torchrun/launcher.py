"""Build a Modal app that runs a user-provided training script under torchrun.

Usage (from a tutorial file):

    from modal_training_gym.frameworks.torchrun import (
        TorchrunConfig, TorchrunModalConfig, build_torchrun_app,
    )

    TRAIN_SCRIPT = \"\"\"
    import argparse
    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--data_dir"); p.add_argument("--output_dir")
        # …
    if __name__ == "__main__":
        main()
    \"\"\"

    class _Torchrun(TorchrunConfig):
        model = Model(BaseModelType.Llama2_7B)
        dataset = ...
        wandb = ...
        train_script_source = TRAIN_SCRIPT

    app = build_torchrun_app(modal=TorchrunModalConfig(gpu="H100"), config=_Torchrun())

Exposes `app.download_dataset`, `app.upload_script`, and `app.train`.
"""

from __future__ import annotations

import inspect
import os

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered, get_cluster_info

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS

from .config import (
    DATASET_MOUNT_PATH,
    MODEL_MOUNT_PATH,
    SCRIPTS_MOUNT_PATH,
    TorchrunConfig,
    TorchrunModalConfig,
)


def build_torchrun_app(
    *,
    modal: TorchrunModalConfig,
    config: TorchrunConfig,
    name: str | None = None,
) -> App:
    app_name = name or f"torchrun-{type(config).__name__.lstrip('_').lower()}"

    # Inline user-defined classes (`_Torchrun`, inline `DatasetConfig`
    # subclasses, etc.) so the remote container doesn't need to re-import
    # the tutorial module.
    caller_module = inspect.getmodule(inspect.stack()[1].frame)
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    # ── Image ────────────────────────────────────────────────────────────────
    train_image = (
        Image.debian_slim(python_version=modal.python_version)
        .apt_install("libibverbs-dev", "libibverbs1")
        .pip_install(*modal.pip_deps)
        .add_local_python_source("modal_training_gym", copy=True)
    )

    # ── Volumes ──────────────────────────────────────────────────────────────
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    model_volume = Volume.from_name(f"{app_name}-model", create_if_missing=True)
    scripts_volume = Volume.from_name(f"{app_name}-scripts", create_if_missing=True)

    data_dir = str(DATASET_MOUNT_PATH)
    model_dir = str(MODEL_MOUNT_PATH)
    scripts_dir = str(SCRIPTS_MOUNT_PATH)
    script_remote_path = f"{scripts_dir}/{config.train_script_name}"

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "framework": "torchrun",
        **config.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_spec = f"{modal.gpu}:{config.gpus_per_node}"

    # ── download_dataset ─────────────────────────────────────────────────────
    @app.function(
        image=train_image,
        volumes={data_dir: data_volume},
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=24 * 60 * 60,
        serialized=True,
        name="download_dataset",
    )
    def download_dataset():
        """Run `config.dataset.prepare()` to populate the data volume."""
        assert config.dataset is not None, "config.dataset must be set"
        data_volume.reload()
        config.dataset.prepare()
        data_volume.commit()

    # ── upload_script ────────────────────────────────────────────────────────
    @app.function(
        image=train_image,
        volumes={scripts_dir: scripts_volume},
        timeout=5 * 60,
        serialized=True,
        name="upload_script",
    )
    def upload_script():
        """Write `config.train_script_source` to the scripts volume."""
        if not config.train_script_source.strip():
            raise RuntimeError("config.train_script_source is empty")
        os.makedirs(scripts_dir, exist_ok=True)
        with open(script_remote_path, "w") as f:
            f.write(config.train_script_source)
        scripts_volume.commit()
        print(
            f"Wrote training script ({len(config.train_script_source)} chars) "
            f"to {script_remote_path}"
        )

    # ── train ────────────────────────────────────────────────────────────────
    @app.function(
        image=train_image,
        gpu=gpu_spec,
        volumes={
            data_dir: data_volume,
            model_dir: model_volume,
            scripts_dir: scripts_volume,
        },
        secrets=[
            Secret.from_name("huggingface-secret"),
            Secret.from_name("wandb-secret"),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train",
    )
    @clustered(size=config.n_nodes, rdma=True)
    def train():
        """Run `torchrun` on the uploaded training script on each node."""
        from torch.distributed.run import parse_args, run

        assert os.path.exists(script_remote_path), (
            f"Training script not found at {script_remote_path}. "
            f"Run `app.upload_script` first."
        )

        cluster_info = get_cluster_info()
        rank = cluster_info.rank
        master_ip = (
            cluster_info.container_ips[0]
            if cluster_info.container_ips
            else "localhost"
        )

        if config.wandb is not None and rank == 0:
            os.environ["WANDB_PROJECT"] = config.wandb.project
            if config.wandb.exp_name or config.wandb.group:
                os.environ["WANDB_RUN_NAME"] = (
                    config.wandb.exp_name or config.wandb.group
                )

        # Pass data/output/model_cache as standard args. The script is free to
        # ignore them; most distributed-training scripts accept them.
        extra = list(config.script_args)
        if "--data_dir" not in extra:
            extra.extend(["--data_dir", data_dir])
        if "--output_dir" not in extra:
            extra.extend(["--output_dir", f"{model_dir}/{app_name}"])

        torchrun_args = [
            f"--nnodes={config.n_nodes}",
            f"--nproc-per-node={config.gpus_per_node}",
            f"--node-rank={rank}",
            f"--master-addr={master_ip}",
            f"--master-port={config.master_port}",
            script_remote_path,
            *extra,
        ]
        print(f"Executing torchrun {' '.join(torchrun_args)}")
        run(parse_args(torchrun_args))

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
