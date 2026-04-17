"""Factory that builds a Modal app for a Miles RL training run.

Usage (from a tutorial file):

    from modal_training_gym.common.models import BaseModelType, Model
    from modal_training_gym.frameworks.miles import (
        MilesConfig, MilesModalConfig, build_miles_app,
    )

    class _Miles(MilesConfig):
        model = Model(BaseModelType.Qwen3_30B_A3B)  # or your own registration
        dataset = ...  # DatasetConfig subclass; prepare() populates /data
        wandb = ...
        recipe_args = \"\"\"
            --disable-bias-linear
            --qk-layernorm
            ...
        \"\"\"

    app = build_miles_app(modal=MilesModalConfig(gpu="H100"), miles=_Miles())

Exposes `app.download_model`, `app.prepare_dataset`, and `app.train_multi_node`.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import shlex

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.ray_cluster import ModalRayCluster

from .config import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MilesConfig,
    MilesModalConfig,
)

_REMOTE_TRAIN_SCRIPT = "/root/miles/train.py"


def build_miles_app(
    *,
    modal: MilesModalConfig,
    miles: MilesConfig,
    name: str | None = None,
) -> App:
    """Return a Modal App with `download_model`, `prepare_dataset`, `train_multi_node`."""
    app_name = name or f"miles-{type(miles).__name__.lstrip('_').lower()}"

    caller_module = inspect.getmodule(inspect.stack()[1].frame)
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    # ── Image ────────────────────────────────────────────────────────────────
    image = Image.from_registry(modal.miles_image).entrypoint([])
    for cmd in modal.image_run_commands:
        image = image.run_commands(cmd)
    image = image.add_local_python_source("modal_training_gym", copy=True)

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True
    )

    hf_cache_str = str(HF_CACHE_PATH)
    data_str = str(DATA_PATH)
    checkpoints_str = str(CHECKPOINTS_PATH)

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "framework": "miles",
        **miles.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_spec = f"{modal.gpu}:{miles.gpus_per_node}"

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=image,
        volumes={hf_cache_str: hf_cache_volume},
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=24 * 60 * 60,
        serialized=True,
        name="download_model",
    )
    def download_model(revision: str | None = None):
        """Snapshot_download `miles.model.hf_checkpoint` into the shared HF cache."""
        from huggingface_hub import snapshot_download

        assert miles.model is not None, "miles.model must be set"
        assert miles.model.hf_checkpoint is not None
        hf_cache_volume.reload()
        path = snapshot_download(
            repo_id=miles.model.hf_checkpoint,
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Downloaded {miles.model.hf_checkpoint} to {path}")
        hf_cache_volume.commit()

    # ── prepare_dataset ──────────────────────────────────────────────────────
    @app.function(
        image=image,
        volumes={data_str: data_volume},
        timeout=4 * 60 * 60,
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        """Run `miles.dataset.prepare()` to populate the data volume."""
        assert miles.dataset is not None, "miles.dataset must be set"
        data_volume.reload()
        miles.dataset.prepare()
        data_volume.commit()

    # ── train_multi_node ─────────────────────────────────────────────────────
    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes={
            hf_cache_str: hf_cache_volume,
            data_str: data_volume,
            checkpoints_str: checkpoints_volume,
        },
        secrets=[
            Secret.from_name("huggingface-secret"),
            Secret.from_name("wandb-secret"),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train_multi_node",
    )
    @clustered(size=miles.n_nodes, rdma=True)
    async def train_multi_node(*extra_argv: str):
        """Multi-node Miles training: bring up Ray, submit Miles job via client.

        Any positional args get appended to the Miles argv — useful for
        one-off overrides like `--train-iters 5`.
        """
        assert miles.model is not None, "miles.model must be set"
        if not miles.recipe_args.strip() and not extra_argv:
            raise RuntimeError(
                "MilesConfig.recipe_args is empty — set it to the Miles CLI "
                "args block describing your model architecture + training hyperparameters."
            )

        checkpoints_volume.reload()
        data_volume.reload()

        cluster = ModalRayCluster()
        cluster.start(n_nodes=miles.n_nodes)

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        # Resolve the model path inside the HF cache volume.
        from huggingface_hub import snapshot_download

        assert miles.model.hf_checkpoint is not None
        try:
            model_path = snapshot_download(
                miles.model.hf_checkpoint, local_files_only=True
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Model {miles.model.hf_checkpoint} not present in HF cache. "
                f"Run `app.download_model` first."
            ) from exc

        # Build the Miles argv: recipe + extras + framework-enforced flags.
        run_id = f"{app_name}-{int(__import__('time').time())}"
        checkpoint_dir = f"{checkpoints_str}/{run_id}"

        argv: list[str] = [
            "python3",
            _REMOTE_TRAIN_SCRIPT,
            *miles.parsed_recipe_args(),
            *extra_argv,
            "--train-backend",
            "megatron",
            "--hf-checkpoint",
            model_path,
            "--ref-load",
            model_path,
            "--save",
            checkpoint_dir,
            "--actor-num-nodes",
            str(miles.resolved_actor_nodes()),
            "--actor-num-gpus-per-node",
            str(miles.gpus_per_node),
            "--num-gpus-per-node",
            str(miles.gpus_per_node),
        ]

        if miles.colocate:
            argv.append("--colocate")
        else:
            rollout_gpus = miles.resolved_rollout_num_gpus()
            assert rollout_gpus is not None
            argv.extend(["--rollout-num-gpus", str(rollout_gpus)])

        if miles.custom_config_yaml.strip():
            cfg_path = f"/tmp/{run_id}-overrides.yaml"
            pathlib.Path(cfg_path).write_text(miles.custom_config_yaml)
            argv.extend(["--custom-config-path", cfg_path])

        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if miles.wandb is not None and wandb_key:
            argv.extend(["--use-wandb", "--wandb-key", wandb_key])

        # Runtime env for the Ray job — Miles expects MASTER_ADDR + its patched
        # Megatron on PYTHONPATH.
        runtime_env = {
            "env_vars": {
                "MASTER_ADDR": cluster.head_addr,
                "MILES_HOST_IP": cluster.head_addr,
                "no_proxy": cluster.head_addr,
                "PYTHONPATH": "/root/miles-modal-patches:/root/Megatron-LM",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_ALGO": "Ring",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        }
        if wandb_key:
            runtime_env["env_vars"]["WANDB_API_KEY"] = wandb_key

        entrypoint = shlex.join(argv)
        async with cluster.forward_dashboard() as tunnel:
            print(f"Ray dashboard: {tunnel.url}")
            print(f"Miles entrypoint: {entrypoint}")
            status = await cluster.submit_and_tail(
                entrypoint, runtime_env=runtime_env
            )
            print(f"Final status: {status}")

        checkpoints_volume.commit()
        return {"run_id": run_id, "checkpoint_dir": checkpoint_dir}

    # Expose registered functions as attributes so notebook callers can do
    # `app.train_multi_node.remote()` instead of app.registered_functions[...].
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
