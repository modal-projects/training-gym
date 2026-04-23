"""Factory that builds a Modal app for a Miles RL training run.

Usage (from a tutorial file):

    from modal_training_gym.common.models import Qwen3_32B
    from modal_training_gym.frameworks.miles import (
        MilesConfig, MilesFrameworkConfig, build_miles_app,
    )

    miles = MilesConfig(
        model=Qwen3_32B(),  # or subclass ModelConfiguration for a custom HF model
        dataset=...,  # DatasetConfig subclass; prepare() populates /data
        wandb=...,
        framework_config=MilesFrameworkConfig(
            gpu="H100",
            recipe_args=\"\"\"
                --disable-bias-linear
                --qk-layernorm
                ...
            \"\"\",
        ),
    )

    app = build_miles_app(miles=miles)

Exposes `app.download_model`, `app.prepare_dataset`, and `app.train_multi_node`.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import time

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.framework import (
    TOOLS_LOCAL_PATH,
    TOOLS_REMOTE_PATH,
    resolve_caller_module,
)
from modal_training_gym.common.ray_cluster import ModalRayCluster

from .config import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MilesConfig,
    model_training_cli_args,
)

_MILES_ROOT = "/root/miles"
_REMOTE_TRAIN_SCRIPT = f"{_MILES_ROOT}/train.py"


def build_miles_app(
    *,
    miles: MilesConfig,
    gpu: str = "H100",
    name: str | None = None,
) -> App:
    """Return a Modal App with `download_model`, `prepare_dataset`, `train_multi_node`."""
    app_name = name or f"miles-{type(miles).__name__.lstrip('_').lower()}"
    framework = miles.framework_config

    caller_module = resolve_caller_module()
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file:
            caller_script = os.path.abspath(mod_file)

    # ── Image ────────────────────────────────────────────────────────────────
    image = Image.from_registry(framework.miles_image).entrypoint([])
    for cmd in framework.image_run_commands:
        image = image.run_commands(cmd)
    image = image.add_local_python_source(
        "modal_training_gym", copy=True
    ).add_local_dir(TOOLS_LOCAL_PATH, remote_path=TOOLS_REMOTE_PATH, copy=True)
    if caller_script is not None:
        caller_remote_path = (
            f"/root/{os.path.splitext(os.path.basename(caller_script))[0]}.py"
        )
        image = image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )

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
        "_modal_framework": "miles",
        **framework.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_spec = f"{gpu}:{framework.gpus_per_node}"

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=image,
        volumes={
            hf_cache_str: hf_cache_volume,
            checkpoints_str: checkpoints_volume,
        },
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=24 * 60 * 60,
        serialized=True,
        name="download_model",
    )
    def download_model():
        """Download model weights via the attached ModelConfiguration's hook."""
        assert miles.model is not None, "miles.model must be set"
        hf_cache_volume.reload()
        checkpoints_volume.reload()
        miles.model.download_model()
        hf_cache_volume.commit()
        checkpoints_volume.commit()

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
            *([] if miles.wandb is None else [Secret.from_name("wandb-secret")]),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train_multi_node",
    )
    @clustered(size=framework.n_nodes, rdma=True)
    async def train_multi_node(*extra_argv: str):
        """Multi-node Miles training: bring up Ray, submit Miles job via client.

        Any positional args get appended to the Miles argv — useful for
        one-off overrides like `--train-iters 5`.
        """
        assert miles.model is not None, "miles.model must be set"
        has_arch = (
            miles.model.architecture is not None
            and miles.model.architecture.num_layers > 0
        )
        if not has_arch and not framework.recipe_args.strip() and not extra_argv:
            raise RuntimeError(
                "No model architecture found and recipe_args is empty. Either "
                "use a model with a ModelArchitecture (e.g. Qwen3_4B) or set "
                "recipe_args to the Miles CLI args for your model."
            )

        checkpoints_volume.reload()
        data_volume.reload()

        cluster = ModalRayCluster()
        cluster.start(n_nodes=framework.n_nodes)

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        assert miles.model is not None
        if miles.model.model_path:
            model_path = str(miles.model.model_path)
        else:
            from huggingface_hub import snapshot_download

            try:
                model_path = snapshot_download(
                    miles.model.model_name, local_files_only=True
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Model {miles.model.model_name} not present in HF cache. "
                    f"Run `app.download_model` first."
                ) from exc

        # Build the Miles argv: recipe + extras + framework-enforced flags.
        run_id = f"{app_name}-{int(time.time())}"
        checkpoint_dir = f"{checkpoints_str}/{run_id}"

        model_arch_args: list[str] = []
        if miles.model and miles.model.architecture:
            model_arch_args = miles.model.architecture.to_megatron_args()

        argv: list[str] = [
            "python3",
            _REMOTE_TRAIN_SCRIPT,
            *framework.cli_args(),
            *model_arch_args,
            *(model_training_cli_args(miles.model) if miles.model else []),
            *framework.parsed_recipe_args(),
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
            str(framework.resolved_actor_nodes()),
            "--actor-num-gpus-per-node",
            str(framework.gpus_per_node),
            "--num-gpus-per-node",
            str(framework.gpus_per_node),
        ]

        if framework.colocate:
            argv.append("--colocate")
        else:
            rollout_gpus = framework.resolved_rollout_num_gpus()
            assert rollout_gpus is not None
            argv.extend(["--rollout-num-gpus", str(rollout_gpus)])

        if framework.custom_config_yaml.strip():
            cfg_path = f"/tmp/{run_id}-overrides.yaml"
            pathlib.Path(cfg_path).write_text(framework.custom_config_yaml)
            argv.extend(["--custom-config-path", cfg_path])

        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if miles.wandb is not None and wandb_key:
            argv.extend(["--use-wandb", "--wandb-key", wandb_key])

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
            status = await cluster.submit_and_tail(entrypoint, runtime_env=runtime_env)
            print(f"Final status: {status}")

        checkpoints_volume.commit()
        return {"run_id": run_id, "checkpoint_dir": checkpoint_dir}

    # Expose registered functions as attributes so notebook callers can do
    # `app.train_multi_node.remote()` instead of app.registered_functions[...].
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
