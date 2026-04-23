"""Factory that builds a Modal app for an ms-swift Megatron training run.

Usage (from a tutorial file):

    from modal_training_gym.common.models import GLM_4_7
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig, MsSwiftFrameworkConfig, build_ms_swift_app,
    )

    swift = MsSwiftConfig(
        model=GLM_4_7(),  # or subclass ModelConfiguration for a custom HF model
        dataset=...,
        wandb=...,
        framework_config=MsSwiftFrameworkConfig(gpu="H100"),
    )

    app = build_ms_swift_app(swift=swift)

Then `modal run <file>.py::app.train` (or `::app.download_model`, etc).
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import cloudpickle
from modal import App, Image, Secret, Volume
from modal.experimental import clustered, get_cluster_info

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.framework import (
    TOOLS_LOCAL_PATH,
    TOOLS_REMOTE_PATH,
    resolve_caller_module,
)

from .config import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    MsSwiftConfig,
)

DATA_PATH = "/data"


def _train_ms_swift_worker(run_id: str | None = None):
    from huggingface_hub import snapshot_download

    model_name = os.environ["MS_SWIFT_MODEL_NAME"]
    dataset_path = os.environ["MS_SWIFT_DATASET_PATH"]
    app_name = os.environ["MS_SWIFT_APP_NAME"]
    checkpoints_root = os.environ["MS_SWIFT_CHECKPOINTS_PATH"]
    checkpoints_volume_name = os.environ["MS_SWIFT_CHECKPOINTS_VOLUME_NAME"]
    gpus_per_node = int(os.environ["MS_SWIFT_GPUS_PER_NODE"])
    wandb_project = os.environ.get("MS_SWIFT_WANDB_PROJECT", "")
    wandb_exp_name = os.environ.get("MS_SWIFT_WANDB_EXP_NAME", "")

    cluster_info = get_cluster_info()
    node_rank = cluster_info.rank

    if run_id is None:
        run_id_file = f"{checkpoints_root}/.current_run_id"
        if node_rank == 0:
            run_id = f"train_{int(time.time())}"
            os.makedirs(checkpoints_root, exist_ok=True)
            with open(run_id_file, "w") as f:
                f.write(run_id)
            Volume.from_name(
                checkpoints_volume_name, create_if_missing=True, version=2
            ).commit()
        else:
            for _ in range(60):
                Volume.from_name(
                    checkpoints_volume_name, create_if_missing=True, version=2
                ).reload()
                if os.path.exists(run_id_file):
                    run_id = open(run_id_file).read().strip()
                    break
                time.sleep(1)
            else:
                run_id = f"train_{int(time.time())}"
    ips = list(cluster_info.container_ipv4_ips or [])
    n_nodes = len(ips) if ips else 1
    master_addr = ips[0] if ips else "localhost"
    print(f"Node {node_rank}/{n_nodes}, Master: {master_addr}")

    model_path_override = os.environ.get("MS_SWIFT_MODEL_PATH", "")
    if model_path_override:
        model_dir = model_path_override
    else:
        try:
            model_dir = snapshot_download(model_name, local_files_only=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Model {model_name} not found in HF cache. Run `download_model` first."
            ) from exc

    if not dataset_path or not os.path.exists(dataset_path):
        raise RuntimeError(
            f"No training data at {dataset_path!r}. Run `prepare_dataset` first."
        )

    checkpoint_dir = f"{checkpoints_root}/{app_name}_{run_id}"
    resuming = False
    if os.path.exists(checkpoint_dir):
        iter_dirs = sorted(
            d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")
        )
        if iter_dirs:
            resuming = True
            print(f"Resuming from {iter_dirs[-1]}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    args_json = os.path.join(checkpoint_dir, "args.json")
    if not os.path.exists(args_json):
        with open(args_json, "w") as f:
            json.dump({"run_id": run_id, "placeholder": True}, f)

    megatron_flags = json.loads(os.environ["MS_SWIFT_BASE_CLI_ARGS_JSON"])
    for i, arg in enumerate(megatron_flags):
        if arg == "--output_dir" and i + 1 < len(megatron_flags):
            megatron_flags[i + 1] = checkpoint_dir
            break
    for i, arg in enumerate(megatron_flags):
        if arg == "--model" and i + 1 < len(megatron_flags):
            megatron_flags[i + 1] = model_dir
            break

    if (
        wandb_project
        and not wandb_exp_name
        and "--wandb_exp_name" not in megatron_flags
    ):
        megatron_flags.extend(["--wandb_exp_name", run_id])

    if resuming:
        megatron_flags.extend(["--load", checkpoint_dir])

    cmd = [
        "torchrun",
        "--no_python",
        "--nproc_per_node",
        str(gpus_per_node),
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

    Volume.from_name(
        checkpoints_volume_name, create_if_missing=True, version=2
    ).commit()
    print(f"Training {run_id} completed; results in {checkpoint_dir}")
    return {"run_id": run_id, "checkpoint_dir": checkpoint_dir}


def build_ms_swift_app(
    swift: MsSwiftConfig,
    name: str | None = None,
) -> App:
    """Return a Modal App with `download_model`, `prepare_dataset`, `train`."""
    app_name = name or f"ms-swift-{type(swift).__name__.lstrip('_').lower()}"
    framework = swift.framework_config

    caller_module = resolve_caller_module()
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file:
            caller_script = os.path.abspath(mod_file)

    # ── Images ───────────────────────────────────────────────────────────────
    download_image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(
            "huggingface_hub==0.27.1",
            "transformers>=4.50",
            "torch==2.5.1",
            "safetensors==0.4.5",
            "datasets>=2.14.0",
            "msgspec",
            "pydantic",
        )
        .add_local_python_source("modal_training_gym", copy=True)
        .add_local_dir(TOOLS_LOCAL_PATH, remote_path=TOOLS_REMOTE_PATH, copy=True)
    )

    train_image = (
        Image.from_registry(framework.image)
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
            f"transformers=={framework.transformers_version}",
            "ms-swift @ git+https://github.com/modelscope/ms-swift.git@joy/initial-setup",
            "einops==0.8.2",
            "wandb==0.19.1",
            "msgspec",
            "pydantic",
            # ms-swift's megatron backend (`megatron sft`) imports megatron.core.
            # The ModelScope base image ships this; other py312 bases (NGC
            # PyTorch, etc.) need it added here.
            "megatron-core",
            # transformers' ProcessorMixin (imported by ms-swift's utils) has
            # lazy deps on pillow/tokenizers that the ModelScope image ships
            # but NGC PyTorch does not.
            "pillow",
            "tokenizers",
        )
        .env(framework.environment)
        .add_local_python_source("modal_training_gym", copy=True)
        .add_local_dir(TOOLS_LOCAL_PATH, remote_path=TOOLS_REMOTE_PATH, copy=True)
    )
    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        caller_remote_path = f"/root/{caller_module_name}.py"
        download_image = download_image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )
        train_image = train_image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True, version=2
    )
    hf_cache_str = str(HF_CACHE_PATH)
    data_str = DATA_PATH
    checkpoints_str = str(CHECKPOINTS_PATH)

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": "ms-swift",
        **framework.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_spec = f"{framework.gpu}:{framework.gpus_per_node}"

    # ── download_model ───────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={
            hf_cache_str: hf_cache_volume,
            checkpoints_str: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="download_model",
    )
    def download_model():
        """Download model weights via the attached ModelConfiguration's hook."""
        assert swift.model is not None, "swift.model must be set"
        hf_cache_volume.reload()
        checkpoints_volume.reload()
        swift.model.download_model()
        hf_cache_volume.commit()
        checkpoints_volume.commit()

    # ── prepare_dataset ──────────────────────────────────────────────────────
    @app.function(
        image=download_image,
        volumes={hf_cache_str: hf_cache_volume, data_str: data_volume},
        timeout=60 * 60,
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        """Run `swift.dataset.prepare()` to populate the data volume."""
        assert swift.dataset is not None, "swift.dataset must be set"
        hf_cache_volume.reload()
        data_volume.reload()
        swift.dataset.prepare()
        hf_cache_volume.commit()
        data_volume.commit()

    # ── train ────────────────────────────────────────────────────────────────
    assert swift.model is not None, "swift.model must be set"
    assert swift.dataset is not None, "swift.dataset must be set"
    base_cli_args = swift.cli_args(output_dir="__OUTPUT_DIR__")
    train_env = {
        "MS_SWIFT_MODEL_NAME": swift.model.model_name,
        "MS_SWIFT_MODEL_PATH": swift.model.model_path or "",
        "MS_SWIFT_DATASET_PATH": getattr(swift.dataset, "prompt_data", ""),
        "MS_SWIFT_APP_NAME": app_name,
        "MS_SWIFT_GPUS_PER_NODE": str(framework.gpus_per_node),
        "MS_SWIFT_CHECKPOINTS_PATH": checkpoints_str,
        "MS_SWIFT_CHECKPOINTS_VOLUME_NAME": f"{app_name}-checkpoints",
        "MS_SWIFT_BASE_CLI_ARGS_JSON": json.dumps(base_cli_args),
        "MS_SWIFT_WANDB_PROJECT": swift.wandb.project if swift.wandb else "",
        "MS_SWIFT_WANDB_EXP_NAME": (
            (swift.wandb.exp_name or swift.wandb.group) if swift.wandb else ""
        ),
    }
    app.function(
        image=train_image,
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
        retries=2,
        memory=1048576,  # 1 TiB
        ephemeral_disk=2_048_000,  # 2 TiB scratch
        experimental_options={"efa_enabled": True},
        env=train_env,
        serialized=False,
        name="train",
    )(clustered(size=framework.n_nodes, rdma=True)(_train_ms_swift_worker))

    # Expose registered functions as attributes so notebook callers can do
    # `app.download_model.remote()` instead of app.registered_functions[...].
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
