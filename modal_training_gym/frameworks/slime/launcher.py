"""Factory that builds a Modal app for a slime training run from config objects.

Usage (from a tutorial file):

    from modal_training_gym.frameworks.slime import ModalConfig, SlimeConfig, build_slime_app

    class _Slime(SlimeConfig):
        hf_checkpoint = "Qwen/Qwen3-8B"
        actor_num_nodes = 1
        actor_num_gpus_per_node = 8
        # ...

    app = build_slime_app(modal=ModalConfig(gpu="H200"), slime=_Slime())

Then: `uv run modal run <tutorial_file>.py::train`.
"""

import asyncio
import inspect
import os
import shlex
import subprocess
import tempfile

from modal import App, Image, Secret, Volume
from modal.experimental import clustered

import cloudpickle

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.framework import TOOLS_LOCAL_PATH, TOOLS_REMOTE_PATH, mount_tools_dir, resolve_caller_module
from modal_training_gym.common.ray_cluster import ModalRayCluster

from .config import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    ModalConfig,
    SlimeConfig,
)
from .modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    get_modal_cluster_context,
    prepare_slime_config,
)

SLIME_ROOT = "/root/slime"


def build_slime_app(
    *,
    modal: ModalConfig,
    slime: SlimeConfig,
    name: str | None = None,
) -> App:
    """Return a Modal App with `download_model`, `prepare_dataset`, `convert_checkpoint`, and `train` defined.

    The returned functions close over the passed-in configs, so a tutorial
    file holds its own Modal app whose shape is determined entirely by the
    config objects it constructs.
    """
    # Preflight: SLIME requires a `ModelArchitecture` on the attached model.
    # Fail locally with an actionable `ValueError` before any `@app.function`
    # decorator fires.
    if slime.model is not None:
        SlimeConfig._validate_custom_model_architecture(slime.model)

    app_name = name or f"slime-{type(slime).__name__.lstrip('_').lower()}"

    # Find the user's tutorial module (walks past framework frames that would
    # otherwise shadow the real caller when invoked via `SlimeConfig.build_app`).
    caller_module = resolve_caller_module()
    if caller_module is not None:
        # Inline the tutorial classes by value so the remote container doesn't
        # need to re-import the tutorial module at deserialization.
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file:
            caller_script = os.path.abspath(mod_file)

    # ── Image ────────────────────────────────────────────────────────────────
    image = (
        Image.from_registry(
            "slimerl/slime:nightly-dev-20260329a"
        )  # Check https://hub.docker.com/r/slimerl/slime/tags for the latest version.
        .entrypoint([])
        .add_local_python_source("modal_training_gym", copy=True).add_local_dir(TOOLS_LOCAL_PATH, remote_path=TOOLS_REMOTE_PATH, copy=True)
    )
    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        caller_remote_path = f"/root/{caller_module_name}.py"
        image = image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )
    for patch in modal.patch_files:
        image = image.add_local_file(
            patch, f"/tmp/{os.path.basename(patch)}", copy=True
        )
    if modal.local_slime:
        image = image.add_local_dir(
            modal.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if modal.image_run_commands:
        image = image.run_commands(*modal.image_run_commands)

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name("slime-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name("slime-checkpoints", create_if_missing=True)
    all_volumes = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        str(CHECKPOINTS_PATH): checkpoints_volume,
    }

    # ── App ──────────────────────────────────────────────────────────────────
    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "framework": "slime",
        **slime.app_tags,
    }
    app = App(app_name, tags=tags)
    gpu_spec = f"{modal.gpu}:{slime.actor_num_gpus_per_node}"

    @app.function(
        image=image,
        volumes={str(HF_CACHE_PATH): hf_cache_volume},
        timeout=2 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="download_model",
    )
    def download_model():
        """Download the model's HF checkpoint into the shared HF cache volume."""
        from huggingface_hub import snapshot_download

        assert slime.model is not None, "slime.model must be set"
        snapshot_download(repo_id=slime.model.model_name)
        hf_cache_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        """Populate the data volume by running `slime.prepare_data()`."""
        data_volume.reload()
        slime.prepare_data()
        data_volume.commit()

    convert_nnodes = get_checkpoint_conversion_policy(slime)[0]

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered(convert_nnodes, rdma=True)
    def convert_checkpoint():
        """HF → torch_dist conversion when `megatron_to_hf_mode == 'raw'`."""
        from huggingface_hub import snapshot_download

        if getattr(slime, "megatron_to_hf_mode", None) == "bridge":
            print("Bridge mode — no conversion needed.")
            return

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        assert slime.model is not None, "slime.model must be set"
        hf_path = snapshot_download(slime.model.model_name, local_files_only=True)
        save_path = str(slime.ref_load)
        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(slime)
        node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        # For multi-node, use our wrapper that honours SKIP_RELEASE_RENAME to
        # prevent volume corruption (see modal_helpers/convert_hf_to_torch_dist.py).
        # Single-node uses the upstream script directly.
        import importlib.util

        convert_script = (
            importlib.util.find_spec(
                "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
            ).origin
            if num_nodes > 1
            else f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"
        )

        cmd = (
            f"source {SLIME_ROOT}/{slime.slime_model_script} && "
            f"torchrun {' '.join(torchrun_args)} {convert_script} "
            f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
            f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
        )

        env = {**os.environ, **slime.environment}
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"

        print(
            f"Conversion layout: nodes={num_nodes}, "
            f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        subprocess.run(["bash", "-c", cmd], check=True, env=env)
        checkpoints_volume.commit()

        if node_rank == 0:
            print(f"Saved torch_dist checkpoint to {save_path}")

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        secrets=[Secret.from_name("wandb-secret")],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train",
    )
    @clustered(slime.total_nodes(), rdma=True)
    async def train():
        """Launch the slime training run on a Ray cluster."""
        await asyncio.gather(
            hf_cache_volume.reload.aio(),
            data_volume.reload.aio(),
            checkpoints_volume.reload.aio(),
        )

        cluster = ModalRayCluster()
        cluster.discover_cluster(slime.total_nodes())

        # slime, sglang, and related processes expect these env vars on every
        # rank before the Ray daemon starts so it inherits them.
        os.environ["SLIME_HOST_IP"] = cluster.node_ip
        os.environ["SGLANG_HOST_IP"] = cluster.node_ip
        os.environ["HOST_IP"] = cluster.node_ip

        cluster.start_ray()

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        prepare_slime_config(slime, tempfile.mkdtemp())

        if wandb_key := os.environ.get("WANDB_API_KEY", ""):
            if slime.wandb is not None:
                slime.wandb.key = wandb_key
            elif getattr(slime, "use_wandb", False):
                slime.wandb_key = wandb_key

        cmd = build_train_cmd(slime, SLIME_ROOT)
        runtime_env = {
            "env_vars": {
                "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                "MASTER_ADDR": cluster.head_addr,
                **slime.environment,
            }
        }

        gpu = f"{modal.gpu}:{slime.actor_num_gpus_per_node}"
        mode = "async" if slime.async_mode else "sync"
        print(
            f"Training {app_name} — "
            f"{slime.total_nodes()} node(s) × {gpu}  ({mode})"
        )
        print(f"Command: {cmd}, runtime_env: {runtime_env}")

        async with cluster.forward_dashboard() as tunnel:
            print(f"Ray dashboard: {tunnel.url}")
            await cluster.submit_and_tail(cmd, runtime_env=runtime_env)

    # Expose registered functions as attributes so notebook callers can do
    # `app.download_model.remote()` instead of
    # `app.registered_functions["download_model"].remote()`.
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
