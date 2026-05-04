"""Factory that builds a Modal app for a slime training run from config objects.

Usage (from a tutorial file):

    from modal_training_gym.common.train import TrainConfig
    from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

    config = TrainConfig(
        model=my_model,
        dataset=my_dataset,
        recipe=SlimeRecipe(...),
    )
    app = config.build_app()

Then: `uv run modal run <tutorial_file>.py::train`.
"""

import asyncio
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict

from modal import App, Image, Secret, Volume
from modal.experimental import clustered

import cloudpickle

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.framework import (
    TOOLS_LOCAL_PATH,
    TOOLS_REMOTE_PATH,
    resolve_caller_module,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.ray_cluster import ModalRayCluster
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.train_result import TrainResult

from modal_training_gym.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeRecipe,
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
    slime: SlimeRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    name: str | None = None,
) -> App:
    """Return a Modal App with `download`, `prepare_dataset`, `convert_checkpoint`, and `train` defined."""
    app_name = name or f"slime-{type(slime).__name__.lstrip('_').lower()}"

    caller_module = resolve_caller_module()
    if caller_module is not None:
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file:
            caller_script = os.path.abspath(mod_file)

    # ── Image ────────────────────────────────────────────────────────────────
    image = (
        Image.from_registry("slimerl/slime:nightly-dev-20260329a")
        .entrypoint([])
        .add_local_python_source("modal_training_gym", copy=True)
        .add_local_dir(TOOLS_LOCAL_PATH, remote_path=TOOLS_REMOTE_PATH, copy=True)
    )
    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        caller_remote_path = f"/root/{caller_module_name}.py"
        image = image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )
    _rm_fn_shipped = False
    if slime.custom_rm_function is not None:
        import inspect as _inspect

        fn = slime.custom_rm_function
        fn_mod = getattr(fn, "__module__", None) or ""
        if not fn_mod.startswith("modal_training_gym"):
            fn_file = os.path.abspath(_inspect.getfile(fn))
            if fn_file != caller_script:
                fn_module_name = os.path.splitext(os.path.basename(fn_file))[0]
                image = image.add_local_file(
                    fn_file,
                    remote_path=f"/root/{fn_module_name}.py",
                    copy=True,
                )
                _rm_fn_shipped = True

    for mod_name in slime.local_python_sources:
        image = image.add_local_python_source(mod_name, copy=True)
    if slime.local_slime:
        image = image.add_local_dir(
            slime.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if slime.image_run_commands is not None:
        image = slime.image_run_commands(image)

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        f"{app_name}-checkpoints", create_if_missing=True
    )
    all_volumes = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        str(CHECKPOINTS_PATH): checkpoints_volume,
    }

    # ── App ──────────────────────────────────────────────────────────────────
    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": "slime",
        **slime.app_tags,
    }
    if slime.wandb is not None:
        tags["_modal_wandb_project"] = slime.wandb.project
        if slime.wandb.group:
            tags["_modal_wandb_group"] = slime.wandb.group
    app = App(app_name, tags=tags)
    gpu_spec = f"{slime.gpu_type}:{slime.actor_num_gpus_per_node}"

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            str(CHECKPOINTS_PATH): checkpoints_volume,
        },
        timeout=2 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="download",
    )
    def download():
        hf_cache_volume.reload()
        checkpoints_volume.reload()
        model.download()
        hf_cache_volume.commit()
        checkpoints_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        data_volume.reload()
        dataset.prepare()
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
        from huggingface_hub import snapshot_download

        if getattr(slime, "megatron_to_hf_mode", None) == "bridge":
            print("Bridge mode — no conversion needed.")
            return

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        if model.model_path:
            hf_path = str(model.model_path)
        else:
            hf_path = snapshot_download(model.model_name, local_files_only=True)
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

        import importlib.util

        convert_script = (
            importlib.util.find_spec(
                "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
            ).origin
            if num_nodes > 1
            else f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"
        )

        cmd = (
            f"torchrun {' '.join(torchrun_args)} {convert_script} "
            f"{' '.join(extra_args)} "
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
        secrets=[
            *([] if slime.wandb is None else [Secret.from_name("wandb-secret")]),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train",
    )
    @clustered(slime.total_nodes, rdma=True)
    async def train():
        await asyncio.gather(
            hf_cache_volume.reload.aio(),
            data_volume.reload.aio(),
            checkpoints_volume.reload.aio(),
        )

        cluster = ModalRayCluster()
        cluster.discover_cluster(slime.total_nodes)

        os.environ["SLIME_HOST_IP"] = cluster.node_ip
        os.environ["SGLANG_HOST_IP"] = cluster.node_ip
        os.environ["HOST_IP"] = cluster.node_ip

        cluster.start_ray()

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        run_id = f"{app_name}-{int(time.time())}"
        config_summary: dict = {
            "model": {"model_name": model.model_name} if model else {},
            "recipe": {
                "gpu_type": slime.gpu_type,
                "actor_num_nodes": slime.actor_num_nodes,
                "actor_num_gpus_per_node": slime.actor_num_gpus_per_node,
            },
            "wandb": (
                {"project": slime.wandb.project, "group": slime.wandb.group}
                if slime.wandb
                else {}
            ),
            "dataset": {
                "prompt_data": getattr(dataset, "prompt_data", ""),
                "hf_repo": getattr(dataset, "hf_repo", ""),
                "name": type(dataset).__name__,
            },
            "lr": slime.lr,
            "global_batch_size": slime.global_batch_size,
        }
        TrainingRun(
            run_id=run_id,
            modal_app_id=os.environ.get("MODAL_APP_ID", ""),
            framework="slime",
            config=config_summary,
        ).save()
        print(f"TrainingRun recorded: {run_id}")

        if model:
            cache_dir = (
                HF_CACHE_PATH / "hub" / f"models--{model.model_name.replace('/', '--')}"
            )
            snapshots_dir = cache_dir / "snapshots"
            has_snapshot = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
            if not has_snapshot:
                print(f"Downloading model {model.model_name}...")
                model.download()
                hf_cache_volume.commit()

        if dataset and not os.path.exists(dataset.prompt_data):
            print(f"Preparing dataset ({dataset.prompt_data} not found)...")
            dataset.prepare()
            data_volume.commit()

        prepare_slime_config(slime, model, tempfile.mkdtemp())

        if wandb_key := os.environ.get("WANDB_API_KEY", ""):
            if slime.wandb is not None:
                slime.wandb.key = wandb_key
            elif getattr(slime, "use_wandb", False):
                slime.wandb_key = wandb_key

        cmd = build_train_cmd(slime, SLIME_ROOT, model=model, dataset=dataset)
        runtime_env = {
            "env_vars": {
                "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                "MASTER_ADDR": cluster.head_addr,
                **slime.environment,
            }
        }

        mode = "async" if slime.async_mode else "sync"
        print(
            f"Training {app_name} — {slime.total_nodes} node(s) × {gpu_spec}  ({mode})"
        )
        print(f"Command: {cmd}, runtime_env: {runtime_env}")

        async with cluster.forward_dashboard() as tunnel:
            print(f"Ray dashboard: {tunnel.url}")
            await cluster.submit_and_tail(cmd, runtime_env=runtime_env)

        save_root = str(slime.save).rstrip("/") if slime.save else str(CHECKPOINTS_PATH)

        wandb_run_id = ""
        if slime.wandb and slime.wandb.project:
            try:
                import wandb

                api = wandb.Api()
                runs = api.runs(
                    slime.wandb.project,
                    filters={"group": slime.wandb.group} if slime.wandb.group else {},
                    order="-created_at",
                    per_page=1,
                )
                if runs:
                    wandb_run_id = runs[0].id
            except Exception as e:
                print(f"Could not fetch W&B run ID: {e}")

        result = TrainResult(
            app_name=app_name,
            framework="slime",
            training_run_id=run_id,
            checkpoint_dir=save_root,
            model_config=model,
            checkpoints_volume_name=f"{app_name}-checkpoints",
            checkpoints_mount_path=str(CHECKPOINTS_PATH).rstrip("/"),
            iteration_prefix=slime.checkpoint.iteration_prefix,
            wandb_project=slime.wandb.project if slime.wandb else "",
            wandb_entity="",
            wandb_training_run_id=wandb_run_id,
        )
        result.save()
        checkpoints_volume.commit()
        print(f"TrainResult saved: {run_id}")
        return asdict(result)

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
