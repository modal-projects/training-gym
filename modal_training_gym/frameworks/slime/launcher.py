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
import base64
import inspect
import os
import shlex
import subprocess
import tempfile
import textwrap
import time
from pathlib import PurePosixPath
from typing import Any
from collections.abc import Callable
from modal import App, Image, Secret, Volume

from modal_training_gym.common import hf_secrets
from modal.experimental import clustered

import cloudpickle

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.dataset import DatasetConfig, HarborDataset
from modal_training_gym.common.framework import (
    mount_tools_dir,
    resolve_caller_module,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.ray_cluster import ModalRayCluster
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
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
from modal_training_gym.common.checkpoint import (
    Checkpoint,
    _get_slime_checkpoint_prefix,
)
from modal_training_gym.common.framework import Framework

SLIME_ROOT = "/root/slime"
# Pin by digest to prevent mutable-tag drift.  Tag: nightly-dev-20260329a
SLIME_IMAGE = "slimerl/slime@sha256:d8ccfba3cd21134b277da4e0f37f57d6290d7b68432e6de047c7d25406c35190"
HARBOR_PKG_VERSION = "0.6.6"


def _build_slime_base_image() -> "Image":
    return (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface")
    )


def _has_torch_dist_checkpoint(save_path: str) -> bool:
    if not os.path.isdir(save_path):
        return False

    tracker_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker_path):
        try:
            with open(tracker_path) as f:
                marker = f.read().strip()
        except OSError:
            marker = ""
        if marker == "release":
            return os.path.isdir(os.path.join(save_path, "release"))
        if marker.isdigit():
            iter_dir = f"iter_{int(marker):07d}"
            return os.path.isdir(os.path.join(save_path, iter_dir))

    try:
        return any(
            entry.is_dir()
            and (entry.name == "release" or entry.name.startswith("iter_"))
            for entry in os.scandir(save_path)
        )
    except OSError:
        return False


def build_slime_app(
    *,
    training_run_id: str,
    slime: SlimeRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
) -> App:
    """Return a Modal App with `download`, `prepare_dataset`, `convert_checkpoint`, and `train` defined."""
    app_name = name or f"slime-{type(slime).__name__.lstrip('_').lower()}"

    SlimeRecipe._validate_custom_model_architecture(model)
    SlimeRecipe._validate_dataset(dataset)

    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file and os.path.isfile(mod_file):
            caller_script = os.path.abspath(mod_file)

    # ── Image ────────────────────────────────────────────────────────────────
    image = _build_slime_base_image()

    if slime.image_overlay is not None:
        image = slime.image_overlay(image)
        object.__setattr__(slime, "image_overlay", None)

    if isinstance(dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    if slime.local_slime:
        image = image.add_local_dir(
            slime.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

    image = image.add_local_python_source("modal_training_gym", copy=True)
    image = mount_tools_dir(image)

    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        caller_remote_path = f"/root/{caller_module_name}.py"
        image = image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )

    def _get_custom_generate_path() -> str:
        cfg = slime.extra_config
        if not isinstance(cfg, dict):
            return ""
        raw = cfg.get("custom_generate_function_path", "")
        return raw if isinstance(raw, str) else ""

    def _set_custom_generate_path(path: str) -> None:
        cfg = dict(slime.extra_config or {})
        cfg["custom_generate_function_path"] = path
        object.__setattr__(slime, "extra_config", cfg)

    def _ship_callable(
        fn: Any,
        *,
        fallback_name: str,
        set_path: Callable[[str], None],
    ) -> None:
        nonlocal image
        if fn is None:
            return
        fn_mod = getattr(fn, "__module__", None) or ""
        if fn_mod.startswith("modal_training_gym"):
            return
        try:
            fn_file = os.path.abspath(inspect.getfile(fn))
        except (TypeError, OSError):
            fn_file = None
        if fn_file and os.path.isfile(fn_file) and fn_file != caller_script:
            fn_module_name = os.path.splitext(os.path.basename(fn_file))[0]
            image = image.add_local_file(
                fn_file,
                remote_path=f"/root/{fn_module_name}.py",
                copy=True,
            )
            return
        fn_name = getattr(fn, "__name__", fallback_name)
        try:
            payload = base64.b64encode(cloudpickle.dumps(fn)).decode("ascii")
        except Exception:
            src = textwrap.dedent(inspect.getsource(fn))
            module_src = src
        else:
            module_src = textwrap.dedent(
                f"""
                import base64
                import cloudpickle

                {fn_name} = cloudpickle.loads(base64.b64decode({payload!r}))
                """
            ).lstrip()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"notebook_{fallback_name}_",
            delete=False,
        ) as tmp:
            tmp.write(module_src)
            tmp_path = tmp.name
        mod_name = os.path.splitext(os.path.basename(tmp_path))[0]
        image = image.add_local_file(
            tmp_path,
            remote_path=f"/root/{mod_name}.py",
            copy=True,
        )
        set_path(f"{mod_name}.{fn_name}")

    def _set_custom_rm_path(path: str) -> None:
        cfg = dict(slime.extra_config or {})
        cfg["custom_rm_path"] = path
        object.__setattr__(slime, "extra_config", cfg)

    _ship_callable(
        slime.custom_rm_function,
        fallback_name="custom_rm",
        set_path=_set_custom_rm_path,
    )
    _ship_callable(
        slime.custom_generate_function,
        fallback_name="custom_generate",
        set_path=_set_custom_generate_path,
    )
    _ship_callable(
        slime.rollout_function if callable(slime.rollout_function) else None,
        fallback_name="rollout_function",
        set_path=lambda path: object.__setattr__(slime, "rollout_function", path),
    )

    if slime.custom_rm_function is not None:
        object.__setattr__(slime, "custom_rm_function", None)
    if slime.custom_generate_function is not None and _get_custom_generate_path():
        object.__setattr__(slime, "custom_generate_function", None)

    # ── SGLang request params auto-wiring ─────────────────────────────────
    if slime.sglang_request_params:
        cfg = dict(slime.extra_config or {})
        cfg["sglang_request_params"] = slime.sglang_request_params
        if "custom_rm_path" not in cfg:
            cfg["custom_rm_path"] = (
                "modal_training_gym.frameworks.slime.opd_reward.reward_func"
            )
        if "custom_reward_post_process_path" not in cfg:
            cfg["custom_reward_post_process_path"] = (
                "modal_training_gym.frameworks.slime.opd_reward.post_process_rewards"
            )
        object.__setattr__(slime, "extra_config", cfg)

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{app_name}-data", create_if_missing=True)
    checkpoints_volume_name = (
        checkpoint.checkpoints_volume_name
        if checkpoint is not None and checkpoint.checkpoints_volume_name
        else f"{app_name}-checkpoints"
    )
    checkpoints_mount_path = (
        checkpoint.checkpoints_mount_path.rstrip("/") or "/"
        if checkpoint is not None and checkpoint.checkpoints_mount_path
        else str(CHECKPOINTS_PATH).rstrip("/")
    )
    checkpoints_volume = Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    if checkpoint is not None and checkpoint.path and not model.model_path:
        model.model_path = checkpoint.path
    all_volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        checkpoints_mount_path: checkpoints_volume,
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
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=2 * 60 * 60,
        secrets=hf_secrets(),
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
        secrets=hf_secrets(),
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        data_volume.reload()
        prompt_data, eval_paths = SlimeRecipe._resolve_data_paths(dataset)
        if dataset.always_prepare and os.path.exists(prompt_data):
            import shutil

            data_dir = os.path.dirname(prompt_data)
            print(f"always_prepare=True — removing {data_dir}")
            shutil.rmtree(data_dir, ignore_errors=True)
        dataset.prepare(prompt_data, eval_paths)
        dataset.validate_prepared(prompt_data)
        for ep in (eval_paths or {}).values():
            dataset.validate_prepared(ep)
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
    @clustered(convert_nnodes, rdma=True)  # pyright: ignore[reportCallIssue, reportOptionalCall]
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
        if _has_torch_dist_checkpoint(save_path):
            print(
                f"Found existing torch_dist checkpoint at {save_path}; skipping conversion."
            )
            return
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

        if num_nodes > 1:
            spec = importlib.util.find_spec(
                "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
            )
            convert_script = spec.origin if spec is not None else None
            if not convert_script:
                raise RuntimeError(
                    "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist not found"
                )
        else:
            convert_script = f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"

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
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="convert_to_hf",
    )
    def convert_to_hf(input_dir: str, output_dir: str):
        from huggingface_hub import snapshot_download

        import importlib.util

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        hf_path = (
            str(model.model_path)
            if model.model_path
            else snapshot_download(model.model_name, local_files_only=True)
        )

        spec = importlib.util.find_spec(
            "modal_training_gym.frameworks.slime.modal_helpers.convert_torch_dist_to_hf"
        )
        convert_script = spec.origin if spec is not None else None
        if not convert_script:
            raise RuntimeError(
                "modal_training_gym.frameworks.slime.modal_helpers.convert_torch_dist_to_hf not found"
            )

        cmd = (
            f"python {convert_script} "
            f"--input-dir {shlex.quote(input_dir)} "
            f"--output-dir {shlex.quote(output_dir)} "
            f"--origin-hf-dir {shlex.quote(hf_path)} "
            f"--force"
        )
        print(f"Converting: {cmd}")
        subprocess.run(["bash", "-c", cmd], check=True)
        checkpoints_volume.commit()
        print(f"Saved HF checkpoint to {output_dir}")

    _multi_node = slime.total_nodes > 1

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        secrets=[
            *(
                []
                if slime.wandb is None
                else [Secret.from_name(slime.wandb.modal_wandb_secret_name)]
            ),
        ],
        timeout=24 * 60 * 60,
        experimental_options={"efa_enabled": True} if _multi_node else {},
        serialized=True,
        name="train",
    )
    @clustered(slime.total_nodes, rdma=_multi_node)  # pyright: ignore[reportCallIssue, reportOptionalCall]
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
    ):
        modal_app_id = modal_app_id or os.environ.get("MODAL_APP_ID", "")
        modal_app_url = modal_app_url or modal_app_dashboard_url(modal_app_id)
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

        created_at = int(time.time())
        print(f"Training run id: {training_run_id}")
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
                "hf_repo": getattr(dataset, "hf_repo", ""),
                "name": type(dataset).__name__,
            },
            "lr": slime.lr,
            "global_batch_size": slime.global_batch_size,
        }
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url or modal_app_dashboard_url(modal_app_id),
            framework=Framework.SLIME,
            config=config_summary,
            created_at=created_at,
            started_at=created_at,
        )
        await run_record.save_async()
        print(f"TrainingRun recorded: {training_run_id}")

        try:
            if model:
                cache_dir = (
                    HF_CACHE_PATH
                    / "hub"
                    / f"models--{model.model_name.replace('/', '--')}"
                )
                snapshots_dir = cache_dir / "snapshots"
                has_snapshot = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
                if not has_snapshot:
                    print(f"Downloading model {model.model_name}...")
                    model.download()
                    await hf_cache_volume.commit.aio()

            if dataset:
                prompt_data, eval_paths = SlimeRecipe._resolve_data_paths(dataset)
                needs_prepare = not os.path.exists(prompt_data)
                if dataset.always_prepare and os.path.exists(prompt_data):
                    import shutil

                    data_dir = os.path.dirname(prompt_data)
                    print(f"always_prepare=True — removing {data_dir}")
                    shutil.rmtree(data_dir, ignore_errors=True)
                    needs_prepare = True
                if needs_prepare:
                    print(f"Preparing dataset ({prompt_data})...")
                    dataset.prepare(prompt_data, eval_paths)
                    await data_volume.commit.aio()
                dataset.validate_prepared(prompt_data)
                for ep in (eval_paths or {}).values():
                    if os.path.exists(ep):
                        dataset.validate_prepared(ep)

            prepare_slime_config(slime, model, tempfile.mkdtemp())

            if wandb_key := os.environ.get("WANDB_API_KEY", ""):
                if slime.wandb is not None:
                    slime.wandb.key = wandb_key

            recipe_default_save_root = str(CHECKPOINTS_PATH).rstrip("/")
            mounted_save_root = checkpoints_mount_path
            configured_save_root = (
                str(slime.save).rstrip("/") if slime.save else mounted_save_root
            )
            base_save_root = (
                mounted_save_root
                if configured_save_root == recipe_default_save_root
                else configured_save_root
            )
            save_root = (
                f"{mounted_save_root}/{training_run_id}"
                if base_save_root == mounted_save_root
                else configured_save_root
            )
            os.makedirs(save_root, exist_ok=True)

            original_save = slime.save
            object.__setattr__(slime, "save", save_root)
            try:
                cmd = build_train_cmd(slime, SLIME_ROOT, model=model, dataset=dataset)
            finally:
                object.__setattr__(slime, "save", original_save)
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

            if model and slime.megatron_to_hf_mode == "bridge":
                from huggingface_hub import snapshot_download as _snap

                hf_path = (
                    str(model.model_path)
                    if model.model_path
                    else _snap(model.model_name, local_files_only=True)
                )
                prefix = _get_slime_checkpoint_prefix()
                ckpt_dirs = (
                    sorted(
                        (
                            d
                            for d in os.scandir(save_root)
                            if d.is_dir()
                            and d.name.startswith(prefix)
                            and not d.name.endswith("_hf")
                        ),
                        key=lambda d: d.name,
                    )
                    if prefix
                    else []
                )

                for entry in ckpt_dirs:
                    hf_dir = f"{entry.path}_hf"
                    if not os.path.exists(hf_dir):
                        cmd = (
                            f"python {SLIME_ROOT}/tools/convert_torch_dist_to_hf.py "
                            f"--input-dir {shlex.quote(entry.path)} "
                            f"--output-dir {shlex.quote(hf_dir)} "
                            f"--origin-hf-dir {shlex.quote(hf_path)} "
                            f"--force"
                        )
                        print(f"Converting {entry.name} → HF: {cmd}")
                        subprocess.run(["bash", "-c", cmd], check=True)
                        print(f"Converted {entry.name} → {hf_dir}")

            result_kwargs = {
                "app_name": app_name,
                "framework": "slime",
                "training_run_id": training_run_id,
                "checkpoint_dir": save_root,
                "model_config": model,
                "checkpoints_volume_name": checkpoints_volume_name,
                "checkpoints_mount_path": checkpoints_mount_path,
            }
            accepted_fields = set(inspect.signature(TrainResult).parameters)
            result = TrainResult(
                **{k: v for k, v in result_kwargs.items() if k in accepted_fields}
            )
            await result.save_async()
            run_record.status = TrainingRunStatus.COMPLETED
            await checkpoints_volume.commit.aio()
            print(f"TrainResult saved: {training_run_id}")
            return result._to_dict()
        except KeyboardInterrupt:
            run_record.status = TrainingRunStatus.STOPPED
            raise
        except BaseException:
            run_record.status = TrainingRunStatus.FAILED
            raise
        finally:
            finished_at = int(time.time())
            run_record.ended_at = finished_at
            if run_record.completed_at is None:
                run_record.completed_at = finished_at
            run_record.duration_seconds = max(0, finished_at - run_record.started_at)
            await run_record.save_async()

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
