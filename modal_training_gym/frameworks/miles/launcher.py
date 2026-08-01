import asyncio
import hashlib
import os
import shlex
import subprocess
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from modal import App, Image, Retries, Secret, Volume
from modal.experimental import clustered

from modal_training_gym.common import (
    hf_secrets,
    proxy_auth_secrets,
)
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.dataset import DatasetConfig, HarborDataset
from modal_training_gym.common.framework import (
    Framework,
    mount_tools_dir,
)
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.ray_cluster import ModalRayCluster
from modal_training_gym.common.run import (
    TrainingRun,
    TrainingRunStatus,
    has_torch_dist_checkpoint,
    mark_training_attempt_finished,
    record_resume_checkpoint,
    torch_dist_resume_checkpoint,
)
from modal_training_gym.common.launcher_helpers import (
    build_app_tags,
    build_terminal_run_record,
    build_train_result,
    compute_save_root,
    init_training_run_record,
    mark_run_failed,
    mark_run_stopped,
    redact_env_values,
    resolve_caller_context,
    resolve_checkpoint_volumes,
    run_download_phase,
    run_prepare_dataset,
    ship_callable,
)
from modal_training_gym.common.status import MilesStatus
from modal_training_gym.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MilesConfig,
)
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.frameworks.miles.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    prepare_miles_config,
    resolve_checkpoint_ref,
)

MILES_ROOT = "/root/miles"
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"

_MILES_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_SGLANG_ABORT_B64 = encode_patch("patch_sglang_abort", _MILES_PATCHES)


def _build_miles_base_image(miles: MilesConfig) -> Image:
    image = (
        Image.from_registry(miles.docker_image)
        .entrypoint([])
        .run_commands(
            f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3",
        )
    )
    if miles.image_env:
        image = image.env(miles.image_env)
    if miles.image_run_commands:
        image = image.run_commands(*miles.image_run_commands)
    return image


def build_miles_app(
    *,
    training_run_id: str,
    miles: MilesConfig,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> App:
    app_name = name or miles.name or f"miles-{type(miles).__name__.lstrip('_').lower()}"
    volume_prefix = miles.name or f"miles-{type(miles).__name__.lstrip('_').lower()}"

    _caller_module, caller_script = resolve_caller_context()

    image = _build_miles_base_image(miles)

    for patch_file in miles.patch_files:
        image = image.add_local_file(
            patch_file,
            remote_path=f"/tmp/{os.path.basename(patch_file)}",
            copy=True,
        )

    if miles.local_miles:
        image = image.add_local_dir(
            miles.local_miles,
            remote_path=MILES_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

    if miles.image_overlay is not None:
        image = miles.image_overlay(image)
        miles.image_overlay = None

    if isinstance(dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    image = image.add_local_python_source("modal_training_gym", copy=True)
    image = image.uv_pip_install("randomname")
    image = mount_tools_dir(image)

    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        image = image.add_local_file(
            caller_script,
            remote_path=f"/root/{caller_module_name}.py",
            copy=True,
        )

    def _set_custom_config_value(key: str, value: str) -> None:
        cfg = (
            dict(miles.custom_config_path or {})
            if isinstance(miles.custom_config_path, dict)
            else {}
        )
        cfg[key] = value
        miles.custom_config_path = cfg

    def _ship_callable(
        fn: Any,
        *,
        fallback_name: str,
        set_path: Callable[[str], None],
    ) -> None:
        nonlocal image
        image = ship_callable(
            image,
            fn,
            caller_script=caller_script,
            fallback_name=fallback_name,
            set_path=set_path,
        )

    _ship_callable(
        miles.custom_rm_function,
        fallback_name="custom_rm",
        set_path=lambda path: _set_custom_config_value("custom_rm_path", path),
    )
    _ship_callable(
        miles.custom_generate_function,
        fallback_name="custom_generate",
        set_path=lambda path: _set_custom_config_value(
            "custom_generate_function_path", path
        ),
    )
    miles.custom_rm_function = None
    miles.custom_generate_function = None

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{volume_prefix}-data", create_if_missing=True)
    checkpoints_volume_name, checkpoints_mount_path, checkpoints_volume = (
        resolve_checkpoint_volumes(
            checkpoint,
            volume_prefix=volume_prefix,
            default_mount_path=str(CHECKPOINTS_PATH),
        )
    )
    if checkpoint is not None and checkpoint.path and not model.model_path:
        model.model_path = checkpoint.path

    all_volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        checkpoints_mount_path: checkpoints_volume,
    }

    tags = build_app_tags(
        framework="miles",
        model=model,
        recipe_app_tags=miles.app_tags,
        wandb=miles.wandb,
    )

    app = App(app_name, tags=tags)
    gpu_spec = f"{miles.gpu_type}:{miles.actor_num_gpus_per_node}"

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        # proxy_auth_secrets: this phase reports status to the dashboard,
        # which may require proxy auth.
        secrets=[*hf_secrets(), *proxy_auth_secrets()],
        serialized=True,
        name="download",
    )
    def download(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        def _download() -> None:
            model.download()
            miles.download_model()
            miles.post_process_model()

        run_download_phase(
            training_run_id=training_run_id,
            phase=MilesStatus.DOWNLOAD_MODEL.value,
            framework_status_url=framework_status_url,
            framework_status_token=framework_status_token,
            volumes=(hf_cache_volume, checkpoints_volume),
            download=_download,
        )

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=4 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        run_prepare_dataset(dataset, data_volume, MilesConfig._resolve_data_paths)

    convert_nnodes = get_checkpoint_conversion_policy(miles, model=model)[0]
    convert_multi_node = convert_nnodes > 1

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        # See download: status reports must pass the dashboard's proxy auth.
        secrets=proxy_auth_secrets(),
        experimental_options={"efa_enabled": True} if convert_multi_node else {},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered(convert_nnodes, rdma=convert_multi_node)
    def convert_checkpoint(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
            flush as flush_status_reporter,
        )

        if training_run_id:
            enqueue_framework_status(
                training_run_id,
                MilesStatus.CONVERT_MODEL.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )

        if getattr(miles, "megatron_to_hf_mode", None) == "bridge":
            print("Bridge mode - no conversion needed.")
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        conversion_hf_checkpoint = (
            getattr(miles, "megatron_conversion_hf_checkpoint", None)
            or getattr(miles, "hf_checkpoint", "")
            or model.model_path
            or model.model_name
        )
        hf_path = resolve_checkpoint_ref(conversion_hf_checkpoint)
        save_path = str(miles.ref_load)
        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            miles, model=model
        )

        if has_torch_dist_checkpoint(save_path):
            print(
                f"Found existing torch_dist checkpoint at {save_path}; skipping conversion."
            )
            return

        if num_nodes == 1:
            node_rank, master_addr, nnodes = 0, "127.0.0.1", 1
        else:
            import modal.experimental

            info = modal.experimental.get_cluster_info()
            node_rank = info.rank
            master_addr = info.container_ipv4_ips[0]
            nnodes = len(info.container_ipv4_ips)

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        import importlib.util

        spec = importlib.util.find_spec(
            "modal_training_gym.frameworks.miles.modal_helpers.convert_hf_to_torch_dist"
        )
        convert_script = (
            spec.origin
            if spec is not None and num_nodes > 1
            else f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
        )
        if not convert_script:
            raise RuntimeError("Miles checkpoint conversion script not found")

        if miles.miles_model_script:
            cmd = (
                f"source {MILES_ROOT}/{miles.miles_model_script} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        else:
            cmd = (
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )

        env = {**os.environ, **miles.environment}
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"

        print(
            f"Conversion layout: nodes={num_nodes}, nproc_per_node={nproc_per_node}, "
            f"node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        subprocess.run(["bash", "-c", cmd], check=True, env=env)
        checkpoints_volume.commit()

        if node_rank == 0:
            print(f"Saved Megatron torch_dist checkpoint to {save_path}")

        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

    _multi_node = miles.total_nodes > 1

    @app.function(
        image=image,
        gpu=gpu_spec,
        memory=miles.memory,
        cloud=miles.cloud,
        region=miles.region,
        volumes=all_volumes,
        secrets=[
            *(
                []
                if miles.wandb is None
                else [Secret.from_name(miles.wandb.modal_wandb_secret_name)]
            ),
            *proxy_auth_secrets(),
        ],
        timeout=24 * 60 * 60,
        retries=Retries(max_retries=10, initial_delay=0.0),
        single_use_containers=True,
        experimental_options={"efa_enabled": True} if _multi_node else {},
        serialized=True,
        name="train",
    )
    @clustered(miles.total_nodes, rdma=_multi_node)
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        modal_app_id = modal_app_id or os.environ.get("MODAL_APP_ID", "")
        modal_app_url = modal_app_url or modal_app_dashboard_url(modal_app_id)
        if framework_status_url:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_URL"] = framework_status_url
        if framework_status_token:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] = framework_status_token

        await asyncio.gather(
            hf_cache_volume.reload.aio(),
            data_volume.reload.aio(),
            checkpoints_volume.reload.aio(),
        )

        cluster = ModalRayCluster()
        cluster.discover_cluster(miles.total_nodes)

        os.environ["MILES_HOST_IP"] = cluster.node_ip
        os.environ["SGLANG_HOST_IP"] = cluster.node_ip
        os.environ["HOST_IP"] = cluster.node_ip

        prep_id = hashlib.sha1(training_run_id.encode("utf-8")).hexdigest()[:16]
        prep_marker = os.path.join(
            checkpoints_mount_path, f".training_gym_prepared_{prep_id}"
        )
        prep_error = f"{prep_marker}.error"

        run_record: TrainingRun | None = None

        wandb_entity = ""
        wandb_run_id = ""

        if cluster.is_head:
            if miles.wandb is not None:
                from modal_training_gym.common.wandb import preflight_wandb

                wandb_entity = preflight_wandb(miles.wandb)
            wandb_run_id = ""

            print(f"Training run id: {training_run_id}")
            config_summary = {
                "model": {"model_name": model.model_name} if model else {},
                "recipe": {
                    "gpu_type": miles.gpu_type,
                    "actor_num_nodes": miles.actor_num_nodes,
                    "actor_num_gpus_per_node": miles.actor_num_gpus_per_node,
                },
                "wandb": (
                    {
                        "project": miles.wandb.project,
                        "group": miles.wandb.group,
                        "entity": wandb_entity,
                        "run_id": wandb_run_id,
                    }
                    if miles.wandb
                    else {}
                ),
                "dataset": {
                    "hf_repo": getattr(dataset, "hf_repo", ""),
                    "name": type(dataset).__name__,
                },
                "lr": miles.lr,
                "global_batch_size": miles.global_batch_size,
            }
            (
                run_record,
                wandb_run_id,
                framework_status_token,
            ) = await init_training_run_record(
                training_run_id=training_run_id,
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_url,
                framework=Framework.MILES,
                initializing_status=MilesStatus.INITIALIZING,
                config_summary=config_summary,
                wandb_cfg=miles.wandb,
                wandb_entity=wandb_entity,
                framework_status_token=framework_status_token,
            )

        # In-flight status updates are fire-and-forget HTTP POSTs to the
        # dashboard so they don't block on Modal Volume writes. Terminal
        # state is committed synchronously below.
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
        )

        async def _set_framework_status(status: MilesStatus) -> None:
            if run_record is None:
                return
            run_record.framework_status = status
            enqueue_framework_status(
                training_run_id, status.value, token=framework_status_token
            )

        async def _prepare_shared_inputs() -> None:
            await _set_framework_status(MilesStatus.DOWNLOAD_MODEL)
            if model:
                cache_dir = (
                    HF_CACHE_PATH
                    / "hub"
                    / ("models--" + model.model_name.replace("/", "--"))
                )
                snapshots_dir = cache_dir / "snapshots"
                has_snapshot = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
                model_path = getattr(model, "model_path", None)
                has_model_path = True
                if model_path:
                    model_path_obj = Path(model_path)
                    has_model_path = model_path_obj.exists() and (
                        not model_path_obj.is_dir() or any(model_path_obj.iterdir())
                    )
                if not has_snapshot or not has_model_path:
                    print(f"Downloading model {model.model_name}...")
                    model.download()
                if hasattr(model, "prepare_runtime_cache"):
                    model.prepare_runtime_cache()

            miles.download_model()
            await _set_framework_status(MilesStatus.CONVERT_MODEL)
            miles.post_process_model()
            await hf_cache_volume.commit.aio()
            await checkpoints_volume.commit.aio()

            await _set_framework_status(MilesStatus.PREPARE_DATASET)
            prompt_data, eval_paths = MilesConfig._resolve_data_paths(dataset)
            needs_prepare = not os.path.exists(prompt_data)
            if dataset.always_prepare and os.path.exists(prompt_data):
                data_dir = os.path.dirname(prompt_data)
                print(f"always_prepare=True - removing {data_dir}")
                shutil.rmtree(data_dir, ignore_errors=True)
                needs_prepare = True
            if needs_prepare:
                print(f"Preparing dataset ({prompt_data})...")
                dataset.prepare(prompt_data, eval_paths)
                await data_volume.commit.aio()
            dataset.validate_prepared(prompt_data)
            for ep in (eval_paths or {}).values():
                dataset.validate_prepared(ep)

        if cluster.is_head:
            try:
                await _prepare_shared_inputs()
            except BaseException as exc:
                if run_record is not None:
                    finished_at = int(time.time())
                    run_record.status = TrainingRunStatus.FAILED
                    run_record.error_message = (
                        run_record.error_message or f"{type(exc).__name__}: {exc}"
                    )
                    mark_training_attempt_finished(
                        run_record, status="failed", ended_at=finished_at
                    )
                    run_record.ended_at = finished_at
                    run_record.completed_at = finished_at
                    run_record.duration_seconds = max(
                        0, finished_at - run_record.started_at
                    )
                    await run_record.save(is_async=True)
                os.makedirs(os.path.dirname(prep_error), exist_ok=True)
                with open(prep_error, "w") as f:
                    f.write(repr(exc))
                await checkpoints_volume.commit.aio()
                raise
            os.makedirs(os.path.dirname(prep_marker), exist_ok=True)
            with open(prep_marker, "w") as f:
                f.write(str(time.time()))
            await checkpoints_volume.commit.aio()
        else:
            deadline = time.time() + 4 * 60 * 60
            while True:
                await asyncio.gather(
                    hf_cache_volume.reload.aio(),
                    data_volume.reload.aio(),
                    checkpoints_volume.reload.aio(),
                )
                if os.path.exists(prep_marker):
                    break
                if os.path.exists(prep_error):
                    with open(prep_error) as f:
                        raise RuntimeError(f"Head preparation failed: {f.read()}")
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for head preparation marker")
                await asyncio.sleep(5)

        cluster.start_ray()

        if not cluster.is_head:
            await cluster.wait_forever()
            return
        assert run_record is not None

        try:  # Wraps all post-setup work so any failure marks the run terminal.
            prepare_miles_config(miles, model, tempfile.mkdtemp())

            if wandb_key := os.environ.get("WANDB_API_KEY", ""):
                if miles.wandb is not None:
                    miles.wandb.key = wandb_key

            save_root = compute_save_root(
                miles.save,
                recipe_default_save_root=str(CHECKPOINTS_PATH).rstrip("/"),
                mounted_save_root=checkpoints_mount_path,
                training_run_id=training_run_id,
            )

            original_save = miles.save
            original_load = miles.load
            miles.save = save_root
            resume_checkpoint = torch_dist_resume_checkpoint(save_root)
            record_resume_checkpoint(run_record, resume_checkpoint)
            await run_record.save(is_async=True)

            if resume_checkpoint is not None:
                print(
                    f"WARNING: detected existing checkpoint in "
                    f"{resume_checkpoint['resume_checkpoint_path']}; "
                    "resuming training from last saved iteration."
                )
                miles.load = save_root
            try:
                cmd = build_train_cmd(miles, MILES_ROOT, model=model, dataset=dataset)
            finally:
                miles.save = original_save
                miles.load = original_load

            wandb_env = {}
            if wandb_run_id:
                wandb_env["WANDB_RUN_ID"] = wandb_run_id
                wandb_env["WANDB_RESUME"] = "allow"
            if wandb_entity:
                wandb_env["WANDB_ENTITY"] = wandb_entity

            runtime_env = {
                "env_vars": {
                    "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                    "MASTER_ADDR": cluster.head_addr,
                    **wandb_env,
                    **miles.environment,
                }
            }

            mode = "async" if miles.async_mode else "sync"
            print(
                f"Training {app_name} - {miles.total_nodes} node(s) x {gpu_spec} ({mode})"
            )
            print(miles.gpu_allocation.summary())
            printable_env = {
                **runtime_env,
                "env_vars": redact_env_values(runtime_env["env_vars"]),
            }
            print(f"Command: {cmd}, runtime_env: {printable_env}")

            await _set_framework_status(MilesStatus.TRAINING)
            async with cluster.forward_dashboard() as tunnel:
                print(f"Ray dashboard: {tunnel.url}")
                result = await cluster.submit_and_tail(cmd, runtime_env=runtime_env)
                if not result.is_success:
                    run_record.error_message = (
                        result.message
                        or f"Ray job finished with status: {result.status}"
                    )
                    raise RuntimeError(run_record.error_message)
                print(f"Ray job completed: {result.status}")
                print(f"Ray job message: {result.message}")

            result = build_train_result(
                app_name=app_name,
                framework=Framework.MILES,
                training_run_id=training_run_id,
                checkpoint_dir=save_root,
                model=model,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
                wandb_cfg=miles.wandb,
                wandb_entity=wandb_entity,
                wandb_run_id=wandb_run_id,
                group_id=group_id,
            )
            await result.save(is_async=True)
            run_record.status = TrainingRunStatus.COMPLETED
            mark_training_attempt_finished(
                run_record, status="completed", ended_at=int(time.time())
            )
            await checkpoints_volume.commit.aio()
            print(f"TrainResult saved: {training_run_id}")
            return result._to_dict()
        except KeyboardInterrupt:
            mark_run_stopped(run_record)
            raise
        except BaseException as exc:
            mark_run_failed(run_record, exc)
            raise
        finally:
            latest_run_record = await build_terminal_run_record(
                run_record, training_run_id
            )
            try:
                await latest_run_record.save(is_async=True)
            except Exception:
                pass

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
