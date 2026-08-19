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
from modal_training_gym.common.launcher_utils import (
    serialize_recipe_params,
    timing_debug_env,
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
    MilesRecipe,
)
from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    REPORTING_PATCH_COMMANDS,
    SGLANG_ABORT_PATCH_COMMAND,
    SUBSTEP_TIMING_PATCH_COMMAND,
)
from modal_training_gym.frameworks.miles.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    model_args_command,
    prepare_miles_config,
    resolve_checkpoint_ref,
)


def _validate_resume_checkpoint(
    resume_from_iteration: int | None, num_rollout: int
) -> None:
    if resume_from_iteration is not None and resume_from_iteration + 1 > num_rollout:
        raise RuntimeError(
            f"Resume would start at rollout {resume_from_iteration + 1}, "
            f"but num_rollout={num_rollout}; nothing would run."
        )
    if resume_from_iteration is not None and resume_from_iteration + 1 == num_rollout:
        print(
            "WARNING: Resume checkpoint is already at the final configured "
            "rollout; the retry will exit without running another rollout.",
            flush=True,
        )


MILES_ROOT = "/root/miles"
SYSTEM_LIB_DIR = "/usr/lib/x86_64-linux-gnu"
# libibverbs and the libmlx5 provider come from incompatible rdma package versions for miles multi-node training
# reinstalling fixes this issue, mooncake transferengine imports successfully
RDMA_RUNTIME_INSTALL_COMMAND = (
    "apt-get update && apt-get install -y --no-install-recommends "
    "--reinstall libibverbs1 ibverbs-providers && "
    "rm -rf /var/lib/apt/lists/*"
)
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"


def _build_miles_base_image(miles: MilesRecipe) -> Image:
    image = (
        Image.from_registry(miles.docker_image)
        .entrypoint([])
        .run_commands(
            f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
            SGLANG_ABORT_PATCH_COMMAND,
            *REPORTING_PATCH_COMMANDS,
            SUBSTEP_TIMING_PATCH_COMMAND,
        )
    )
    if miles.total_nodes > 1:
        image = image.run_commands(RDMA_RUNTIME_INSTALL_COMMAND)
    if miles.image_env:
        image = image.env(miles.image_env)
    return image


def _response_parser_path(model: Any) -> str:
    """Import path of the model's response parser so the rollout recorder can
    resolve and apply it remotely. Empty when the model sets no parser."""
    fn = getattr(model, "response_parser", None) if model is not None else None
    if fn is None:
        return ""
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "")
    return f"{module}.{qualname}" if module and qualname else ""


def _compose_ld_library_path() -> str:
    parts = [SYSTEM_LIB_DIR]
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if part and part not in parts:
            parts.append(part)
    return ":".join(parts)


def build_ray_runtime_env(
    *,
    head_addr: str,
    wandb_env: dict[str, str],
    environment: dict,
    extra_env: dict[str, str] | None = None,
    framework_status_token: str = "",
    substep_timing: str = "auto",
) -> dict:
    """Runtime env for the Ray job that runs miles.

    Ray workers do not pick up the container's linker path on their own, and
    without it the Megatron actor can resolve a libibverbs that does not match
    the image's libmlx5 and die importing mooncake. The system lib dir is put
    in front for that reason; the rest is read from the container, so whatever
    the image exports — including any wheel-shipped nvidia lib dirs — is
    carried through. Composing it here rather than in an ``image_env`` entry
    keeps it independent of whether the base image exports ``LD_LIBRARY_PATH``
    in its own ``ENV``: a Dockerfile ``$LD_LIBRARY_PATH`` expands to an empty
    string when it does not, which would drop those dirs and leave a trailing
    empty entry that the loader reads as the working directory. A recipe can
    still override the whole thing through ``environment``.
    """
    env_vars: dict[str, str] = {
        "no_proxy": f"127.0.0.1,{head_addr}",
        "MASTER_ADDR": head_addr,
        "LD_LIBRARY_PATH": _compose_ld_library_path(),
        "TRAINING_GYM_SUBSTEP_TIMING": substep_timing,
    }
    env_vars.update(extra_env or {})
    env_vars.update(wandb_env)
    env_vars.update(environment)
    env_vars.update(timing_debug_env())
    if framework_status_token:
        # Applied after `environment` so a recipe override can't blank the
        # dashboard auth token by accident.
        env_vars["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
    return {"env_vars": env_vars}


def build_miles_app(
    *,
    training_run_id: str,
    miles: MilesRecipe,
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
        # The local checkout just overwrote the patched miles sources;
        # re-apply the built-in patches.
        image = image.run_commands(
            SGLANG_ABORT_PATCH_COMMAND
            + " || echo 'WARNING: sglang abort patch did not apply to the"
            " local_miles checkout; transient router failures during rollout"
            " cleanup may crash the run'",
            *REPORTING_PATCH_COMMANDS,
        )

    if miles.image_run_commands:
        image = image.run_commands(*miles.image_run_commands)

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
        cfg = dict(miles.extra_config or {})
        cfg[key] = value
        miles.extra_config = cfg

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

    # rm/generate paths live in the YAML custom-config; Miles reads the rest off
    # dedicated --<name>-path flags, so those resolve back onto the field itself
    # and MilesRecipe._fields emits them.
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

    for attr, fallback_name in (
        ("custom_reward_post_process_function", "custom_reward_post_process"),
        ("rollout_function", "rollout_function"),
    ):
        value = getattr(miles, attr)
        # A str is already an import path the user vouches for — nothing to ship.
        if not callable(value):
            continue
        _ship_callable(
            value,
            fallback_name=fallback_name,
            set_path=lambda path, attr=attr: object.__setattr__(miles, attr, path),
        )

    # The gym intercepts these four hooks for dashboard reporting: the CLI
    # flag always points at the phase-reporting wrapper (MilesRecipe._fields),
    # and the user's own hook rides along in the YAML custom-config under a
    # `training_gym_*` key. Str paths were stashed there by the recipe
    # validator; inline callables are shipped by value here and the stashed
    # `__pending__` placeholder is overwritten with the resolved path.
    for attr, config_key, fallback_name in (
        (
            "custom_rollout_log_function",
            "training_gym_custom_rollout_log_function_path",
            "custom_rollout_log",
        ),
        (
            "custom_eval_rollout_log_function",
            "training_gym_custom_eval_rollout_log_function_path",
            "custom_eval_rollout_log",
        ),
        (
            "custom_megatron_before_log_prob_hook",
            "training_gym_custom_megatron_before_log_prob_hook_path",
            "before_log_prob_hook",
        ),
        (
            "custom_megatron_before_train_step_hook",
            "training_gym_custom_megatron_before_train_step_hook_path",
            "before_train_step_hook",
        ),
    ):
        value = getattr(miles, attr)
        if not callable(value):
            continue
        _ship_callable(
            value,
            fallback_name=fallback_name,
            set_path=lambda path, key=config_key: _set_custom_config_value(key, path),
        )
        setattr(miles, attr, None)

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{volume_prefix}-data", create_if_missing=True)
    checkpoints_volume_name, checkpoints_mount_path, checkpoints_volume = (
        resolve_checkpoint_volumes(
            checkpoint,
            volume_prefix=volume_prefix,
            default_mount_path=str(CHECKPOINTS_PATH),
        )
    )
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
        run_prepare_dataset(dataset, data_volume, MilesRecipe._resolve_data_paths)

    convert_nnodes = get_checkpoint_conversion_policy(miles, model=model)[0]
    convert_multi_node = convert_nnodes > 1

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=60 * 60,
        secrets=[*hf_secrets(), *proxy_auth_secrets()],
        serialized=True,
        name="resolve_checkpoint",
    )
    def resolve_checkpoint(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ) -> str | None:
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
            return None

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        save_path = str(miles.ref_load)
        if has_torch_dist_checkpoint(save_path):
            print(
                f"Found existing torch_dist checkpoint at {save_path}; "
                "skipping conversion."
            )
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return None

        conversion_hf_checkpoint = (
            getattr(miles, "megatron_conversion_hf_checkpoint", None)
            or getattr(miles, "hf_checkpoint", "")
            or model.model_path
            or model.model_name
        )
        return resolve_checkpoint_ref(conversion_hf_checkpoint)

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        secrets=proxy_auth_secrets() or None,
        experimental_options={"efa_enabled": True} if convert_multi_node else {},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered(convert_nnodes, rdma=convert_multi_node)
    def convert_checkpoint(
        hf_path: str,
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from modal_training_gym.common.status_reporter import (
            flush as flush_status_reporter,
        )

        save_path = str(miles.ref_load)
        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            miles, model=model
        )

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
        elif model_args_cmd := model_args_command(miles, MILES_ROOT):
            cmd = (
                f'MODEL_ARGS_LINE="$({model_args_cmd})" || exit 1; '
                f'read -ra MODEL_ARGS <<< "$MODEL_ARGS_LINE"; '
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

    train_secrets = [
        *(
            []
            if miles.wandb is None
            else [Secret.from_name(miles.wandb.modal_wandb_secret_name)]
        ),
        *hf_secrets(),
        *proxy_auth_secrets(),
    ]
    train_experimental_options: dict[str, Any] = (
        {"efa_enabled": True} if _multi_node else {}
    )

    train_function_kwargs = dict(miles.train_function_kwargs or {})
    user_secrets = train_function_kwargs.pop("secrets", None)
    if user_secrets is not None:
        if not isinstance(user_secrets, (list, tuple)):
            user_secrets = [user_secrets]
        train_secrets.extend(user_secrets)
    user_experimental_options = train_function_kwargs.pop("experimental_options", None)
    if user_experimental_options is not None:
        train_experimental_options.update(user_experimental_options)
    train_ephemeral_disk = train_function_kwargs.pop("ephemeral_disk", None)
    if train_function_kwargs:
        unsupported = ", ".join(sorted(train_function_kwargs))
        raise TypeError(f"Unsupported miles.train_function_kwargs keys: {unsupported}")

    @app.function(
        image=image,
        gpu=gpu_spec,
        memory=miles.memory,
        ephemeral_disk=train_ephemeral_disk,
        cloud=miles.cloud,
        region=miles.region,
        volumes=all_volumes,
        secrets=train_secrets,
        timeout=24 * 60 * 60,
        retries=Retries(max_retries=10, initial_delay=0.0),
        single_use_containers=True,
        experimental_options=train_experimental_options,
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
                    **serialize_recipe_params(miles, dataset=dataset, model=model),
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
            prompt_data, eval_paths = MilesRecipe._resolve_data_paths(dataset)
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
                resume_from_iteration = resume_checkpoint.get("resume_from_iteration")
                _validate_resume_checkpoint(resume_from_iteration, miles.num_rollout)
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

            phase_report_url = (
                os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL")
                or framework_status_url
                or ""
            )
            if not phase_report_url:
                print(
                    "WARNING: no dashboard URL passed to train() and no "
                    "TRAINING_GYM_FRAMEWORK_STATUS_URL set inside the "
                    "container. Phase reporting is disabled for this run."
                )

            wandb_env = {}
            if wandb_run_id:
                wandb_env["WANDB_RUN_ID"] = wandb_run_id
                wandb_env["WANDB_RESUME"] = "allow"
            if wandb_entity:
                wandb_env["WANDB_ENTITY"] = wandb_entity

            runtime_env = build_ray_runtime_env(
                head_addr=cluster.head_addr,
                wandb_env=wandb_env,
                environment=miles.environment,
                substep_timing=miles.substep_timing,
                extra_env={
                    "TRAINING_GYM_TRAINING_RUN_ID": training_run_id,
                    "TRAINING_GYM_APP_NAME": app_name,
                    "TRAINING_GYM_TOTAL_STEPS": str(miles.num_rollout),
                    "TRAINING_GYM_RESPONSE_PARSER_PATH": _response_parser_path(model),
                    "TRAINING_GYM_CAPTURE_TRACE": (
                        "1" if getattr(miles, "capture_trace", False) else ""
                    ),
                    "TRAINING_GYM_TRACE_SAMPLE_LIMIT": str(
                        getattr(miles, "trace_sample_limit", 16)
                    ),
                    "TRAINING_GYM_FRAMEWORK_STATUS_URL": phase_report_url,
                },
                framework_status_token=framework_status_token,
            )

            mode = "async" if miles.async_mode else "sync"
            print(
                f"Training {app_name} - {miles.total_nodes} node(s) x {gpu_spec} ({mode})"
            )
            print(miles.gpu_allocation.summary())
            print(f"Command: {cmd}, runtime_env: {runtime_env}")

            await _set_framework_status(MilesStatus.TRAINING)
            result = await cluster.submit_and_tail(cmd, runtime_env=runtime_env)
            if not result.is_success:
                message = (
                    result.message or f"Ray job finished with status: {result.status}"
                )
                error = RuntimeError(f"{message} (training_run_id={training_run_id})")
                error.training_run_id = training_run_id  # pyright: ignore[reportAttributeAccessIssue]  # exception metadata is consumed by downstream callers
                run_record.error_message = str(error)
                raise error
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
