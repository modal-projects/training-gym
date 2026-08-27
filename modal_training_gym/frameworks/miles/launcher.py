import asyncio
import contextlib
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from modal import App, Dict as ModalDict, Image, Retries, Volume
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
from modal_training_gym.common.metrics import (
    metric_metadata,
    metric_runtime_env,
    metric_secrets,
    preflight_metric,
)
from modal_training_gym.common.wandb import WandbConfig
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
    write_dataset_if_needed,
)
from modal_training_gym.common.status import MilesStatus
from modal_training_gym.train_recipes.miles_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MilesRecipe,
)
from modal_training_gym.common.patches import _MEGATRON_PATCHES, encode_patch
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

_MILES_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_SGLANG_ABORT_B64 = encode_patch("patch_sglang_abort", _MILES_PATCHES)
_PATCH_ROLLOUT_STATUS_B64 = encode_patch(
    "patch_rollout_status_reporting", _MILES_PATCHES
)
_PATCH_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", _MILES_PATCHES)
_PATCH_SUBSTEP_TIMING_B64 = encode_patch("patch_substep_timing", _MILES_PATCHES)

_REPORTING_PATCH_COMMANDS = (
    f"echo {_PATCH_ROLLOUT_STATUS_B64} | base64 -d | python3",
    f"echo {_PATCH_ADVANTAGE_DIST_B64} | base64 -d | python3",
)

# Megatron-level torch_dist save fixes, shared with the slime image. Both no-op when
# their target source doesn't match, so they are safe for every miles image; the
# checkpoint-save one is skipped in the shell below when its target is absent
# entirely, since guarding inside the script would change the bytes the slime image
# already builds from.
_PATCH_DIST_CKPT_QUANTIZED_B64 = encode_patch(
    "patch_dist_ckpt_quantized", _MEGATRON_PATCHES
)
_PATCH_CHECKPOINT_SAVE_B64 = encode_patch("patch_checkpoint_save", _MEGATRON_PATCHES)
_MEGATRON_TORCH_STRATEGY_PY = (
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py"
)


_CONVERT_LOCK_DICT_NAME = "training-gym-convert-lock"
_CONVERT_LOCK_TTL_S = 900.0
_CONVERT_LOCK_REFRESH_S = 300.0
_CONVERT_TOKEN_TTL_S = 4 * _CONVERT_LOCK_TTL_S


def _convert_lock_dict() -> Any:
    return ModalDict.from_name(_CONVERT_LOCK_DICT_NAME, create_if_missing=True)


def _convert_lock_key(volume_name: str, save_path: str) -> str:
    return f"lock:{volume_name}:{save_path}"


def _sweep_convert_tokens(locks: Any, key: str) -> None:
    """Drop takeover tokens old enough that no live contender can act on one.

    A token has to outlive the takeover it settles, or a contender still holding the
    stale read it was written for could win a fresh one and pop the new owner's claim.
    Past that it is only litter, and nothing else reclaims it, so tokens are swept on
    release and on the next takeover for the same path.
    """
    prefix = f"{key}:takeover:"
    cutoff = time.time() - _CONVERT_TOKEN_TTL_S
    try:
        names = [str(name) for name in locks.keys() if str(name).startswith(prefix)]
        for name in names:
            token = locks.get(name)
            written_at = token.get("at") if isinstance(token, dict) else None
            if not isinstance(written_at, (int, float)) or written_at < cutoff:
                locks.pop(name, None)
    except Exception as exc:
        print(f"WARNING: could not sweep stale conversion tokens: {exc}")


def _acquire_convert_lock(run_id: str, volume_name: str, save_path: str) -> str:
    """Claim the right to convert into ``save_path``; return the current holder.

    Two launches of the same recipe share one ``ref_load``, so without this the
    loser's cleanup deletes the winner's still-metadata-less conversion output.

    ``Dict.put(..., skip_if_exists=True)`` is the atomic compare-and-set that decides
    the winner; a read-then-write claim would let two runs both observe an unheld lock
    and both proceed. The claim expires after ``_CONVERT_LOCK_TTL_S``, sized against the
    heartbeat interval so a holder that cannot run its release path frees the path in a
    TTL rather than a conversion's worth of time.

    Taking a stale claim over cannot be a delete followed by a put: two runs that both
    read the same stale holder would delete each other's fresh claim and both come away
    believing they own the path. Instead each contender first claims a token naming the
    stale holder it saw, which only one can win, and only that winner replaces the
    claim. Those tokens are swept by age, not on use — see ``_sweep_convert_tokens``.
    """
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    claim = {"run_id": run_id, "claimed_at": time.time()}
    if locks.put(key, claim, skip_if_exists=True):
        return run_id

    owner, claimed_at = "", None
    holder = locks.get(key)
    if isinstance(holder, dict):
        owner = str(holder.get("run_id") or "")
        claimed_at = holder.get("claimed_at")
        if owner == run_id:
            return run_id
        if (
            owner
            and isinstance(claimed_at, (int, float))
            and time.time() - claimed_at < _CONVERT_LOCK_TTL_S
        ):
            return owner

    takeover_key = f"{key}:takeover:{owner}:{claimed_at}"
    token = {"run_id": run_id, "at": time.time()}
    if not locks.put(takeover_key, token, skip_if_exists=True):
        winner = locks.get(takeover_key)
        if isinstance(winner, dict) and winner.get("run_id"):
            return str(winner["run_id"])
        return owner or run_id

    _sweep_convert_tokens(locks, key)
    try:
        locks.pop(key)
    except KeyError:
        pass
    if locks.put(key, claim, skip_if_exists=True):
        return run_id
    holder = locks.get(key)
    if isinstance(holder, dict) and holder.get("run_id"):
        return str(holder["run_id"])
    return run_id


def _refresh_convert_lock(run_id: str, volume_name: str, save_path: str) -> None:
    """Extend an existing claim of this run's; never create one.

    A refresh that wrote unconditionally would resurrect the claim when a tick
    straddles the release, leaving it held with nobody to give it back.
    """
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    holder = locks.get(key)
    if not isinstance(holder, dict) or str(holder.get("run_id") or "") != run_id:
        return
    locks[key] = {"run_id": run_id, "claimed_at": time.time()}


@contextlib.contextmanager
def _convert_lock_heartbeat(run_id: str, volume_name: str, save_path: str):
    """Keep this run's claim fresh for as long as the body runs.

    Refreshing on elapsed time rather than on work done: a torch_dist conversion
    writes only about one file per rank, so any progress-based cadence never fires,
    and the claim would lapse mid-conversion and let a later launch delete the output
    being written.
    """
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(_CONVERT_LOCK_REFRESH_S):
            try:
                _refresh_convert_lock(run_id, volume_name, save_path)
            except Exception as exc:
                print(f"WARNING: could not refresh conversion claim: {exc}")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5.0)


def _release_convert_lock(run_id: str, volume_name: str, save_path: str) -> None:
    """Drop the conversion claim, but only if this run still holds it."""
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    holder = locks.get(key)
    if isinstance(holder, dict) and str(holder.get("run_id") or "") != run_id:
        return
    try:
        locks.pop(key)
    except KeyError:
        pass
    _sweep_convert_tokens(locks, key)


def _is_resumable_checkpoint(path: str) -> bool:
    """Whether ``path`` holds a training save that can be resumed from.

    Looser than ``_is_complete_torch_dist_checkpoint`` because miles writes more than
    one save shape: a LoRA run stores ``adapter/`` with per-rank ``.pt`` files and no
    ``.metadata``/``common.pt``/``.distcp`` at all, so the conversion predicate would
    report every adapter checkpoint as absent and silently restart from ``ref_load``.
    Only the crashed-torch_dist signature is rejected — ``.distcp`` shards present but
    the ``.metadata`` that is written last missing.
    """
    try:
        names = os.listdir(path)
    except OSError:
        return False
    if not names:
        return False
    if any(name.endswith(".distcp") for name in names):
        return ".metadata" in names
    return True


def _unresumable_save_dirs(save_root: str) -> list[str]:
    """Save directories that exist but cannot be resumed from."""
    try:
        names = os.listdir(save_root)
    except OSError:
        return []
    return sorted(
        name
        for name in names
        if (name == "release" or name.startswith("iter_"))
        and os.path.isdir(os.path.join(save_root, name))
        and not _is_resumable_checkpoint(os.path.join(save_root, name))
    )


def _is_complete_torch_dist_checkpoint(path: str) -> bool:
    """Whether ``path`` holds a *finished* torch_dist checkpoint.

    ``.metadata`` is written last, by ``save_state_dict_async_finalize``, so its
    presence is what separates a completed save from a crashed one. Without this
    check ``torch_dist_resume_checkpoint``'s ``iter_*`` scan accepts any directory
    (its default ``is_complete`` is ``os.path.isdir``), so a conversion that died
    mid-write is reported as a cache hit and silently skips re-conversion — which
    then feeds partial weights to training. A crashed conversion does leave
    ``common.pt`` and the ``.distcp`` shards behind, so those alone are not enough
    to tell the two apart.
    """
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return (
        ".metadata" in names
        and "common.pt" in names
        and any(name.endswith(".distcp") for name in names)
    )


def _build_miles_base_image(miles: MilesRecipe) -> Image:
    image = (
        Image.from_registry(miles.docker_image)
        .entrypoint([])
        .run_commands(
            f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3",
            f"echo {_PATCH_DIST_CKPT_QUANTIZED_B64} | base64 -d | python3",
            (
                f"if test -f {_MEGATRON_TORCH_STRATEGY_PY}; then "
                f"echo {_PATCH_CHECKPOINT_SAVE_B64} | base64 -d | python3; "
                f"else echo 'WARNING: {_MEGATRON_TORCH_STRATEGY_PY} not found, "
                "skipping checkpoint-save patch'; fi"
            ),
            *_REPORTING_PATCH_COMMANDS,
            f"echo {_PATCH_SUBSTEP_TIMING_B64} | base64 -d | python3",
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
    metric_env: dict[str, str],
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
    env_vars.update(metric_env)
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
            f"echo {_PATCH_SGLANG_ABORT_B64} | base64 -d | python3"
            " || echo 'WARNING: sglang abort patch did not apply to the"
            " local_miles checkout; transient router failures during rollout"
            " cleanup may crash the run'",
            *_REPORTING_PATCH_COMMANDS,
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
        metrics=miles.metrics,
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
        run_prepare_dataset(
            dataset,
            data_volume,
            MilesRecipe._resolve_data_path,
        )

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
        if has_torch_dist_checkpoint(
            save_path, is_complete=_is_complete_torch_dist_checkpoint
        ):
            print(
                f"Found existing torch_dist checkpoint at {save_path}; "
                "skipping conversion."
            )
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return None

        holder = _acquire_convert_lock(
            training_run_id, checkpoints_volume_name, save_path
        )
        if holder != training_run_id:
            raise RuntimeError(
                f"Run {holder} is already converting into {save_path}. Two runs of "
                "this recipe share one ref_load, so continuing would delete that "
                "run's in-flight conversion and interleave shards. Wait for it to "
                "finish and relaunch to pick up the cached checkpoint, or point this "
                "run at a different ref_load."
            )

        try:
            # Clear partial torch_dist writes from an earlier crash here — the
            # single-container step that decides to convert — so the conversion cannot mix
            # fresh shards with stale ones from a different parallelism. Only the
            # ``iter_*``/``release`` directories the converter itself writes, plus the
            # iteration tracker the resume scan prefers over them, are removed, and only
            # those failing the completeness check: ``ref_load`` is user-settable
            # and may hold a hand-placed checkpoint in a layout this predicate rejects.
            if os.path.isdir(save_path):
                stale = [
                    name
                    for name in sorted(os.listdir(save_path))
                    if (name == "release" or name.startswith("iter_"))
                    and os.path.isdir(os.path.join(save_path, name))
                    and not _is_complete_torch_dist_checkpoint(
                        os.path.join(save_path, name)
                    )
                ]
                tracker = "latest_checkpointed_iteration.txt"
                tracker_path = os.path.join(save_path, tracker)
                has_tracker = stale and os.path.isfile(tracker_path)
                if stale:
                    print(
                        f"Removing incomplete torch_dist checkpoint state at {save_path}: "
                        + ", ".join([*stale, *([tracker] if has_tracker else [])])
                    )
                    for name in stale:
                        shutil.rmtree(os.path.join(save_path, name), ignore_errors=True)
                    if has_tracker:
                        os.remove(tracker_path)
                    checkpoints_volume.commit()

            conversion_hf_checkpoint = (
                getattr(miles, "megatron_conversion_hf_checkpoint", None)
                or getattr(miles, "hf_checkpoint", "")
                or model.model_path
                or model.model_name
            )
            return resolve_checkpoint_ref(conversion_hf_checkpoint)
        except BaseException:
            _release_convert_lock(training_run_id, checkpoints_volume_name, save_path)
            raise

    @app.function(
        image=image,
        gpu=gpu_spec,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        secrets=proxy_auth_secrets() or None,
        ephemeral_disk=miles.convert_ephemeral_disk_mb,
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
        convert_script = spec.origin if spec is not None else None
        if not convert_script:
            raise RuntimeError(
                "modal_training_gym.frameworks.miles.modal_helpers.convert_hf_to_torch_dist not found"
            )

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
        if any(arg.startswith("--pipeline-model-parallel-size ") for arg in extra_args):
            env["CONVERT_KEEP_PP1"] = "1"
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"

        print(
            f"Conversion layout: nodes={num_nodes}, nproc_per_node={nproc_per_node}, "
            f"node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        if node_rank == 0:
            _refresh_convert_lock(training_run_id, checkpoints_volume_name, save_path)
        heartbeat = (
            _convert_lock_heartbeat(training_run_id, checkpoints_volume_name, save_path)
            if node_rank == 0
            else contextlib.nullcontext()
        )
        try:
            with heartbeat:
                subprocess.run(["bash", "-c", cmd], check=True, env=env)

                checkpoints_volume.commit()

                if node_rank == 0:
                    print(f"Saved Megatron torch_dist checkpoint to {save_path}")
                    # Fail loudly here rather than leaving a partial checkpoint for a later
                    # run to mistake for a cache hit.
                    if not has_torch_dist_checkpoint(
                        save_path, is_complete=_is_complete_torch_dist_checkpoint
                    ):
                        raise RuntimeError(
                            f"Conversion finished but {save_path} holds no complete "
                            "torch_dist checkpoint (missing .metadata)."
                        )
            if node_rank == 0:
                _release_convert_lock(
                    training_run_id, checkpoints_volume_name, save_path
                )
        except BaseException:
            if node_rank == 0:
                _release_convert_lock(
                    training_run_id, checkpoints_volume_name, save_path
                )
            raise

        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

    _multi_node = miles.total_nodes > 1

    train_secrets = [
        *(metric_secrets(miles.metrics) if miles.metrics is not None else []),
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

        metric_entity = ""
        metric_run_id = ""

        if cluster.is_head:
            metric_entity = preflight_metric(miles.metrics)

            print(f"Training run id: {training_run_id}")
            config_summary = {
                "model": {"model_name": model.model_name} if model else {},
                "recipe": {
                    "gpu_type": miles.gpu_type,
                    **serialize_recipe_params(
                        miles,
                        dataset=dataset,
                        model=model,
                    ),
                },
                "metrics": metric_metadata(
                    miles.metrics,
                    entity=metric_entity,
                    run_id=metric_run_id,
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
                metric_run_id,
                framework_status_token,
            ) = await init_training_run_record(
                training_run_id=training_run_id,
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_url,
                framework=Framework.MILES,
                initializing_status=MilesStatus.INITIALIZING,
                config_summary=config_summary,
                metric_cfg=miles.metrics,
                metric_entity=metric_entity,
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
            wrote_data = write_dataset_if_needed(
                dataset,
                MilesRecipe._resolve_data_path(dataset, "train"),
            )
            if wrote_data:
                await data_volume.commit.aio()

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
                if isinstance(miles.metrics, WandbConfig):
                    miles.metrics.key = wandb_key

            save_root = compute_save_root(
                miles.save,
                recipe_default_save_root=str(CHECKPOINTS_PATH).rstrip("/"),
                mounted_save_root=checkpoints_mount_path,
                training_run_id=training_run_id,
            )

            original_save = miles.save
            original_load = miles.load
            miles.save = save_root
            resume_checkpoint = torch_dist_resume_checkpoint(
                save_root, is_complete=_is_resumable_checkpoint
            )
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
            elif unresumable := _unresumable_save_dirs(save_root):
                print(
                    f"WARNING: {save_root} holds saves that cannot be resumed "
                    f"({', '.join(unresumable)}) — they carry torch_dist shards "
                    "without the .metadata written last, so they are interrupted "
                    "writes. Training restarts from ref_load and their progress is "
                    "discarded; Megatron follows latest_checkpointed_iteration.txt, "
                    "so resuming into one of these would load a partial save."
                )
            try:
                cmd = build_train_cmd(
                    miles,
                    MILES_ROOT,
                    model=model,
                    dataset=dataset,
                )
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

            runtime_env = build_ray_runtime_env(
                head_addr=cluster.head_addr,
                metric_env=metric_runtime_env(
                    miles.metrics,
                    run_id=metric_run_id,
                    entity=metric_entity,
                ),
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
                metric_cfg=miles.metrics,
                metric_entity=metric_entity,
                metric_run_id=metric_run_id,
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
