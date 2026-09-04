"""Framework-agnostic launcher helpers shared by the slime and miles apps.

The two launchers are structurally identical: they ship user callables into the
image, resolve checkpoint volumes, tag the Modal app, run the download/prepare
phases, initialize/finalize the ``TrainingRun`` record, and build the
``TrainResult``. That shared machinery lives here; each framework passes its own
status enum / recipe hooks so the behavior stays framework-specific where it
must.
"""

from __future__ import annotations

import base64
import inspect
import os
import secrets as _secrets
import tempfile
import textwrap
import time
from typing import Any, Callable

import cloudpickle
from modal import Image, Volume

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, modal_tag_value
from modal_training_gym.common.framework import (
    Framework,
    resolve_caller_module,
)
from modal_training_gym.common.modal_refs import register_modal_cloudpickle_reducers
from modal_training_gym.common.run import (
    TrainingRun,
    TrainingRunStatus,
    mark_training_attempt_finished,
    mark_training_attempt_started,
    metric_run_id_for_attempt,
    record_metric_attempt,
    run_scoped_save_root,
    set_checkpoint_location,
)
from modal_training_gym.common.checkpoint import require_within_volume_mount
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.metrics import MetricConfig, metric_metadata
from modal_training_gym.utils.metadata import MetadataStore, vol_put


def resolve_caller_context() -> tuple[Any, str | None]:
    """Register the caller module for cloudpickle-by-value and return
    ``(caller_module, caller_script)``. The script path lets ``ship_callable``
    tell a user's inline callable apart from one imported from a shipped file."""
    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file and os.path.isfile(mod_file):
            caller_script = os.path.abspath(mod_file)
    return caller_module, caller_script


def ship_callable(
    image: "Image",
    fn: Any,
    *,
    caller_script: str | None,
    fallback_name: str,
    set_path: Callable[[str], None],
) -> "Image":
    """Make a user-provided callable importable inside the remote container.

    Package-internal callables need no shipping but still get ``set_path``.
    A callable defined in its own module is added as a local file pointed at
    ``module.symbol``. Inline callables are cloudpickled into a tiny loader
    module. Returns the (possibly extended) image.
    """
    if fn is None:
        return image
    fn_mod = getattr(fn, "__module__", None) or ""
    if fn_mod.startswith("modal_training_gym"):
        set_path(f"{fn_mod}.{getattr(fn, '__name__', fallback_name)}")
        return image
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
        set_path(f"{fn_module_name}.{getattr(fn, '__name__', fallback_name)}")
        return image
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
    return image


def resolve_checkpoint_volumes(
    checkpoint: Any,
    *,
    volume_prefix: str,
    default_mount_path: str,
) -> tuple[str, str, "Volume"]:
    """Resolve the checkpoints volume name / mount path / Volume, honoring an
    optional ``CheckpointConfig`` override."""
    checkpoints_volume_name = (
        checkpoint.checkpoints_volume_name
        if checkpoint is not None and checkpoint.checkpoints_volume_name
        else f"{volume_prefix}-checkpoints"
    )
    checkpoints_mount_path = (
        checkpoint.checkpoints_mount_path.rstrip("/") or "/"
        if checkpoint is not None and checkpoint.checkpoints_mount_path
        else default_mount_path.rstrip("/")
    )
    checkpoints_volume = Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    return checkpoints_volume_name, checkpoints_mount_path, checkpoints_volume


def build_app_tags(
    *,
    framework: str,
    model: Any,
    recipe_app_tags: dict[str, str],
    metrics: "MetricConfig | None",
) -> dict[str, str]:
    """Build the Modal app tag dict for dashboard auto-discovery."""
    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": framework,
        "_modal_model_name": modal_tag_value(model.model_name),
        **recipe_app_tags,
    }
    if metrics is not None:
        tags["_modal_metric_provider"] = metrics.provider
        tags["_modal_metric_project"] = modal_tag_value(metrics.project)
        if metrics.group:
            tags["_modal_metric_group"] = modal_tag_value(metrics.group)
        if metrics.provider == "wandb":
            tags["_modal_wandb_project"] = modal_tag_value(metrics.project)
            if metrics.group:
                tags["_modal_wandb_group"] = modal_tag_value(metrics.group)
    return tags


def run_download_phase(
    *,
    training_run_id: str,
    phase: str,
    framework_status_url: str,
    framework_status_token: str,
    volumes: "tuple[Volume, ...]",
    download: Callable[[], None],
) -> None:
    """Report the download phase, reload the given volumes, run ``download``,
    then commit the volumes and flush the status reporter."""
    from modal_training_gym.common.status_reporter import (
        enqueue_framework_status,
        flush as flush_status_reporter,
    )

    if training_run_id:
        enqueue_framework_status(
            training_run_id,
            phase,
            url=framework_status_url or None,
            token=framework_status_token or None,
            is_active=True,
        )
    for volume in volumes:
        volume.reload()
    download()
    for volume in volumes:
        volume.commit()
    if training_run_id:
        flush_status_reporter(timeout_seconds=2.0)


def run_prepare_dataset(
    dataset: Any,
    data_volume: "Volume",
    resolve_data_paths: Callable[[Any], tuple[str, Any]],
) -> None:
    """Materialize the dataset onto the data volume, honoring ``always_prepare``
    and validating the prepared prompt/eval paths."""
    data_volume.reload()
    prompt_data, eval_paths = resolve_data_paths(dataset)
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


async def init_training_run_record(
    *,
    training_run_id: str,
    modal_app_id: str,
    modal_app_url: str,
    framework: "Framework",
    initializing_status: Any,
    config_summary: dict[str, Any],
    metric_cfg: "MetricConfig | None",
    metric_entity: str,
    framework_status_token: str,
    checkpoint_dir: str,
    checkpoints_volume_name: str,
    checkpoints_mount_path: str,
) -> tuple[Any, str, str]:
    """Create or resume the ``TrainingRun`` record for this attempt and persist
    the framework-status token. Returns
    ``(run_record, metric_run_id, framework_status_token)``.

    Reuses the record the local ``TrainConfig.train()`` driver creates before
    invoking download/convert (so those phases are visible in the dashboard);
    falls back to a fresh record when ``train()`` is invoked directly.
    """
    try:
        run_record = await TrainingRun.from_id(training_run_id, is_async=True)
        run_record.modal_app_id = modal_app_id
        run_record.modal_app_url = modal_app_url
        run_record.config = config_summary
        run_record.framework_status = initializing_status
    except KeyError:
        created_at = int(time.time())
        run_record = TrainingRun(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url,
            framework=framework,
            config=config_summary,
            framework_status=initializing_status,
            created_at=created_at,
            started_at=created_at,
        )
    set_checkpoint_location(
        run_record,
        checkpoint_dir=checkpoint_dir,
        checkpoints_volume_name=checkpoints_volume_name,
        checkpoints_mount_path=checkpoints_mount_path,
    )
    attempt_count = mark_training_attempt_started(
        run_record, started_at=int(time.time())
    )
    metric_run_id = ""
    if metric_cfg is not None:
        metric_run_id = metric_run_id_for_attempt(training_run_id, attempt_count)
        metric_data = metric_metadata(
            metric_cfg, entity=metric_entity, run_id=metric_run_id
        )
        run_record.config["metrics"] = metric_data
        record_metric_attempt(
            run_record,
            **metric_data,
            attempt_count=attempt_count,
        )
    if attempt_count > 1:
        print(
            f"WARNING: training run {training_run_id} is retrying after preemption "
            f"or interruption (attempt {attempt_count})."
        )
    if not framework_status_token:
        framework_status_token = _secrets.token_urlsafe(32)
    await run_record.save(is_async=True)
    await vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        training_run_id,
        {"token": framework_status_token},
        is_async=True,
    )
    print(f"TrainingRun recorded: {training_run_id}")
    return run_record, metric_run_id, framework_status_token


def compute_save_root(
    save: str | None,
    *,
    recipe_default_save_root: str,
    mounted_save_root: str,
    training_run_id: str,
) -> str:
    """Resolve the run-scoped checkpoint save root. A configured ``save`` equal
    to the recipe default is redirected to the mounted volume path so checkpoints
    land on the checkpoints Volume."""
    configured_save_root = str(save).rstrip("/") if save else mounted_save_root
    save_root = (
        mounted_save_root
        if configured_save_root == recipe_default_save_root
        else configured_save_root
    )
    return require_within_volume_mount(
        run_scoped_save_root(save_root, training_run_id),
        mounted_save_root,
    )[0]


def build_train_result(
    *,
    app_name: str,
    framework: "Framework",
    training_run_id: str,
    checkpoint_dir: str,
    model: Any,
    checkpoints_volume_name: str,
    checkpoints_mount_path: str,
    metric_cfg: "MetricConfig | None",
    metric_entity: str,
    metric_run_id: str,
    group_id: str | None,
) -> "TrainResult":
    """Construct a ``TrainResult``, filtering to the fields the dataclass
    actually accepts (older/newer TrainResult versions differ)."""
    result_kwargs = {
        "app_name": app_name,
        "framework": framework,
        "training_run_id": training_run_id,
        "checkpoint_dir": checkpoint_dir,
        "model_config": model,
        "checkpoints_volume_name": checkpoints_volume_name,
        "checkpoints_mount_path": checkpoints_mount_path,
        "metrics": metric_metadata(
            metric_cfg, entity=metric_entity, run_id=metric_run_id
        ),
        "group_id": group_id or "",
    }
    accepted_fields = set(inspect.signature(TrainResult).parameters)
    return TrainResult(
        **{k: v for k, v in result_kwargs.items() if k in accepted_fields}
    )


def mark_run_stopped(run_record: Any) -> None:
    """Mark the run STOPPED (e.g. KeyboardInterrupt)."""
    run_record.status = TrainingRunStatus.STOPPED
    mark_training_attempt_finished(
        run_record, status="stopped", ended_at=int(time.time())
    )


def mark_run_failed(run_record: Any, exc: BaseException) -> None:
    """Mark the run FAILED, preserving any more-specific error already set."""
    run_record.status = TrainingRunStatus.FAILED
    terminal_error = f"{type(exc).__name__}: {exc}"
    # Prefer a more specific message already set (e.g. the raw Ray driver
    # message from the is_success check) over the generic wrapper.
    run_record.error_message = run_record.error_message or terminal_error
    mark_training_attempt_finished(
        run_record, status="failed", ended_at=int(time.time())
    )


async def build_terminal_run_record(run_record: Any, training_run_id: str) -> Any:
    """Re-fetch the latest record from the volume and stamp terminal fields
    (status, ended/completed timestamps, duration, error) onto it so the caller
    can persist a consistent terminal record."""
    finished_at = int(time.time())
    try:
        latest_run_record = await TrainingRun.from_id(training_run_id, is_async=True)
    except Exception:
        latest_run_record = run_record

    latest_run_record.status = run_record.status
    latest_run_record.ended_at = finished_at
    # Propagate the terminal error onto the re-fetched record so the save
    # below persists it (the fresh fetch wouldn't carry it).
    if run_record.error_message:
        latest_run_record.error_message = run_record.error_message
    source_metadata = run_record.metadata or {}
    latest_metadata = dict(latest_run_record.metadata or {})
    for key in ("last_attempt_status", "last_attempt_ended_at", "terminal_reason"):
        if key in source_metadata:
            latest_metadata[key] = source_metadata[key]
    if latest_metadata:
        latest_run_record.metadata = latest_metadata
    if latest_run_record.completed_at is None:
        latest_run_record.completed_at = finished_at
    latest_run_record.duration_seconds = max(
        0, finished_at - latest_run_record.started_at
    )
    return latest_run_record
