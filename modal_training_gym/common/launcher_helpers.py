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
from pathlib import PurePosixPath
from typing import Any, Callable

import cloudpickle
from modal import Image, Volume

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, modal_tag_value
from modal_training_gym.common.attempts import (
    validate_attempt_id,
    validate_run_contract_sha256,
)
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
    record_wandb_attempt,
    run_scoped_save_root,
    wandb_run_id_for_attempt,
)
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_put


TRAIN_RESULT_ACCEPTANCE_KEY = "training_gym_acceptance"
TRAIN_RESULT_ACCEPTANCE_SCHEMA_VERSION = 1


class AcceptedTrainResultError(RuntimeError):
    """An existing result was present but could not authenticate acceptance."""


def bind_accepted_train_result(
    result: TrainResult,
    *,
    run_contract_sha256: str,
    accepted_attempt_id: str,
) -> TrainResult:
    """Bind a TrainResult to the immutable attempt it accepts.

    The bound result is published only after checkpoint-volume commit.  It is
    therefore both the returned handle and the fail-closed marker a clustered
    wrapper may use to distinguish intentional post-success Ray shutdown from
    a causal head failure.
    """

    contract = validate_run_contract_sha256(run_contract_sha256)
    attempt_id = validate_attempt_id(accepted_attempt_id)
    checkpoint_dir = str(result.checkpoint_dir or "")
    checkpoint_parts = PurePosixPath(checkpoint_dir).parts
    if len(checkpoint_parts) < 2 or checkpoint_parts[-2:] != (
        "attempts",
        attempt_id,
    ):
        raise ValueError(
            "accepted TrainResult checkpoint_dir is not owned by its attempt"
        )
    receipt = {
        "schema_version": TRAIN_RESULT_ACCEPTANCE_SCHEMA_VERSION,
        "run_contract_sha256": contract,
        "accepted_attempt_id": attempt_id,
        "checkpoint_dir": checkpoint_dir,
    }
    extra = dict(result.extra or {})
    existing = extra.get(TRAIN_RESULT_ACCEPTANCE_KEY)
    if existing is not None and existing != receipt:
        raise ValueError("TrainResult already has a conflicting acceptance binding")
    extra[TRAIN_RESULT_ACCEPTANCE_KEY] = receipt
    result.extra = extra
    return result


async def load_accepted_train_result(
    training_run_id: str,
    *,
    expected_framework: Framework,
    expected_run_contract_sha256: str,
) -> TrainResult | None:
    """Load and authenticate the published result for idempotent re-entry.

    Absence means the logical run has not been accepted.  Presence with a
    missing or conflicting binding is corruption and fails closed rather than
    allowing another attempt to overwrite the existing result.
    """

    expected_contract = validate_run_contract_sha256(expected_run_contract_sha256)
    try:
        payload = await vol_get(
            MetadataStore.TRAIN_RESULTS,
            training_run_id,
            is_async=True,
        )
    except KeyError:
        return None
    try:
        result = TrainResult(**TrainResult._parse_model_config(payload))
    except Exception as exc:
        raise AcceptedTrainResultError("existing TrainResult is malformed") from exc
    framework_value = (
        result.framework.value
        if isinstance(result.framework, Framework)
        else str(result.framework)
    )
    if result.training_run_id != training_run_id:
        raise AcceptedTrainResultError(
            "existing TrainResult belongs to another logical run"
        )
    if framework_value != expected_framework.value:
        raise AcceptedTrainResultError(
            "existing TrainResult belongs to another framework"
        )
    extra = result.extra if isinstance(result.extra, dict) else {}
    receipt = extra.get(TRAIN_RESULT_ACCEPTANCE_KEY)
    if not isinstance(receipt, dict):
        raise AcceptedTrainResultError(
            "existing TrainResult lacks an acceptance binding"
        )
    if receipt.get("schema_version") != TRAIN_RESULT_ACCEPTANCE_SCHEMA_VERSION:
        raise AcceptedTrainResultError(
            "existing TrainResult acceptance schema is unsupported"
        )
    try:
        accepted_attempt_id = validate_attempt_id(
            str(receipt.get("accepted_attempt_id") or "")
        )
        observed_contract = validate_run_contract_sha256(
            str(receipt.get("run_contract_sha256") or "")
        )
    except ValueError as exc:
        raise AcceptedTrainResultError(
            "existing TrainResult acceptance binding is malformed"
        ) from exc
    if observed_contract != expected_contract:
        raise AcceptedTrainResultError(
            "existing TrainResult belongs to another run contract"
        )
    checkpoint_dir = str(result.checkpoint_dir or "")
    if receipt.get("checkpoint_dir") != checkpoint_dir:
        raise AcceptedTrainResultError(
            "existing TrainResult checkpoint binding disagrees"
        )
    checkpoint_parts = PurePosixPath(checkpoint_dir).parts
    if len(checkpoint_parts) < 2 or checkpoint_parts[-2:] != (
        "attempts",
        accepted_attempt_id,
    ):
        raise AcceptedTrainResultError(
            "existing TrainResult checkpoint is outside its attempt"
        )
    return result


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

    Package-internal callables need nothing (they ship with the package). A
    callable defined in its own module is added as a local file and the slime/
    miles arg is pointed at ``module.symbol``. Anything defined inline (e.g. in a
    notebook or the caller script) is cloudpickled into a tiny loader module.
    Returns the (possibly extended) image.
    """
    if fn is None:
        return image
    fn_mod = getattr(fn, "__module__", None) or ""
    if fn_mod.startswith("modal_training_gym"):
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
        # Point the slime arg at the shipped module's symbol. Without this the
        # file is shipped but the path stays unset, so a custom_rm_function
        # defined outside the entrypoint silently falls back to rule-based RM.
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
    wandb: "WandbConfig | None",
) -> dict[str, str]:
    """Build the Modal app tag dict for dashboard auto-discovery."""
    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": framework,
        "_modal_model_name": modal_tag_value(model.model_name),
        **recipe_app_tags,
    }
    if wandb is not None:
        tags["_modal_wandb_project"] = modal_tag_value(wandb.project)
        if wandb.group:
            tags["_modal_wandb_group"] = modal_tag_value(wandb.group)
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
    wandb_cfg: "WandbConfig | None",
    wandb_entity: str,
    framework_status_token: str,
    max_attempts: int | None = None,
) -> tuple[Any, str, str]:
    """Create or resume the ``TrainingRun`` record for this attempt and persist
    the framework-status token. Returns
    ``(run_record, wandb_run_id, framework_status_token)``.

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
    if max_attempts is not None:
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        try:
            prior_attempt_count = int(
                (run_record.metadata or {}).get("attempt_count") or 0
            )
        except (TypeError, ValueError):
            prior_attempt_count = 0
        if prior_attempt_count >= max_attempts:
            raise RuntimeError(
                "logical attempt limit exceeded before attempt creation or Ray "
                "bootstrap: "
                f"next_attempt={prior_attempt_count + 1} max_attempts={max_attempts}"
            )
    attempt_count = mark_training_attempt_started(
        run_record, started_at=int(time.time())
    )
    wandb_run_id = ""
    if wandb_cfg is not None:
        wandb_run_id = wandb_run_id_for_attempt(training_run_id, attempt_count)
        run_record.config["wandb"]["run_id"] = wandb_run_id
        record_wandb_attempt(
            run_record,
            entity=wandb_entity,
            project=wandb_cfg.project,
            group=wandb_cfg.group,
            run_id=wandb_run_id,
            attempt_count=attempt_count,
        )
    if attempt_count > 1:
        print(
            f"WARNING: training run {training_run_id} is retrying after preemption "
            f"or interruption (attempt {attempt_count})."
        )
    if not framework_status_token:
        framework_status_token = _secrets.token_urlsafe(32)
    await run_record.save(is_async=True, event_kind="started")
    await vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        training_run_id,
        {"token": framework_status_token},
        is_async=True,
    )
    print(f"TrainingRun recorded: {training_run_id}")
    return run_record, wandb_run_id, framework_status_token


def compute_save_root(
    save: str | None,
    *,
    recipe_default_save_root: str,
    mounted_save_root: str,
    training_run_id: str,
) -> str:
    """Resolve the run-scoped checkpoint save root and ensure it exists. A
    configured ``save`` equal to the recipe default is redirected to the mounted
    volume path so checkpoints land on the checkpoints Volume."""
    configured_save_root = str(save).rstrip("/") if save else mounted_save_root
    save_root = run_scoped_save_root(
        mounted_save_root
        if configured_save_root == recipe_default_save_root
        else configured_save_root,
        training_run_id,
    )
    os.makedirs(save_root, exist_ok=True)
    return save_root


def build_train_result(
    *,
    app_name: str,
    framework: "Framework",
    training_run_id: str,
    checkpoint_dir: str,
    model: Any,
    checkpoints_volume_name: str,
    checkpoints_mount_path: str,
    wandb_cfg: "WandbConfig | None",
    wandb_entity: str,
    wandb_run_id: str,
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
        "wandb_project": wandb_cfg.project if wandb_cfg else "",
        "wandb_entity": wandb_entity,
        "wandb_training_run_id": wandb_run_id,
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


def _has_ray_failure_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    attempt_id: str | None,
    attempt_count: int | None,
) -> bool:
    """Return whether this logical attempt already owns a Ray snapshot."""
    if attempt_id:
        return any(item.get("attempt_id") == attempt_id for item in diagnostics)
    if attempt_count is not None:
        return any(item.get("attempt_count") == attempt_count for item in diagnostics)
    return False


def record_ray_failure_diagnostic(
    run_record: Any,
    snapshot: dict[str, Any],
    *,
    attempt_id: str | None = None,
    attempt_count: int | None = None,
    ray_job_id: str | None = None,
    ray_job_status: str,
    failure_stage: str,
) -> bool:
    """Attach at most one bounded Ray snapshot to a training attempt.

    The return value says whether a new entry was appended. Keeping one entry
    per immutable attempt avoids recording a second snapshot when the normal
    failed-job path records diagnostics and then raises into the launcher's
    terminal exception handler.
    """
    metadata = dict(run_record.metadata or {})
    if attempt_id is None:
        attempt_id = str(metadata.get("active_attempt_id") or "") or None
    if attempt_count is None:
        try:
            attempt_count = int(metadata.get("attempt_count"))
        except (TypeError, ValueError):
            attempt_count = None
    if attempt_count is not None and attempt_count < 1:
        attempt_count = None

    raw_diagnostics = metadata.get("ray_failure_diagnostics")
    diagnostics = (
        [dict(item) for item in raw_diagnostics[-8:] if isinstance(item, dict)]
        if isinstance(raw_diagnostics, list)
        else []
    )
    if _has_ray_failure_diagnostic(
        diagnostics,
        attempt_id=attempt_id,
        attempt_count=attempt_count,
    ):
        return False

    diagnostic: dict[str, Any] = {
        "ray_job_status": ray_job_status,
        "failure_stage": failure_stage,
        "snapshot": dict(snapshot),
    }
    if attempt_id:
        diagnostic["attempt_id"] = attempt_id
    if attempt_count is not None:
        diagnostic["attempt_count"] = attempt_count
    if ray_job_id:
        diagnostic["ray_job_id"] = ray_job_id

    diagnostics.append(diagnostic)
    metadata["ray_failure_diagnostics"] = diagnostics[-8:]
    run_record.metadata = metadata
    return True


def capture_and_record_ray_failure_diagnostic(
    run_record: Any,
    capture_snapshot: Callable[[], dict[str, Any]],
    *,
    attempt_id: str | None = None,
    attempt_count: int | None = None,
    ray_job_id: str | None = None,
    ray_job_status: str,
    failure_stage: str,
) -> bool:
    """Best-effort fallback capture that cannot replace the launch failure."""
    metadata = dict(run_record.metadata or {})
    raw_diagnostics = metadata.get("ray_failure_diagnostics")
    diagnostics = (
        [item for item in raw_diagnostics[-8:] if isinstance(item, dict)]
        if isinstance(raw_diagnostics, list)
        else []
    )
    resolved_attempt_id = (
        attempt_id or str(metadata.get("active_attempt_id") or "") or None
    )
    resolved_attempt_count = attempt_count
    if resolved_attempt_count is None:
        try:
            resolved_attempt_count = int(metadata.get("attempt_count"))
        except (TypeError, ValueError):
            resolved_attempt_count = None
    if _has_ray_failure_diagnostic(
        diagnostics,
        attempt_id=resolved_attempt_id,
        attempt_count=resolved_attempt_count,
    ):
        return False

    try:
        snapshot = capture_snapshot()
    except Exception as exc:  # noqa: BLE001 - never mask the causal failure
        print(
            "Failed to capture Ray diagnostics while preserving the original "
            f"launcher exception: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False

    return record_ray_failure_diagnostic(
        run_record,
        snapshot,
        attempt_id=resolved_attempt_id,
        attempt_count=resolved_attempt_count,
        ray_job_id=ray_job_id,
        ray_job_status=ray_job_status,
        failure_stage=failure_stage,
    )


def record_last_committed_boundary_snapshot(
    run_record: Any,
    boundary: dict[str, Any] | None,
    *,
    active_attempt_id: str | None = None,
    captured_at: int | None = None,
) -> dict[str, Any]:
    """Record a fixed-size, metadata-only view of durable training progress.

    A generated-ahead batch is intentionally separate from
    ``trained_through_rollout_id`` so an in-flight "Generating N/N" status
    cannot be interpreted as evidence that rollout N was optimized.
    """
    metadata = dict(run_record.metadata or {})
    resolved_active_attempt_id = (
        active_attempt_id or str(metadata.get("active_attempt_id") or "") or None
    )
    pending = (
        boundary.get("pending_rollout")
        if isinstance(boundary, dict)
        and isinstance(boundary.get("pending_rollout"), dict)
        else None
    )
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": int(time.time()) if captured_at is None else int(captured_at),
        "metadata_only": True,
        "found": boundary is not None,
        "active_attempt_id": resolved_active_attempt_id,
        "committed_attempt_id": (
            str(boundary.get("attempt_id") or "") or None
            if boundary is not None
            else None
        ),
        "scientific_commit_id": (
            str(boundary.get("scientific_commit_id") or "") or None
            if boundary is not None
            else None
        ),
        "parent_commit_id": (
            str(boundary.get("parent_commit_id") or "") or None
            if boundary is not None
            else None
        ),
        "trained_through_rollout_id": (
            int(boundary["rollout_id"]) if boundary is not None else None
        ),
        "checkpoint_iteration": (
            int(boundary["checkpoint_iteration"]) if boundary is not None else None
        ),
        "pending_generated_rollout_id": (
            int(pending["rollout_id"]) if pending is not None else None
        ),
        "terminal": bool(boundary.get("terminal")) if boundary is not None else False,
        "boundary_sha256": (
            str(boundary.get("boundary_sha256") or "") or None
            if boundary is not None
            else None
        ),
    }
    metadata["last_committed_boundary"] = snapshot
    run_record.metadata = metadata
    return snapshot


def record_attempt_failure(
    run_record: Any,
    error_message: str,
    *,
    attempt_id: str | None = None,
    attempt_count: int | None = None,
    recorded_at: int | None = None,
) -> str:
    """Record an attempt failure without replacing the run's root cause.

    Modal reports the exception from the final retry. That exception is often
    secondary (for example, a fail-closed resume guard rejecting a retry) and
    can otherwise overwrite the exception that killed the first attempt.

    ``error_message`` is recorded against the current attempt while
    ``metadata["primary_failure"]`` and the top-level ``error_message`` retain
    the first observed failure. The returned string is that primary message so
    callers can raise it as the terminal exception Modal surfaces.
    """
    message = str(error_message).strip() or "Training attempt failed."
    metadata = dict(run_record.metadata or {})

    if attempt_id is None:
        for key in ("active_attempt_id", "current_attempt_id", "attempt_id"):
            value = metadata.get(key)
            if value:
                attempt_id = str(value)
                break

    if attempt_count is None:
        try:
            attempt_count = int(metadata.get("attempt_count"))
        except (TypeError, ValueError):
            attempt_count = None
    if attempt_count is not None and attempt_count < 1:
        attempt_count = None

    failure: dict[str, Any] = {
        "message": message,
        "recorded_at": int(time.time()) if recorded_at is None else int(recorded_at),
    }
    if attempt_id:
        failure["attempt_id"] = attempt_id
    if attempt_count is not None:
        failure["attempt_count"] = attempt_count

    raw_failures = metadata.get("attempt_failures")
    failures = (
        [dict(item) for item in raw_failures if isinstance(item, dict)]
        if isinstance(raw_failures, list)
        else []
    )

    def _same_attempt(item: dict[str, Any]) -> bool:
        if attempt_id:
            return item.get("attempt_id") == attempt_id
        if attempt_count is not None:
            return item.get("attempt_count") == attempt_count
        return False

    matching_index = next(
        (index for index, item in enumerate(failures) if _same_attempt(item)),
        None,
    )
    if matching_index is None:
        failures.append(failure)
    else:
        failures[matching_index] = failure
    metadata["attempt_failures"] = failures

    raw_primary = metadata.get("primary_failure")
    primary = dict(raw_primary) if isinstance(raw_primary, dict) else None
    primary_message = (
        str(primary.get("message", "")).strip() if primary is not None else ""
    )
    if not primary_message:
        existing_error = str(run_record.error_message or "").strip()
        if existing_error:
            primary_message = existing_error
            primary = {
                "message": existing_error,
                # For records created before per-attempt failure provenance,
                # the original attempt identity is unknown.
                "recorded_at": failure["recorded_at"],
            }
        else:
            primary_message = message
            primary = dict(failure)
        metadata["primary_failure"] = primary

    run_record.metadata = metadata
    run_record.error_message = primary_message
    return primary_message


def mark_run_failed(run_record: Any, exc: BaseException) -> str:
    """Mark the run FAILED and return the preserved causal failure message."""
    run_record.status = TrainingRunStatus.FAILED
    terminal_error = f"{type(exc).__name__}: {exc}"
    metadata = dict(run_record.metadata or {})
    active_attempt_id = str(metadata.get("active_attempt_id") or "")
    raw_failures = metadata.get("attempt_failures")
    already_recorded = isinstance(raw_failures, list) and any(
        isinstance(item, dict)
        and active_attempt_id
        and item.get("attempt_id") == active_attempt_id
        for item in raw_failures
    )
    if not already_recorded:
        primary_error = record_attempt_failure(run_record, terminal_error)
    else:
        raw_primary = metadata.get("primary_failure")
        primary_error = (
            str(raw_primary.get("message") or "").strip()
            if isinstance(raw_primary, dict)
            else ""
        )
        if not primary_error:
            primary_error = str(run_record.error_message or "").strip()
        if not primary_error:
            primary_error = terminal_error
        run_record.error_message = primary_error
    mark_training_attempt_finished(
        run_record, status="failed", ended_at=int(time.time())
    )
    return primary_error


async def build_terminal_run_record(run_record: Any, training_run_id: str) -> Any:
    """Re-fetch the latest record from the volume and stamp terminal fields
    (status, ended/completed timestamps, duration, error) onto it so the caller
    can persist a consistent terminal record."""
    finished_at = int(time.time())
    try:
        latest_run_record = await TrainingRun.from_id(training_run_id, is_async=True)
    except Exception:
        latest_run_record = run_record

    local_metadata = dict(run_record.metadata or {})
    latest_metadata = dict(latest_run_record.metadata or {})
    local_attempt_id = str(local_metadata.get("active_attempt_id") or "").strip()
    latest_attempt_id = str(latest_metadata.get("active_attempt_id") or "").strip()
    if local_attempt_id and latest_attempt_id and local_attempt_id != latest_attempt_id:
        print(
            "Ignoring stale terminal finalization from attempt "
            f"{local_attempt_id}; active attempt is {latest_attempt_id}.",
            flush=True,
        )
        return latest_run_record

    latest_run_record.status = run_record.status
    latest_run_record.ended_at = finished_at
    # Framework-status reports can update the persisted record while the
    # launcher is running, hence the re-fetch above.  Conversely, terminal
    # attempt provenance is accumulated only on the launcher's in-memory
    # record.  Merge those explicit fields instead of replacing all metadata
    # (which would roll back newer framework progress).
    terminal_metadata_keys = {
        "attempts",
        "attempt_count",
        "active_attempt_id",
        "active_attempt_root",
        "attempt_mode",
        "event_journal_contract",
        "event_journal_enabled",
        "attempt_failures",
        "finalized_from_terminal_parent",
        "last_attempt_ended_at",
        "last_attempt_started_at",
        "last_attempt_status",
        "last_committed_boundary",
        "logical_save_root",
        "max_retries",
        "max_attempts",
        "primary_failure",
        "ray_failure_diagnostics",
        "resume_boundary",
        "wandb_accepted_run_id",
        "wandb_attempts",
        "wandb_latest_run_id",
    }
    for key in terminal_metadata_keys:
        if key in local_metadata:
            latest_metadata[key] = local_metadata[key]
    latest_run_record.metadata = latest_metadata
    # Propagate the terminal error onto the re-fetched record so the save
    # below persists it (the fresh fetch wouldn't carry it).
    latest_run_record.error_message = run_record.error_message
    if latest_run_record.completed_at is None:
        latest_run_record.completed_at = finished_at
    latest_run_record.duration_seconds = max(
        0, finished_at - latest_run_record.started_at
    )
    return latest_run_record


async def load_preserved_primary_failure(training_run_id: str) -> str:
    """Return the journaled first failure, including during retry setup.

    A Modal retry can fail before it constructs its in-memory ``run_record``.
    Reading through ``TrainingRun.from_id`` materializes the append-only event
    journal, so a setup failure cannot hide the earlier causal exception.
    """
    try:
        run_record = await TrainingRun.from_id(training_run_id, is_async=True)
    except Exception:
        return ""
    metadata = dict(run_record.metadata or {})
    primary = metadata.get("primary_failure")
    if isinstance(primary, dict):
        message = str(primary.get("message") or "").strip()
        if message:
            return message
    return str(run_record.error_message or "").strip()


async def record_setup_failure(
    training_run_id: str,
    exc: BaseException,
) -> str:
    """Terminalize an attempt that failed before the launcher's main guard.

    ``init_training_run_record`` publishes the immutable start before writing
    the status token. If that or any later setup operation fails, recover the
    materialized attempt here, journal its failure, and preserve any older
    primary cause.
    """
    try:
        run_record = await TrainingRun.from_id(training_run_id, is_async=True)
    except Exception:
        return ""
    if run_record.status == TrainingRunStatus.RUNNING:
        primary = mark_run_failed(run_record, exc)
        metadata = dict(run_record.metadata or {})
        has_attempt = bool(metadata.get("active_attempt_id")) and bool(
            metadata.get("attempt_count")
        )
        try:
            if has_attempt:
                await run_record.save(is_async=True, event_kind="failure")
            else:
                await run_record.save_cache(is_async=True)
        except Exception as save_exc:  # noqa: BLE001
            print(
                "Failed to persist setup-failure reporting while preserving "
                f"the original exception: {type(save_exc).__name__}: {save_exc}",
                flush=True,
            )
        return primary
    return await load_preserved_primary_failure(training_run_id)
