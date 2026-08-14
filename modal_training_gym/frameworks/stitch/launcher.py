"""Factory that builds a Modal app for a disaggregated miles run via stitch.

Same shape as the colocated launchers (``build_slime_app`` / ``build_miles_app``):
:func:`build_stitch_app` returns a ``modal.App`` with ``download``,
``prepare_dataset``, ``prepare_checkpoints``, and ``train``, so a run is one
call::

    TrainConfig(model=..., dataset=..., recipe=StitchRecipe(...)).train()

What differs is what the app contains: rollouts are served by a Modal Flash pool
of SGLang replicas (the ``Server`` class, brought up with the app) that self-sync
to sparse weight deltas the clustered ``train`` function publishes to a Modal
Volume bulletin board. The trainer reaches the pool through its Flash gateway,
resolved from the in-app class handle — so the single ``train()`` call works in an
ephemeral run, with no separate ``modal deploy`` step.

The app is still deployable (``modal deploy``) when a pool should outlive a single
run; only then can the publish hook wake replicas by app name — otherwise they
pick the new pointer up on their next reconcile poll.

This packages the stitch ``miles_disagg`` cookbook (``cookbook/miles_disagg``)
around a training-gym ``StitchRecipe`` + ``ModelConfig`` + ``DatasetConfig``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast

import cloudpickle
import modal
import modal.experimental

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS, modal_tag_value
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.framework import Framework, resolve_caller_module
from modal_training_gym.common.modal_refs import register_modal_cloudpickle_reducers
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.ray_cluster import (
    RAY_PORT,
    clustered_if,
    start_ray_head,
    start_ray_worker,
)
from modal_training_gym.common.run import (
    TrainingRun,
    TrainingRunStatus,
    record_wandb_attempt,
    wandb_run_id_for_attempt,
)
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.common.wandb import preflight_wandb
from modal_training_gym.frameworks.stitch import serving_image
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    MEGATRON_PATH,
    MILES_ROOT,
    stitch_install_commands,
)
from modal_training_gym.train_recipes.stitch_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    HOOK_CONFIG_FIELDS,
    YAML_CONFIG_FIELDS,
    StitchRecipe,
    fields_to_argv,
)
from modal_training_gym.train_recipes.stitch_recipe.train import StitchTrainConfig

MINUTES = 60
# The cookbook's own ports (``cookbook.common.constants``), repeated because the
# ``@app.cls`` decorators are evaluated client-side, where it isn't importable.
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
# A replica boots with the app, i.e. while prepare_checkpoints may still be
# building the baseline it serves. Waiting it out has to fit inside the startup
# budget, so a longer conversion means the replica exits and Modal reboots it
# into another wait rather than the engine failing on a missing checkpoint.
BASELINE_WAIT_TIMEOUT = SERVER_STARTUP_TIMEOUT - 2 * MINUTES
BASELINE_POLL_SECONDS = 30
# Ephemeral host-local full HF checkpoint the sidecar patches in place per delta.
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
# What the engine needs to seed a delta from the base checkpoint: weights plus the
# config/tokenizer files beside them. Restricting the resolve to these keeps it
# from failing on a cache SGLang populated itself (it fetches no README/figures,
# and a snapshot missing *any* file is "incomplete").
_CHECKPOINT_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
]


class _MilesArgs:
    """Runtime carrier for the miles args the trainer runs with.

    The recipe + model + dataset resolve to a plain field dict
    (:meth:`StitchRecipe.to_payload`); ``train`` rebuilds this carrier, injects
    the per-run fields (rollout endpoint, bulletin dir, custom config), then
    materializes YAML configs and builds the ``train.py`` command from
    :meth:`cli_args`.
    """

    _CONTROL = {"async_mode", "miles_model_script"}

    # Per-run fields the trainer injects (the rest come from the field dict).
    rollout_endpoint_url: str
    update_weight_disk_dir: str
    custom_config_path: dict | str
    te_precision_config_file: dict | str | None

    def __init__(
        self, fields: dict, *, async_mode: bool, miles_model_script: str
    ) -> None:
        for key, val in fields.items():
            setattr(self, key, val)
        self.async_mode = async_mode
        self.miles_model_script = miles_model_script

    def cli_args(self) -> list[str]:
        fields = {k: v for k, v in vars(self).items() if k not in self._CONTROL}
        return fields_to_argv(fields)


def _local_checkpoint(model_name: str, volume_name: str) -> str:
    """The served baseline as a local directory.

    A repo id is materialized from the HF cache volume. A prepared (quantized)
    baseline is already a path, but the pool comes up with the app, so it can be
    waited on: ``prepare_checkpoints`` builds into a ``.partial`` sibling and
    renames, so the path appearing means the baseline is complete.
    """
    from huggingface_hub import snapshot_download

    if not model_name.startswith("/"):
        return snapshot_download(model_name, allow_patterns=_CHECKPOINT_PATTERNS)

    volume = modal.Volume.from_name(volume_name)
    deadline = time.monotonic() + BASELINE_WAIT_TIMEOUT
    while not os.path.isdir(model_name):
        if time.monotonic() > deadline:
            raise RuntimeError(f"served baseline {model_name} never appeared")
        print(f"waiting for the served baseline at {model_name}...")
        time.sleep(BASELINE_POLL_SECONDS)
        volume.reload()
    return model_name


def _stitch_trainer_image(train: StitchTrainConfig) -> modal.Image:
    """The miles trainer image. The rollout pool serves on a different image
    (:func:`serving_image.build_serving_image`): it installs no trainer package, and
    it needs the SGLang fork that exposes ``/stage_weight_update``.

    The base image bakes Megatron-LM (native ``--fp4-format`` NVFP4) plus
    TransformerEngine; the pinned miles fork — which speaks the bulletin protocol
    — is cloned over it.
    """
    image = (
        modal.Image.from_registry(train.docker_image)
        .entrypoint([])
        # TransformerEngine 2.17 declares this, but the dated miles image installs
        # its TE wheels with --no-deps.
        .pip_install("onnxscript==0.7.1")
        # RDMA/EFA userspace, so a multi-node NCCL binds EFA rather than TCP.
        .apt_install(
            "libibverbs-dev", "libibverbs1", "libhwloc-dev", "libnl-route-3-200"
        )
        # The base image bakes an HF cache; drop it so the mounted cache volume
        # at the same path isn't shadowed.
        .run_commands(f"rm -rf {HF_CACHE_PATH}")
        .run_commands(
            f"rm -rf {MILES_ROOT}"
            f" && git clone {train.miles_repo_url} {MILES_ROOT}"
            f" && cd {MILES_ROOT}"
            f" && git fetch origin {train.miles_repo_ref} && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {MILES_ROOT}"
        )
        .pip_install(
            "httpx",  # stitch's pool client (wake fan-out)
            # miles is installed --no-deps, but the trainer-side delta ENCODER needs
            # these (zstd compress + xxh3/blake3 checksums).
            "zstandard",
            "xxhash",
            "blake3",
        )
        .run_commands(*stitch_install_commands())
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
    )
    if train.image_run_commands:
        image = image.run_commands(*train.image_run_commands)
    if train.image_env:
        image = image.env(train.image_env)
    if train.image_overlay is not None:
        image = train.image_overlay(image)
    # Mount the package so the trainer and the Ray workers can import the hooks.
    image = image.add_local_python_source("modal_training_gym", copy=True)
    return image


def _resolve_container_app_id() -> str:
    """Best-effort Modal app id from inside the running Trainer container, used
    as a fallback when the client didn't thread one in. A spawned deployed
    function does not reliably get ``MODAL_APP_ID`` in its env, so also consult
    the container's bound App object."""
    app_id = os.environ.get("MODAL_APP_ID", "")
    if app_id:
        return app_id
    try:
        container_app = modal.App._get_container_app()
        return (container_app.app_id if container_app else "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _record_run_started(
    *,
    run_id: str,
    recipe: StitchRecipe,
    model: ModelConfig | None,
    dataset: DatasetConfig | None,
    config_fields: dict,
    modal_app_id: str = "",
) -> TrainingRun | None:
    """Write a ``RUNNING`` :class:`TrainingRun` to the ``training-gym-metadata``
    Volume so the disagg run shows up in the dashboard (the deployed app is
    already tagged for auto-discovery; this adds the run record miles writes for
    itself in the colocated flow). Best-effort: a metadata hiccup must never take
    down the training run, so failures are logged and swallowed.

    The launching client writes the record first (sweep metadata, initializing
    phase), so this loads and mutates it rather than replacing it — a fresh
    record would drop the group id and blank the phase."""
    try:
        modal_app_id = modal_app_id or _resolve_container_app_id()
        wandb_block: dict = {}
        if recipe.wandb is not None:
            # Resolve the W&B entity for a dashboard deep-link; keep the run
            # alive if the probe fails (bad key / no access).
            entity = recipe.wandb.entity
            try:
                entity = preflight_wandb(recipe.wandb) or entity
            except Exception as exc:  # noqa: BLE001
                print(f"W&B preflight for dashboard deep-link failed: {exc}")
            wandb_block = {
                "project": recipe.wandb.project,
                "group": recipe.wandb.group,
                "entity": entity,
                "run_id": wandb_run_id_for_attempt(run_id, 1),
            }
        config_summary = {
            "model": {"model_name": model.model_name} if model else {},
            "dataset": (
                {
                    "hf_repo": getattr(dataset, "hf_repo", ""),
                    "name": type(dataset).__name__,
                }
                if dataset
                else {}
            ),
            # gpu_type is a miles _SKIP_FIELD (infra, not a CLI flag), but the
            # dashboard's cluster column and checkpoint conversion read it.
            "recipe": {"gpu_type": recipe.train.gpu_type, **config_fields},
            "wandb": wandb_block,
            "lr": recipe.train.lr,
            "global_batch_size": recipe.train.global_batch_size,
        }
        created_at = int(time.time())
        try:
            run_record = cast(TrainingRun, TrainingRun.from_id(run_id))
            run_record.modal_app_id = modal_app_id
            run_record.modal_app_url = modal_app_dashboard_url(modal_app_id)
            run_record.config = config_summary
            run_record.status = TrainingRunStatus.RUNNING
            run_record.started_at = run_record.started_at or created_at
        except KeyError:
            run_record = TrainingRun(
                training_run_id=run_id,
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_dashboard_url(modal_app_id),
                framework=Framework.STITCH,
                config=config_summary,
                status=TrainingRunStatus.RUNNING,
                created_at=created_at,
                started_at=created_at,
            )
        if wandb_block:
            record_wandb_attempt(
                run_record,
                entity=wandb_block["entity"],
                project=wandb_block["project"],
                group=wandb_block["group"],
                run_id=wandb_block["run_id"],
                attempt_count=1,
            )
        run_record.save()
        print(f"TrainingRun recorded for dashboard: {run_id}")
        return run_record
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to record TrainingRun {run_id} for dashboard: {exc}")
        return None


def _record_run_finished(
    run_record: TrainingRun | None, status: TrainingRunStatus
) -> None:
    """Stamp the terminal status + duration on the dashboard run record.
    Best-effort, mirroring :func:`_record_run_started`."""
    if run_record is None:
        return
    try:
        finished_at = int(time.time())
        run_record.status = status
        run_record.ended_at = finished_at
        if run_record.completed_at is None:
            run_record.completed_at = finished_at
        run_record.duration_seconds = max(0, finished_at - run_record.started_at)
        run_record.save()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to finalize dashboard TrainingRun: {exc}")


def build_stitch_app(
    *,
    model: ModelConfig,
    dataset: DatasetConfig,
    recipe: StitchRecipe,
    training_run_id: str = "",
    name: str | None = None,
    group_id: str | None = None,
    checkpoint: Checkpoint | None = None,
) -> modal.App:
    """Build the Modal App for disaggregated miles training.

    Returns an app with ``download``, ``prepare_dataset``,
    ``prepare_checkpoints``, and ``train`` (``build_miles_app``'s surface plus the
    served-baseline step), and the ``Server`` Flash-pool class that serves
    rollouts. ``train`` brings the pool's gateway up, claims it for the run, and
    drives miles; :class:`~modal_training_gym.common.train.TrainConfig` calls it
    for :class:`StitchRecipe` recipes.
    """
    if checkpoint is not None:
        # Resuming would have to move both halves at once: the trainer's load
        # path and the pool's served baseline (every delta applies against it),
        # which for a quantized run means re-running the conversion off the
        # checkpoint. Rejected rather than silently starting from the base model.
        raise TrainingGymConfigError(
            "resuming from a checkpoint is not supported for stitch runs: the "
            "rollout pool's served baseline is built by prepare_checkpoints "
            "from the recipe's source checkpoint, so a resumed trainer and the "
            "pool would disagree on the delta baseline"
        )
    StitchRecipe._resolve_data_paths(dataset)  # validate dataset paths resolve

    # Serialize the caller's module by value so inline ModelConfig/DatasetConfig
    # subclasses defined in a user script reach the containers.
    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    train_recipe, serve_recipe = recipe.train, recipe.serve
    app_name = recipe.name or name or f"stitch-{modal_tag_value(model.model_name)}"
    # Volumes are keyed by recipe (not by run) so runs of the same recipe reuse
    # the same dataset / checkpoints / bulletin board.
    volume_prefix = f"stitch-{modal_tag_value(type(recipe).__name__)}"
    delta_volume_name = (
        serve_recipe.delta_volume_name or f"{volume_prefix}-delta-bulletin"
    )
    delta_bulletin_root = serve_recipe.bulletin_root
    # Fresh run id per app: the trainer writes this run's chain under
    # <bulletin_root>/<run_id>/ and both halves are scoped to it, so a new run
    # never picks up a finished one's pointer — no manual bulletin reset needed.
    # Fixed at build time (not inside ``train``) because it is the sidecar's fence
    # token too, and the pool comes up with the app.
    run_id = uuid.uuid4().hex[:12]
    run_bulletin_root = f"{delta_bulletin_root}/{run_id}"
    # miles owns <run>/updates; stitch owns the pointer beside it.
    update_weight_disk_dir = f"{run_bulletin_root}/updates"
    # What the pool serves: the prepared baseline for a quantized run, else the
    # model's own checkpoint.
    served_model = serve_recipe.served_checkpoint_path or (
        model.model_path or model.model_name
    )
    rollout_concurrency = serve_recipe.concurrency
    n_train_nodes = train_recipe.actor_num_nodes

    tags = {
        **COMMON_TRAINING_GYM_TAGS,
        "_modal_framework": Framework.STITCH.value,
        "_modal_job_type": "training",
        **{str(k): str(v) for k, v in recipe.app_tags.items()},
    }
    if recipe.wandb is not None:
        if recipe.wandb.project:
            tags["wandb_project"] = modal_tag_value(recipe.wandb.project)
        if recipe.wandb.group:
            tags["wandb_group"] = modal_tag_value(recipe.wandb.group)

    image = _stitch_trainer_image(train_recipe)
    server_image = serving_image.build_serving_image(
        hf_cache_path=str(HF_CACHE_PATH),
        delta_volume_name=delta_volume_name,
        bulletin_root=delta_bulletin_root,
        runtime=serve_recipe.runtime,
        extra_env=serve_recipe.env,
    )
    sglang_server_args = serve_recipe.engine_args(model_name=served_model)
    delta_update_mode = serve_recipe.delta_update_mode
    commit_mode = serve_recipe.commit_mode
    flush_cache_on_commit = serve_recipe.flush_cache_on_commit
    gpus_per_replica = serve_recipe.gpus_per_replica

    hf_cache_volume = modal.Volume.from_name(
        "huggingface-cache", create_if_missing=True
    )
    data_volume = modal.Volume.from_name(
        f"{volume_prefix}-data", create_if_missing=True
    )
    checkpoints_volume_name = f"{volume_prefix}-checkpoints"
    checkpoints_volume = modal.Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    delta_volume = modal.Volume.from_name(
        delta_volume_name, create_if_missing=True, version=2
    )
    sglang_cache_volume = modal.Volume.from_name("sglang-cache", create_if_missing=True)
    train_volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        str(CHECKPOINTS_PATH): checkpoints_volume,
        delta_bulletin_root: delta_volume,
    }

    hf_secret = modal.Secret.from_name("huggingface-secret")
    train_secrets = [hf_secret]
    if recipe.wandb is not None:
        train_secrets.append(modal.Secret.from_name("wandb-secret"))

    memory = train_recipe.memory
    app = modal.App(app_name, tags=tags)

    @app.cls(
        image=server_image,
        gpu=f"{serve_recipe.gpu}:{serve_recipe.gpus_per_replica}",
        cloud=train_recipe.cloud,
        region=train_recipe.region,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            serving_image.SGLANG_CACHE_PATH: sglang_cache_volume,
            # A prepared (quantized) baseline lives on the checkpoints Volume,
            # so the replicas mount it read-only alongside the bulletin board.
            str(CHECKPOINTS_PATH): checkpoints_volume,
            delta_bulletin_root: delta_volume,
        },
        secrets=[hf_secret],
        memory=serve_recipe.memory,
        ephemeral_disk=serve_recipe.ephemeral_disk,
        min_containers=serve_recipe.min_containers,
        max_containers=serve_recipe.max_containers,
        timeout=40 * MINUTES,
        scaledown_window=15 * MINUTES,
        serialized=True,
    )
    @modal.experimental.http_server(
        port=SIDECAR_PORT,
        proxy_regions=serve_recipe.proxy_regions,
        exit_grace_period=25,
        startup_timeout=SERVER_STARTUP_TIMEOUT,
    )
    @modal.concurrent(target_inputs=rollout_concurrency)
    class Server:
        """One SGLang rollout server plus the stitch weight-sync sidecar."""

        @modal.enter()
        def startup(self) -> None:
            from cookbook.common import server

            server.serve_startup(
                self,
                # A local path (both the engine's model and the sidecar's delta
                # baseline): the sidecar can't seed a delta from a repo id, and a
                # post-boot resolve would race the cache SGLang warms itself.
                model_name=_local_checkpoint(served_model, checkpoints_volume_name),
                sglang_args=sglang_server_args,
                tp=gpus_per_replica,
                concurrency=rollout_concurrency,
                bulletin_root=run_bulletin_root,
                local_checkpoint_dir=LOCAL_CHECKPOINT_PATH,
                delta_update_mode=delta_update_mode,
                volume_name=delta_volume_name,
                run_id=run_id,
                commit_mode=commit_mode,
                flush_cache_on_commit=flush_cache_on_commit,
                startup_timeout=SERVER_STARTUP_TIMEOUT,
            )

        @modal.exit()
        def stop(self) -> None:
            from cookbook.common import server

            server.serve_stop(self)

    @app.function(
        image=image,
        gpu=f"{train_recipe.gpu_type}:{train_recipe.actor_num_gpus_per_node}",
        memory=memory,
        cloud=train_recipe.cloud,
        region=train_recipe.region,
        volumes=train_volumes,
        secrets=train_secrets,
        ephemeral_disk=train_recipe.ephemeral_disk,
        timeout=24 * 60 * MINUTES,
        startup_timeout=20 * MINUTES,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="train",
    )
    @clustered_if(True, n_train_nodes, gpu_type=train_recipe.gpu_type)
    def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
        rollout_endpoint_url: str = "",
    ) -> dict:
        """Bring up Ray, claim the rollout pool for this run, and drive miles."""
        del modal_app_url  # derived from modal_app_id in the run record
        from cookbook.common import hooks, launch, process, ray_cluster

        from modal_training_gym.frameworks.stitch import trainer_helpers

        rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(n_train_nodes)
        os.environ.update(
            {
                "MILES_HOST_IP": my_ip,
                "SGLANG_HOST_IP": my_ip,
                "HOST_IP": my_ip,
                "MASTER_ADDR": master_addr,
                "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
                "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
                "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
                "PYTHONPATH": train_recipe.megatron_pythonpath,
                **{str(k): str(v) for k, v in train_recipe.environment.items()},
            }
        )
        if framework_status_url:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_URL"] = framework_status_url
        if framework_status_token:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] = framework_status_token
        # Megatron is a source checkout in the image, so R3 dispatch + the
        # reshardable optimizer step arrive as patches. Applied on every node,
        # before the rank gate: each node's Ray actors import their own copy.
        if train_recipe.megatron_runtime_patches:
            process.apply_git_patches(
                train_recipe.megatron_runtime_patches, MEGATRON_PATH, "megatron-patch"
            )
        # Same reason: a Ray actor on another node re-reads this file by path, so
        # it can't live in rank 0's per-launch tmpdir.
        cfg_yaml_owner = _MilesArgs(
            {"te_precision_config_file": train_recipe.te_precision_config_file},
            async_mode=train_recipe.async_mode,
            miles_model_script=train_recipe.miles_model_script,
        )
        launch.materialize_node_local_yaml(cfg_yaml_owner, "te_precision_config_file")

        # Rank 0 drives the run; the other ranks only host Ray workers, and stay
        # alive until Modal tears the cluster down with rank 0's input.
        if rank != 0:
            start_ray_worker(my_ip, master_addr)
            while True:
                time.sleep(10)
        start_ray_head(my_ip, n_train_nodes, worker_wait_retries=180)
        for volume in (hf_cache_volume, data_volume, checkpoints_volume):
            volume.reload()
        # ``launch(prepare_inputs=False)`` (the default, and what a sweep uses)
        # skips the client-side prep calls, so the trainer prepares its own
        # inputs when they're missing rather than failing on a cold volume.
        prompt_data, eval_paths = StitchRecipe._resolve_data_paths(dataset)
        if not Path(prompt_data).exists():
            print(f"Preparing dataset ({prompt_data})...")
            dataset.prepare(prompt_data, eval_paths)
            data_volume.commit()
        # Both are no-ops on a warm cache.
        model.download()
        train_recipe.download_model()
        hf_cache_volume.commit()
        # The served baseline is the one input the trainer can't build here: the
        # conversion wants its own GPU function (prepare_checkpoints).
        for path in (train_recipe.hf_checkpoint, train_recipe.bf16_checkpoint_path):
            if str(path).startswith("/") and not Path(path).exists():
                raise RuntimeError(
                    f"prepared checkpoint {path} is missing — run the app's "
                    "prepare_checkpoints function (TrainConfig.train(), or "
                    "launch(prepare_inputs=True)) before training"
                )

        payload = recipe.to_payload(model=model, dataset=dataset)
        cfg = _MilesArgs(
            payload["fields"],
            async_mode=payload["async_mode"],
            miles_model_script=payload["miles_model_script"],
        )
        cfg.te_precision_config_file = cfg_yaml_owner.te_precision_config_file
        # The pool's Flash gateway, resolved by whoever launched this call (see
        # _PoolAwareTrain). Falls back to a lookup by app name, which only works
        # against a deployed pool.
        cfg.rollout_endpoint_url = (
            rollout_endpoint_url or trainer_helpers.deployed_gateway_url(app_name)
        )
        # Flash holds requests through a cold-starting pool, but the trainer's
        # first rollout would otherwise meet engines that are still loading.
        trainer_helpers.await_gateway_ready(
            cfg.rollout_endpoint_url, timeout_seconds=SERVER_STARTUP_TIMEOUT
        )
        cfg.update_weight_disk_dir = update_weight_disk_dir
        # stitch's publish + request hooks read these off the miles args
        # namespace; merge over any user extra_config already on
        # custom_config_path.
        custom_config = dict(getattr(cfg, "custom_config_path", None) or {})
        custom_config.update(
            {
                field: getattr(train_recipe, field)
                for field in sorted(HOOK_CONFIG_FIELDS)
            }
        )
        custom_config.update(
            {
                "experiment_volume_name": delta_volume_name,
                "rollout_modal_flash_app_name": app_name,
                "rollout_modal_flash_server_cls_name": "Server",
                "run_id": run_id,
            }
        )
        cfg.custom_config_path = custom_config

        # ``custom_config_path`` is a *path* flag: in the colocated flow the dict
        # is materialized while still named ``extra_config`` (which is in miles'
        # YAML_CONFIG_FIELDS) and renamed after. Here it is already renamed, so
        # it has to be materialized under its final name or miles is handed a
        # dict repr as a filename.
        launch.resolve_config(
            cfg, tempfile.mkdtemp(), (*YAML_CONFIG_FIELDS, "custom_config_path")
        )
        cmd = launch.build_train_cmd(cfg, MILES_ROOT, "miles_model_script")

        # Claim the pool for this run before miles publishes: write the empty
        # pointer and wake the pool so every replica resets to base now.
        hooks.claim_pool(
            SimpleNamespace(
                update_weight_disk_dir=cfg.update_weight_disk_dir,
                **custom_config,
            )
        )

        print(
            f"Training on {app_name}: nodes={n_train_nodes}, "
            f"rollout_endpoint={cfg.rollout_endpoint_url}"
        )
        print(f"Command: {cmd}")

        record_id = training_run_id or run_id
        run_record = _record_run_started(
            run_id=record_id,
            recipe=recipe,
            model=model,
            dataset=dataset,
            config_fields=payload["fields"],
            modal_app_id=modal_app_id,
        )
        wandb_run_id = ""
        if recipe.wandb is not None:
            # Force miles' W&B run to use the same id recorded in the
            # dashboard deep-link (miles/wandb honor these env vars). Without
            # this, wandb autogenerates a run id and the dashboard link 404s.
            wandb_run_id = wandb_run_id_for_attempt(record_id, 1)
            os.environ["WANDB_RUN_ID"] = wandb_run_id
            os.environ["WANDB_RESUME"] = "allow"
            if recipe.wandb.entity:
                os.environ["WANDB_ENTITY"] = recipe.wandb.entity
        # Tee the trainer's output to the checkpoints volume: a container's log
        # window only keeps the tail, so a failure whose traceback scrolled past
        # (rollout retries are loud) is otherwise unreadable afterwards.
        trainer_log = CHECKPOINTS_PATH / "logs" / f"{record_id}-trainer.log"
        trainer_log.parent.mkdir(parents=True, exist_ok=True)
        status = TrainingRunStatus.COMPLETED
        try:
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"set -o pipefail; ({cmd}) 2>&1 | tee -a {trainer_log}",
                ],
                check=True,
            )
        except BaseException:
            status = TrainingRunStatus.FAILED
            raise
        finally:
            _record_run_finished(run_record, status)
            checkpoints_volume.commit()

        result = TrainResult(
            app_name=app_name,
            framework=Framework.STITCH,
            training_run_id=record_id,
            checkpoint_dir=str(train_recipe.save),
            checkpoints_volume_name=checkpoints_volume_name,
            checkpoints_mount_path=str(CHECKPOINTS_PATH),
            model_config=model,
            wandb_project=recipe.wandb.project if recipe.wandb else "",
            wandb_entity=recipe.wandb.entity if recipe.wandb else "",
            wandb_training_run_id=wandb_run_id,
            group_id=group_id or "",
            extra={"rollout_endpoint_url": cfg.rollout_endpoint_url, "run_id": run_id},
        )
        result.save()
        checkpoints_volume.commit()
        return result._to_dict()

    @app.function(
        image=image,
        volumes={str(HF_CACHE_PATH): hf_cache_volume},
        timeout=2 * 60 * MINUTES,
        secrets=[hf_secret],
        serialized=True,
        name="download",
    )
    def download() -> None:
        model.download()
        train_recipe.download_model()
        hf_cache_volume.commit()

    @app.function(
        image=image,
        # The NVFP4 conversion runs miles' TE-direct quantizer, which needs a GPU;
        # a BF16 baseline is only a download.
        gpu=(
            f"{train_recipe.gpu_type}:1"
            if train_recipe.served_checkpoint_format != "bf16"
            else None
        ),
        memory=memory,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            str(CHECKPOINTS_PATH): checkpoints_volume,
        },
        timeout=6 * 60 * MINUTES,
        secrets=[hf_secret],
        ephemeral_disk=train_recipe.ephemeral_disk,
        serialized=True,
        name="prepare_checkpoints",
    )
    def prepare_checkpoints() -> None:
        """Build the trainer's BF16 masters and the pool's served baseline.

        A quantized run can't serve the HF repo directly: every replica boots from
        the baseline each sparse delta is applied against, so it must be the
        byte-exact output of the same quantizer the trainer exports with.
        """
        from cookbook.miles_disagg import prep

        # The cookbook's prep reads its constants off an experiment *module*; the
        # recipe is the same values under gym names.
        prep.prepare_checkpoints(
            SimpleNamespace(
                SOURCE_MODEL=train_recipe.source_hf_checkpoint
                or train_recipe.hf_checkpoint,
                BF16_CHECKPOINT_PATH=train_recipe.bf16_checkpoint_path,
                SERVED_CHECKPOINT_FORMAT=train_recipe.served_checkpoint_format,
                PREP_ENV=dict(train_recipe.prep_env),
                miles=train_recipe,
            ),
            checkpoints_volume,
        )
        hf_cache_volume.commit()

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * MINUTES,
        secrets=[hf_secret],
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset() -> None:
        data_volume.reload()
        prompt_data, eval_paths = StitchRecipe._resolve_data_paths(dataset)
        dataset.prepare(prompt_data, eval_paths)
        data_volume.commit()

    # Expose the functions as attributes (app.train, app.download, …) the way the
    # other launchers do, so callers address them without the registry.
    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)
    app.train = _PoolAwareTrain(train, Server)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    return app


class _PoolAwareTrain:
    """``app.train`` proxy that resolves the rollout pool's gateway client-side.

    The trainer can't discover it itself: an ephemeral app can't be looked up by
    name (``flash_get_containers`` / ``Cls.from_name`` need a deployed app), its
    containers' app object exposes no sibling objects, and a ``Cls`` handle can't
    be captured in the trainer's closure (Modal refuses to serialize unhydrated
    objects). The launching client, inside ``app.run()``, does have the hydrated
    handle — so it passes the gateway URL in as an argument.
    """

    def __init__(self, fn: modal.Function, server_cls: modal.Cls) -> None:
        self._fn = fn
        self._server_cls = server_cls

    def _with_gateway(self, kwargs: dict) -> dict:
        from modal_training_gym.frameworks.stitch import trainer_helpers

        if not kwargs.get("rollout_endpoint_url"):
            kwargs["rollout_endpoint_url"] = trainer_helpers.flash_gateway_url(
                self._server_cls
            )
        return kwargs

    def spawn(self, *args, **kwargs) -> modal.FunctionCall:
        return self._fn.spawn(*args, **self._with_gateway(kwargs))

    def remote(self, *args, **kwargs):
        return self._fn.remote(*args, **self._with_gateway(kwargs))

    def __getattr__(self, name: str):
        return getattr(self._fn, name)
