# ── Checkpoint ───────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import time

from modal import Volume
from modal.exception import NotFoundError

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.deploy_recipes import SglangRecipe, VllmRecipe
from modal_training_gym.frameworks.slime.launcher import SLIME_IMAGE


class CheckpointType(Enum):
    hf = "hf"
    megatron = "megatron"


@dataclass
class Checkpoint:
    """A single discovered checkpoint on the local filesystem."""

    checkpoint_type: CheckpointType
    name: str
    path: str
    timestamp: float
    training_run_id: str = ""
    app_name: str = ""
    checkpoints_volume_name: str = ""
    checkpoints_mount_path: str = ""


CheckpointConfig = Checkpoint


def list_checkpoints(training_run_id: str) -> list[Checkpoint]:
    result = TrainResult.from_training_run_id(training_run_id)
    if result.framework in {Framework.SLIME, Framework.SLIME.value}:
        return _list_checkpoints_for_slime(result)
    raise ValueError(f"Unsupported framework: {result.framework}")


def _get_slime_checkpoint_prefix() -> str:
    return "iter_"


def _to_volume_path(checkpoint_dir: str, checkpoints_mount_path: str) -> str:
    checkpoint_dir_norm = os.path.normpath(checkpoint_dir)
    checkpoints_mount_path_norm = os.path.normpath(checkpoints_mount_path)

    if os.path.isabs(checkpoint_dir_norm):
        if (
            checkpoint_dir_norm == checkpoints_mount_path_norm
            or checkpoint_dir_norm.startswith(checkpoints_mount_path_norm + os.sep)
        ):
            rel = os.path.relpath(checkpoint_dir_norm, checkpoints_mount_path_norm)
            return "" if rel == "." else rel
        return checkpoint_dir_norm.lstrip("/")

    return checkpoint_dir_norm.lstrip("/")


def _list_checkpoints_for_slime(train_result: "TrainResult") -> list[Checkpoint]:
    checkpoint_dir = train_result.checkpoint_dir.rstrip("/")
    if not checkpoint_dir:
        return []

    def _entry_name(entry: object) -> str:
        return getattr(entry, "path", "").rstrip("/").rsplit("/", 1)[-1]

    def _checkpoint_type(name: str) -> CheckpointType:
        return CheckpointType.hf if name.endswith("_hf") else CheckpointType.megatron

    checkpoints_volume_name = (
        train_result.checkpoints_volume_name
        or f"{train_result.app_name}-checkpoints"
    )
    checkpoints_mount_path = train_result.checkpoints_mount_path or "/checkpoints"
    volume = Volume.from_name(checkpoints_volume_name, create_if_missing=True)
    prefix = _get_slime_checkpoint_prefix()
    rel = _to_volume_path(checkpoint_dir, checkpoints_mount_path)

    try:
        entries = {
            _entry_name(entry): entry
            for entry in volume.iterdir(rel or "/", recursive=False)
        }
    except (FileNotFoundError, NotFoundError):
        return []

    if not prefix:
        entry = entries.get(os.path.basename(checkpoint_dir))
        if entry is None:
            return []
        name = os.path.basename(checkpoint_dir)
        checkpoint_type = (
            CheckpointType.hf
            if name.endswith("_hf")
            else CheckpointType.megatron
        )
        return [
            Checkpoint(
                checkpoint_type=checkpoint_type,
                name=name,
                path=checkpoint_dir,
                timestamp=float(getattr(entry, "mtime", 0.0)),
                training_run_id=train_result.training_run_id,
                app_name=train_result.app_name,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
            )
        ]

    def _is_dir_entry(entry: object) -> bool:
        is_dir_fn = getattr(entry, "is_dir", None)
        if callable(is_dir_fn):
            return bool(is_dir_fn())
        entry_type = getattr(entry, "type", None)
        if entry_type is None:
            return False
        entry_type_name = getattr(entry_type, "name", "")
        if isinstance(entry_type_name, str):
            return entry_type_name.upper() == "DIRECTORY"
        return False

    checkpoints: list[Checkpoint] = []
    for entry in sorted(
        (entry for entry in entries.values() if _is_dir_entry(entry)),
        key=lambda entry: _entry_name(entry),
    ):
        name = _entry_name(entry)
        if not name.startswith(prefix):
            continue

        checkpoints.append(
            Checkpoint(
                checkpoint_type=_checkpoint_type(name),
                name=name,
                path=os.path.join(checkpoint_dir, name),
                timestamp=float(getattr(entry, "mtime", 0.0)),
                training_run_id=train_result.training_run_id,
                app_name=train_result.app_name,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
            )
        )
    return checkpoints


def _conversion_gpu_spec(checkpoint: Checkpoint, recipe: VllmRecipe | SglangRecipe) -> str:
    run_id = getattr(checkpoint, "training_run_id", "")
    if run_id:
        try:
            training_run = TrainingRun.from_id(run_id)
        except KeyError:
            training_run = None
        if training_run is not None:
            recipe_config = training_run.config.get("recipe", {})
            gpu_type = recipe_config.get("gpu_type")
            n_gpu = recipe_config.get("actor_num_gpus_per_node")
            if gpu_type and n_gpu:
                n_gpu = int(n_gpu)
                return f"{gpu_type}:{n_gpu}" if n_gpu > 1 else str(gpu_type)

    if isinstance(recipe, SglangRecipe):
        gpu = recipe.gpu
        n_gpu = recipe.tp or 1
    else:
        gpu = recipe.gpu
        n_gpu = recipe.n_gpu or 1
    return f"{gpu}:{n_gpu}" if n_gpu > 1 else str(gpu)

def convert_checkpoint_to_hf(
    checkpoint: Checkpoint,
    model: ModelConfig,
    recipe: VllmRecipe | SglangRecipe,
) -> Checkpoint:
    import modal
    from modal import App, Image, Secret, Volume

    checkpoints_volume_name = checkpoint.checkpoints_volume_name
    if not checkpoints_volume_name:
        raise ValueError(
            "Cannot convert checkpoint without checkpoints volume metadata."
        )
    checkpoints_mount_path = checkpoint.checkpoints_mount_path or "/checkpoints"

    model_ref = model.model_name or model.model_path
    if not model_ref:
        raise ValueError(
            "Cannot convert a megatron checkpoint without model_name or model_path."
        )

    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    checkpoints_volume = Volume.from_name(
        checkpoints_volume_name,
        create_if_missing=True,
    )
    image = (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface")
        .add_local_python_source("modal_training_gym", copy=True)
    )
    conversion_app = App("training-gym-checkpoint-convert")
    gpu_spec = _conversion_gpu_spec(checkpoint, recipe)

    @conversion_app.function(
        image=image,
        gpu=gpu_spec,
        volumes={
            "/root/.cache/huggingface": hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=4 * 60 * 60,
        secrets=[Secret.from_name("huggingface-secret")],
        serialized=True,
        name="convert_megatron_to_hf",
    )
    def convert_megatron_to_hf(
        input_dir: str,
        output_dir: str,
        model_ref: str,
    ) -> str:
        import importlib.util
        import shlex
        import subprocess

        from huggingface_hub import snapshot_download

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        if os.path.isabs(model_ref) and os.path.isdir(model_ref):
            hf_path = model_ref
        else:
            hf_path = snapshot_download(model_ref, local_files_only=True)

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
        print(f"Converting checkpoint for serving: {cmd}")
        subprocess.run(["bash", "-c", cmd], check=True)
        checkpoints_volume.commit()
        return output_dir

    output_path = f"{checkpoint.path}_hf"
    with modal.enable_output():
        with conversion_app.run():
            output_path = convert_megatron_to_hf.remote(
                input_dir=checkpoint.path,
                output_dir=output_path,
                model_ref=model_ref,
            )

    return Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name=os.path.basename(output_path),
        path=output_path,
        timestamp=time.time(),
        training_run_id=checkpoint.training_run_id,
        app_name=checkpoint.app_name,
        checkpoints_volume_name=checkpoints_volume_name,
        checkpoints_mount_path=checkpoints_mount_path,
    )
