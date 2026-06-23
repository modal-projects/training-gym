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
import secrets as _secrets
import shlex
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Callable
from enum import Enum
from modal import App, Image, Secret, Volume

from modal_training_gym.common import hf_secrets

import cloudpickle

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS
from modal_training_gym.common.dataset import DatasetConfig, HarborDataset
from modal_training_gym.common.framework import (
    mount_tools_dir,
    resolve_caller_module,
)
from modal_training_gym.common.modal_refs import register_modal_cloudpickle_reducers
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.ray_cluster import (
    ModalRayCluster,
    _supports_rdma,
    clustered_if,
)
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.utils.metadata import MetadataStore, vol_put_async

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
    resolve_checkpoint_ref,
)
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.framework import Framework

SLIME_ROOT = "/root/slime"
# Pin by digest to prevent mutable-tag drift.  Tag: nightly-dev-20260529a
SLIME_IMAGE = "slimerl/slime@sha256:087a57732cf4fb271729df47530b01a9530144f4339247efc422f03e2b6988e1"
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"

_SLIME_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_VALIDATION_B64 = encode_patch("patch_validation", _SLIME_PATCHES)
_PATCH_MEGATRON_BRIDGE_B64 = encode_patch("patch_megatron_bridge", _SLIME_PATCHES)
_PATCH_TORCH_LOAD_B64 = encode_patch("patch_torch_load", _SLIME_PATCHES)
_PATCH_GLOBAL_PLAN_B64 = encode_patch("patch_global_plan", _SLIME_PATCHES)
_PATCH_CHECKPOINT_SAVE_B64 = encode_patch("patch_checkpoint_save", _SLIME_PATCHES)
_PATCH_ADVANTAGES_B64 = encode_patch("patch_advantages", _SLIME_PATCHES)
_PATCH_BRIDGE_NONE_TASK_B64 = encode_patch("patch_bridge_none_task", _SLIME_PATCHES)
_PATCH_GDN_PACKED_SEQ_B64 = encode_patch("patch_gdn_packed_seq", _SLIME_PATCHES)
_PATCH_BRIDGE_PER_TOKEN_LOSS_B64 = encode_patch(
    "patch_bridge_provider_per_token_loss", _SLIME_PATCHES
)
_PATCH_STOP_TOKEN_DIAG_B64 = encode_patch("patch_stop_token_diagnostic", _SLIME_PATCHES)
# The Qwen3-ASR Megatron->HF converter (registers the qwen3_asr mapping incl. the
# audio tower). It lives in the base image — not the ASR recipe — because torch_dist
# -> HF conversion runs in the shared convert_checkpoint_to_hf path (deploy/eval),
# which has no recipe; baking it here makes both train-time export and deploy-time
# conversion ASR-capable. Additive + idempotent, so non-ASR runs are untouched.
_PATCH_QWEN3_ASR_EXPORT_B64 = encode_patch(
    "patch_qwen3_asr_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_asr",
)
_PATCH_ROLLOUT_STATUS_B64 = encode_patch(
    "patch_rollout_status_reporting", _SLIME_PATCHES
)
_PATCH_LOG_ELIDE_B64 = encode_patch("patch_log_elide", _SLIME_PATCHES)


def _build_slime_base_image() -> "Image":
    return (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands(
            "rm -rf /root/.cache/huggingface",
            f"echo {_PATCH_MEGATRON_BRIDGE_B64} | base64 -d | python3",
            f"echo {_PATCH_ADVANTAGES_B64} | base64 -d | python3",
            f"echo {_PATCH_BRIDGE_NONE_TASK_B64} | base64 -d | python3",
            f"echo {_PATCH_STOP_TOKEN_DIAG_B64} | base64 -d | python3",
            f"echo {_PATCH_QWEN3_ASR_EXPORT_B64} | base64 -d | python3",
            f"echo {_PATCH_ROLLOUT_STATUS_B64} | base64 -d | python3",
            f"echo {_PATCH_LOG_ELIDE_B64} | base64 -d | python3",
        )
    )


def _build_conversion_config(slime_cfg: Any, model: Any = None) -> dict[str, Any]:
    """Build a dict of parameters that affect the torch_dist checkpoint layout.

    Written to ``.conversion_config.json`` inside the checkpoint directory after
    conversion so that later runs can detect stale checkpoints whose parallelism
    no longer matches the current recipe.
    """
    from modal_training_gym.frameworks.slime.modal_helpers.utils import (
        get_checkpoint_conversion_policy,
    )

    num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
        slime_cfg, model=model
    )
    return {
        "num_nodes": num_nodes,
        "nproc_per_node": nproc_per_node,
        "extra_args": extra_args,
        "model_name": model.model_name if model else None,
        "slime_model_script": getattr(slime_cfg, "slime_model_script", ""),
    }


_CONVERSION_CONFIG_FILE = ".conversion_config.json"


def _response_parser_path(model: Any) -> str:
    """Import path of the model's response parser so the rollout recorder can
    resolve and apply it remotely. Empty when the model sets no parser."""
    fn = getattr(model, "response_parser", None) if model is not None else None
    if fn is None:
        return ""
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "")
    return f"{module}.{qualname}" if module and qualname else ""


def _has_torch_dist_checkpoint(save_path: str) -> bool:
    if not os.path.isdir(save_path):
        return False

    def _is_complete_checkpoint_dir(path: str) -> bool:
        try:
            names = os.listdir(path)
        except OSError:
            return False
        return "common.pt" in names and any(name.endswith(".distcp") for name in names)

    tracker_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker_path):
        try:
            with open(tracker_path) as f:
                marker = f.read().strip()
        except OSError:
            marker = ""
        if marker == "release":
            return _is_complete_checkpoint_dir(os.path.join(save_path, "release"))
        if marker.isdigit():
            iter_dir = f"iter_{int(marker):07d}"
            return _is_complete_checkpoint_dir(os.path.join(save_path, iter_dir))

    try:
        return any(
            entry.is_dir()
            and (entry.name == "release" or entry.name.startswith("iter_"))
            and _is_complete_checkpoint_dir(entry.path)
            for entry in os.scandir(save_path)
        )
    except OSError:
        return False


def _serialize_recipe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path | PurePosixPath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_recipe_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_serialize_recipe_value(v) for v in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return f"{module}.{name}" if module and name else repr(value)
    return repr(value)


def _is_sensitive_recipe_field(name: str) -> bool:
    normalized = name.lower()
    if normalized.endswith("token_id") or normalized.endswith("token_ids"):
        return False
    return (
        any(
            marker in normalized
            for marker in ("api_key", "access_key", "secret", "password", "token")
        )
        or normalized == "wandb_key"
    )


def _serialize_slime_param_value(name: str, value: Any) -> Any:
    if _is_sensitive_recipe_field(name):
        return "[redacted]" if value not in (None, "", False) else value
    if isinstance(value, dict):
        return {
            str(k): _serialize_slime_param_value(str(k), v) for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [_serialize_slime_param_value(name, v) for v in value]
    return _serialize_recipe_value(value)


def _serialize_slime_params(
    recipe: SlimeRecipe,
    *,
    dataset: DatasetConfig | None = None,
    model: ModelConfig | None = None,
) -> dict[str, Any]:
    return {
        key: _serialize_slime_param_value(key, value)
        for key, value in recipe._fields(dataset=dataset, model=model).items()
    }


def _preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Thin wrapper around :func:`~modal_training_gym.common.wandb.preflight_wandb`."""
    from modal_training_gym.common.wandb import preflight_wandb

    return preflight_wandb(wandb_cfg)


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
    volume_prefix = f"slime-{type(slime).__name__.lstrip('_').lower()}"

    SlimeRecipe._validate_custom_model_architecture(model)
    SlimeRecipe._validate_dataset(dataset)

    # Models that can't do THD packing (model.requires_bshd, e.g. Qwen3-ASR) must
    # train on padded (bshd) batches; fail fast with the fix if the recipe didn't.
    if model and getattr(model, "requires_bshd", False):
        cfg = slime.extra_config or {}
        if cfg.get("qkv_format") != "bshd" or slime.use_dynamic_batch_size:
            raise ValueError(
                f"{model.model_name} requires padded (bshd) batches: its "
                "megatron-bridge forward doesn't implement THD sequence packing. "
                'Set extra_config={"qkv_format": "bshd", "micro_batch_size": N} and '
                "use_dynamic_batch_size=False — or use Qwen3_ASR_1_7b_Recipe, which sets "
                f"these. Got qkv_format={cfg.get('qkv_format')!r}, "
                f"use_dynamic_batch_size={slime.use_dynamic_batch_size}."
            )

    if (
        model
        and getattr(slime, "megatron_to_hf_mode", "") != "bridge"
        and not slime.ref_load
    ):
        # Non-bridge: pre-convert HF -> torch_dist (convert_checkpoint) and load that as the
        # reference checkpoint. In bridge mode we instead load the HF weights directly via
        # AutoBridge; ref_load is set to the local HF snapshot dir at train time.
        slug = model.model_name.replace("/", "--")
        object.__setattr__(slime, "ref_load", f"/checkpoints/torch_dist/{slug}-v31")

    # ── GDN compatibility ─────────────────────────────────────────────────
    # Models with Gated Delta Net (GDN) layers (use_gated_attention=True)
    # don't support packed sequences in the older Megatron-LM bundled with
    # slime.  Slime's get_batch() always creates PackedSeqParams for THD
    # format, which GDN rejects with NotImplementedError.  A build-time
    # patch (patch_gdn_packed_seq.py) neutralises the raise so GDN falls
    # back to unpacked processing.
    _has_gdn = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "use_gated_attention", False)
        and not slime.slime_model_script
    )

    caller_module = resolve_caller_module()
    if caller_module is not None and caller_module.__name__ != "__main__":
        cloudpickle.register_pickle_by_value(caller_module)
    register_modal_cloudpickle_reducers()

    caller_script = None
    if caller_module is not None:
        mod_file = getattr(caller_module, "__file__", None)
        if mod_file and os.path.isfile(mod_file):
            caller_script = os.path.abspath(mod_file)

    # ── Image ────────────────────────────────────────────────────────────────
    image = _build_slime_base_image()

    # Hybrid models have layers with different parameter sets (e.g. GDN
    # layers carry linear_attn.dt_bias that standard attention layers lack).
    # Megatron's validate_sharding_integrity rejects this because not every
    # position in the global tensor is covered.  Patch both the conversion
    # and training images so saving and loading both succeed.
    _has_hybrid_spec = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "megatron_spec", None)
        and not slime.slime_model_script
    )
    if slime.image_overlay is not None:
        image = slime.image_overlay(image)
        object.__setattr__(slime, "image_overlay", None)

    for patch in slime.patch_files:
        image = image.add_local_file(
            patch,
            remote_path=f"/tmp/{os.path.basename(patch)}",
            copy=True,
        )

    if isinstance(dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    if slime.local_slime:
        image = image.add_local_dir(
            slime.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

    if slime.image_run_commands:
        image = image.run_commands(*slime.image_run_commands)
    if slime.image_env:
        image = image.env(slime.image_env)

    image = image.add_local_python_source("modal_training_gym", copy=True)
    image = image.uv_pip_install("randomname")
    image = mount_tools_dir(image)

    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        caller_remote_path = f"/root/{caller_module_name}.py"
        image = image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )

    # Patch both conversion and training images for hybrid models.
    # The validation patch lets save/load succeed despite non-uniform
    # layer parameters.  The torch.py patch handles BytesIO entries
    # from _extra_state during checkpoint loading.
    if _has_hybrid_spec:
        image = image.run_commands(
            f"echo {_PATCH_VALIDATION_B64} | base64 -d | python3",
        )

    def _get_custom_generate_path() -> str:
        cfg = slime.extra_config
        if not isinstance(cfg, dict):
            return ""
        raw = cfg.get("custom_generate_function_path", "")
        return raw if isinstance(raw, str) else ""

    def _set_custom_generate_path(path: str) -> None:
        cfg = dict(slime.extra_config) if isinstance(slime.extra_config, dict) else {}
        cfg["custom_generate_function_path"] = path
        object.__setattr__(slime, "extra_config", cfg)

    def _set_extra_config_path(key: str) -> Callable[[str], None]:
        def setter(path: str) -> None:
            cfg = (
                dict(slime.extra_config) if isinstance(slime.extra_config, dict) else {}
            )
            cfg[key] = path
            object.__setattr__(slime, "extra_config", cfg)

        return setter

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
            # Point the slime arg at the shipped module's symbol. Without this the
            # file is shipped but the path stays unset, so a custom_rm_function
            # defined outside the entrypoint silently falls back to rule-based RM.
            set_path(f"{fn_module_name}.{getattr(fn, '__name__', fallback_name)}")
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
        cfg = dict(slime.extra_config) if isinstance(slime.extra_config, dict) else {}
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
        value = getattr(slime, attr)
        _ship_callable(
            value if callable(value) else None,
            fallback_name=fallback_name,
            set_path=_set_extra_config_path(config_key),
        )
        if callable(value):
            object.__setattr__(slime, attr, None)

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

    # Build train_image AFTER _ship_callable so shipped modules are included.
    train_image = image
    if _has_hybrid_spec:
        train_image = image.run_commands(
            f"echo {_PATCH_TORCH_LOAD_B64} | base64 -d | python3",
            f"echo {_PATCH_GLOBAL_PLAN_B64} | base64 -d | python3",
            f"echo {_PATCH_CHECKPOINT_SAVE_B64} | base64 -d | python3",
        )
    if _has_gdn:
        train_image = train_image.run_commands(
            f"echo {_PATCH_GDN_PACKED_SEQ_B64} | base64 -d | python3",
        )
    if slime.megatron_to_hf_mode == "bridge":
        train_image = train_image.run_commands(
            f"echo {_PATCH_BRIDGE_PER_TOKEN_LOSS_B64} | base64 -d | python3",
        )

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{volume_prefix}-data", create_if_missing=True)
    checkpoints_volume_name = (
        checkpoint.checkpoints_volume_name
        if checkpoint is not None and checkpoint.checkpoints_volume_name
        else f"{volume_prefix}-checkpoints"
    )
    checkpoints_mount_path = (
        checkpoint.checkpoints_mount_path.rstrip("/") or "/"
        if checkpoint is not None and checkpoint.checkpoints_mount_path
        else str(CHECKPOINTS_PATH).rstrip("/")
    )
    checkpoints_volume = Volume.from_name(
        checkpoints_volume_name, create_if_missing=True
    )
    metadata_volume = Volume.from_name("training-gym-metadata", create_if_missing=True)
    if checkpoint is not None and checkpoint.path and not model.model_path:
        model.model_path = checkpoint.path
    all_volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        checkpoints_mount_path: checkpoints_volume,
        "/metadata": metadata_volume,
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
    def download(
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
                SlimeStatus.DOWNLOAD_MODEL.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )
        hf_cache_volume.reload()
        checkpoints_volume.reload()
        model.download()
        hf_cache_volume.commit()
        checkpoints_volume.commit()
        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

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

    convert_nnodes = get_checkpoint_conversion_policy(slime, model=model)[0]

    @app.function(
        image=image,
        gpu=gpu_spec,
        memory=slime.memory,
        cloud=slime.cloud,
        region=slime.region,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered_if(convert_nnodes > 1, convert_nnodes, gpu_type=slime.gpu_type)
    def convert_checkpoint(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from huggingface_hub import snapshot_download
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
            flush as flush_status_reporter,
        )

        if training_run_id:
            enqueue_framework_status(
                training_run_id,
                SlimeStatus.CONVERT_MODEL.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )

        # Bridge mode loads the HF weights directly into Megatron via AutoBridge at train time
        # (slime's _load_checkpoint_hf), so there is no offline HF→torch_dist conversion to run.
        # The HF reference path is wired into `ref_load` in `train` below.
        if getattr(slime, "megatron_to_hf_mode", None) == "bridge":
            print(
                "Bridge mode — HF weights loaded directly via AutoBridge; no conversion needed."
            )
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        if slime.megatron_conversion_hf_checkpoint:
            hf_path = resolve_checkpoint_ref(slime.megatron_conversion_hf_checkpoint)
        elif model.model_path:
            hf_path = str(model.model_path)
        else:
            hf_path = snapshot_download(model.model_name, local_files_only=True)
        save_path = str(slime.ref_load)

        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            slime, model=model
        )
        node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

        import json
        import shutil

        current_config = _build_conversion_config(slime, model=model)

        if os.path.exists(save_path):
            complete = _has_torch_dist_checkpoint(save_path)
            stale = True
            if complete:
                config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
                if os.path.isfile(config_path):
                    try:
                        with open(config_path) as f:
                            stored_config = json.load(f)
                        stale = stored_config != current_config
                        if stale and node_rank == 0:
                            print(
                                f"Checkpoint at {save_path} was built with "
                                f"different config:\n  stored: {stored_config}"
                                f"\n  current: {current_config}"
                            )
                    except (OSError, json.JSONDecodeError):
                        stale = True
                        if node_rank == 0:
                            print(
                                f"Checkpoint at {save_path} has unreadable "
                                f"conversion config — reconverting."
                            )
                else:
                    # Legacy checkpoint without config metadata — cannot
                    # verify compatibility.  Force re-conversion so the
                    # checkpoint is rebuilt with the correct layout and
                    # metadata for future validation.
                    stale = True
                    if node_rank == 0:
                        print(
                            f"Checkpoint at {save_path} has no conversion "
                            f"config metadata — reconverting to ensure "
                            f"compatibility."
                        )
                if not stale:
                    if node_rank == 0:
                        print(f"Using existing torch_dist checkpoint at {save_path}.")
                    if training_run_id:
                        flush_status_reporter(timeout_seconds=2.0)
                    return

            # Either incomplete (e.g. a preempted conversion) or stale: rebuild
            # it. Rank 0 removes the directory and commits; other ranks wait for
            # the removal to land before reconverting.
            if node_rank == 0:
                import shutil

                reason = "stale" if complete else "incomplete"
                print(f"Removing {reason} torch_dist checkpoint at {save_path}.")
                shutil.rmtree(save_path, ignore_errors=True)
                checkpoints_volume.commit()
            else:
                time.sleep(5)
                checkpoints_volume.reload()

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        import importlib.util

        mmt = ""
        if model and getattr(model, "architecture", None):
            mmt = getattr(model.architecture, "megatron_model_type", "")

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
        if mmt or slime.slime_model_script:
            model_script = (
                f"{SLIME_ROOT}/{slime.slime_model_script}"
                if slime.slime_model_script
                else f"{SLIME_ROOT}/scripts/models/{mmt}.sh"
            )
            cmd = (
                f"source {model_script} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                '"${MODEL_ARGS[@]}" '
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        else:
            cmd = (
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )

        env = {**os.environ, **slime.environment}
        env.pop("NCCL_NVLS_ENABLE", None)
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"
        print(
            f"Conversion layout: nodes={num_nodes}, "
            f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        subprocess.run(["bash", "-c", cmd], check=True, env=env)

        if node_rank == 0:
            config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
            try:
                with open(config_path, "w") as f:
                    json.dump(current_config, f)
            except OSError as exc:
                print(f"WARNING: could not write conversion config: {exc}")
        checkpoints_volume.commit()

        if node_rank == 0:
            print(f"Saved torch_dist checkpoint to {save_path}")

        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

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

    # Use Modal's clustered scheduler with RDMA when using a full node (8+ GPUs)
    # on RDMA-capable hardware, or for any multi-node run.  The `rdma=True` flag
    # provides CAP_IPC_LOCK and NVSwitch device access that slime's colocated
    # weight sync (UpdateWeightFromTensor) needs for fast CUDA IPC transfers.
    _multi_node = slime.total_nodes > 1
    _full_node = slime.actor_num_gpus_per_node >= 8
    _use_clustered = _multi_node or (_full_node and _supports_rdma(slime.gpu_type))

    train_secrets: list[Secret] = []
    if slime.wandb is not None:
        train_secrets.append(Secret.from_name(slime.wandb.modal_wandb_secret_name))
    train_experimental_options: dict[str, Any] = {"efa_enabled": True}

    train_function_kwargs = dict(slime.train_function_kwargs or {})
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
        raise TypeError(f"Unsupported slime.train_function_kwargs keys: {unsupported}")

    @app.function(
        image=train_image,
        gpu=gpu_spec,
        memory=slime.memory,
        cloud=slime.cloud,
        region=slime.region,
        volumes=all_volumes,
        secrets=train_secrets or None,
        ephemeral_disk=train_ephemeral_disk,
        timeout=24 * 60 * 60,
        experimental_options=train_experimental_options or None,
        serialized=True,
        name="train",
    )
    @clustered_if(_use_clustered, slime.total_nodes, gpu_type=slime.gpu_type)
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        modal_app_id = modal_app_id or os.environ.get("MODAL_APP_ID", "")
        modal_app_url = modal_app_url or modal_app_dashboard_url(modal_app_id)

        # Make the dashboard URL visible to both the launcher's own
        # status_reporter and (via runtime_env below) the slime worker
        # process. The toml file lives on the user's local machine and isn't
        # accessible inside this container, so the URL has to be passed in.
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
        cluster.discover_cluster(slime.total_nodes)

        os.environ["SLIME_HOST_IP"] = cluster.node_ip
        os.environ["SGLANG_HOST_IP"] = cluster.node_ip
        os.environ["HOST_IP"] = cluster.node_ip

        cluster.start_ray()

        if not cluster.is_head:
            await cluster.wait_forever()
            return

        # Fail fast on W&B access before any GPU work, not as a recurring CommError
        # mid-training.
        wandb_entity = ""
        if slime.wandb is not None:
            wandb_entity = _preflight_wandb(slime.wandb)

        wandb_run_id = training_run_id[:8] if slime.wandb else ""

        print(f"Training run id: {training_run_id}")
        config_summary: dict = {
            "model": {"model_name": model.model_name} if model else {},
            "recipe": _serialize_slime_params(slime, dataset=dataset, model=model),
            "wandb": (
                {
                    "project": slime.wandb.project,
                    "group": slime.wandb.group,
                    "entity": wandb_entity,
                    "run_id": wandb_run_id,
                }
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
        # The local TrainConfig.train() driver creates the initial TrainingRun
        # record before invoking download/convert_checkpoint so those phases
        # are visible in the dashboard. Reuse it; fall back to a fresh record
        # if someone invokes train() directly (e.g. older callers).
        try:
            run_record = await TrainingRun.from_id_async(training_run_id)
            run_record.modal_app_id = modal_app_id
            run_record.modal_app_url = modal_app_url or modal_app_dashboard_url(
                modal_app_id
            )
            run_record.config = config_summary
            run_record.framework_status = SlimeStatus.INITIALIZING
        except KeyError:
            created_at = int(time.time())
            run_record = TrainingRun(
                training_run_id=training_run_id,
                modal_app_id=modal_app_id,
                modal_app_url=modal_app_url or modal_app_dashboard_url(modal_app_id),
                framework=Framework.SLIME,
                config=config_summary,
                framework_status=SlimeStatus.INITIALIZING,
                created_at=created_at,
                started_at=created_at,
            )
        if not framework_status_token:
            framework_status_token = _secrets.token_urlsafe(32)
        await run_record.save_async()
        await vol_put_async(
            MetadataStore.FRAMEWORK_STATUS_TOKENS,
            training_run_id,
            {"token": framework_status_token},
        )
        print(f"TrainingRun recorded: {training_run_id}")

        try:  # Wraps all post-setup work so any failure marks the run terminal.
            # In-flight status updates are fire-and-forget via the dashboard's
            # /api/framework-status endpoint so the training thread doesn't pay
            # the ~300ms volume-write latency on each transition. Terminal state
            # (COMPLETED/FAILED/STOPPED) still goes through run_record.save_async
            # below to guarantee delivery before the container exits.
            from modal_training_gym.common.status_reporter import (
                enqueue_framework_status,
            )

            def _set_framework_status(status: SlimeStatus) -> None:
                run_record.framework_status = status
                enqueue_framework_status(
                    training_run_id, status.value, token=framework_status_token
                )

            async def _set_framework_status_async(status: SlimeStatus) -> None:
                _set_framework_status(status)

            if model:
                await _set_framework_status_async(SlimeStatus.DOWNLOAD_MODEL)
                cache_dir = (
                    HF_CACHE_PATH
                    / "hub"
                    / f"models--{model.model_name.replace('/', '--')}"
                )
                snapshots_dir = cache_dir / "snapshots"
                has_snapshot = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
                if not has_snapshot:
                    print(f"Downloading model {model.model_name}...")
                model.download()  # Always run (idempotent; applies config patches to cached snapshots)
                await hf_cache_volume.commit.aio()

            if dataset:
                await _set_framework_status_async(SlimeStatus.PREPARE_DATASET)
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

            await _set_framework_status_async(SlimeStatus.CONVERT_MODEL)
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
            original_load = slime.load
            original_ref_load = slime.ref_load
            object.__setattr__(slime, "save", save_root)

            # Resolve the local HF snapshot dir (used for bridge-mode load below).
            _hf_ref: str | None = None
            if model and (slime.megatron_to_hf_mode == "bridge" or slime.ref_load):
                from huggingface_hub import snapshot_download as _snap0

                _hf_ref = (
                    str(model.model_path)
                    if model.model_path
                    else _snap0(model.model_name, local_files_only=True)
                )

            if _has_torch_dist_checkpoint(save_root):
                print(
                    f"Detected existing checkpoint in {save_root}; "
                    "will resume training from last saved iteration."
                )
                object.__setattr__(slime, "load", save_root)
            elif (
                slime.megatron_to_hf_mode == "bridge" and not slime.ref_load and _hf_ref
            ):
                # Fresh bridge run: load the HF weights directly via AutoBridge. slime falls back
                # args.load -> args.ref_load, and _load_checkpoint_hf maps the HF dir into Megatron
                # (weights only — no optimizer/RNG state, so no torch_dist is required). Pointing
                # ref_load at a torch_dist here would instead trigger the full-resume path and fail
                # on the missing optimizer state.
                object.__setattr__(slime, "ref_load", _hf_ref)
            try:
                cmd = build_train_cmd(slime, SLIME_ROOT, model=model, dataset=dataset)
            finally:
                object.__setattr__(slime, "save", original_save)
                object.__setattr__(slime, "load", original_load)
                object.__setattr__(slime, "ref_load", original_ref_load)

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

            runtime_env = {
                "env_vars": {
                    "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                    "MASTER_ADDR": cluster.head_addr,
                    "TRAINING_GYM_TRAINING_RUN_ID": training_run_id,
                    "TRAINING_GYM_APP_NAME": app_name,
                    "TRAINING_GYM_TOTAL_STEPS": str(slime.num_rollout),
                    "TRAINING_GYM_RESPONSE_PARSER_PATH": _response_parser_path(model),
                    "TRAINING_GYM_CAPTURE_TRACE": (
                        "1" if getattr(slime, "capture_trace", False) else ""
                    ),
                    "TRAINING_GYM_TRACE_SAMPLE_LIMIT": str(
                        getattr(slime, "trace_sample_limit", 16)
                    ),
                    "TRAINING_GYM_FRAMEWORK_STATUS_URL": phase_report_url,
                    **wandb_env,
                    **slime.environment,
                    "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN": framework_status_token,
                }
            }

            mode = "async" if slime.async_mode else "sync"
            print(
                f"Training {app_name} — {slime.total_nodes} node(s) × {gpu_spec}  ({mode})"
            )
            print(f"Command: {cmd}, runtime_env: {runtime_env}")

            await _set_framework_status_async(SlimeStatus.ROLLOUT_INITIALIZING)
            async with cluster.forward_dashboard() as tunnel:
                print(f"Ray dashboard: {tunnel.url}")
                await cluster.submit_and_tail(cmd, runtime_env=runtime_env)

            result_kwargs = {
                "app_name": app_name,
                "framework": Framework.SLIME,
                "training_run_id": training_run_id,
                "checkpoint_dir": save_root,
                "model_config": model,
                "checkpoints_volume_name": checkpoints_volume_name,
                "checkpoints_mount_path": checkpoints_mount_path,
                "wandb_project": slime.wandb.project if slime.wandb else "",
                "wandb_entity": wandb_entity,
                "wandb_training_run_id": wandb_run_id,
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
            try:
                await run_record.save_async()
            except Exception:
                pass

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
