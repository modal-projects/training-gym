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
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Callable, Mapping
from modal import App, Dict as ModalDict, Image, Secret, Volume, Retries

from modal_training_gym.common import hf_secrets, proxy_auth_secrets


from modal_training_gym.common.dataset import DatasetConfig, HarborDataset
from modal_training_gym.common.framework import (
    mount_tools_dir,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.ray_cluster import (
    ModalRayCluster,
    _supports_rdma,
    clustered_if,
)
from modal_training_gym.common.run import (
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
from modal_training_gym.common.launcher_utils import (
    serialize_recipe_params,
)
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.step_timing import Substep

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
# Pin by digest to prevent mutable-tag drift.  Tag: nightly-dev-20260722a
SLIME_IMAGE = "slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342"
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
# -> HF conversion runs in the shared convert_megatron_checkpoint_to_hf path (deploy/eval),
# which has no recipe; baking it here makes both train-time export and deploy-time
# conversion ASR-capable. Additive + idempotent, so non-ASR runs are untouched.
_PATCH_QWEN3_ASR_EXPORT_B64 = encode_patch(
    "patch_qwen3_asr_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_asr",
)
# The Qwen3-VL Megatron->HF converters: a qwen3_vl per-param mapping (language
# stack + frozen-ViT identity passthrough) and a torch_dist->HF shim that skips
# the frozen ViT's stacked layers.
_PATCH_QWEN3_VL_EXPORT_B64 = encode_patch(
    "patch_qwen3_vl_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_vl",
)
_PATCH_QWEN3_VL_TORCH_DIST_B64 = encode_patch(
    "patch_qwen3_vl_torch_dist",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_vl",
)
_PATCH_ROLLOUT_STATUS_B64 = encode_patch(
    "patch_rollout_status_reporting", _SLIME_PATCHES
)
_PATCH_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", _SLIME_PATCHES)
_PATCH_LOG_ELIDE_B64 = encode_patch("patch_log_elide", _SLIME_PATCHES)
# Backport of NVIDIA/Megatron-LM #3845: dequantize quantized CUDA tensors in the
# async dist-checkpoint writer before serialization. slime pins a pre-#3845
# Megatron, so FP8/TE _extra_state tensors otherwise crash the torch_dist save
# with inline_container.cc "unexpected pos" (e.g. the GLM-5.2 convert). No-op for
# non-quantized tensors, so safe for every image.
_PATCH_DIST_CKPT_QUANTIZED_B64 = encode_patch(
    "patch_dist_ckpt_quantized", _SLIME_PATCHES
)
# OPD / multi-turn: zero-std metrics must skip non-numeric rewards (dict/None).
_PATCH_ZERO_STD_METRICS_B64 = encode_patch("patch_zero_std_metrics", _SLIME_PATCHES)
_PATCH_SGLANG_PARALLEL_ALIASES_B64 = encode_patch(
    "patch_sglang_parallel_aliases", _SLIME_PATCHES
)


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
            f"echo {_PATCH_QWEN3_VL_EXPORT_B64} | base64 -d | python3",
            f"echo {_PATCH_QWEN3_VL_TORCH_DIST_B64} | base64 -d | python3",
            f"echo {_PATCH_ROLLOUT_STATUS_B64} | base64 -d | python3",
            f"echo {_PATCH_ADVANTAGE_DIST_B64} | base64 -d | python3",
            f"echo {_PATCH_LOG_ELIDE_B64} | base64 -d | python3",
            f"echo {_PATCH_DIST_CKPT_QUANTIZED_B64} | base64 -d | python3",
            f"echo {_PATCH_ZERO_STD_METRICS_B64} | base64 -d | python3",
            f"echo {_PATCH_SGLANG_PARALLEL_ALIASES_B64} | base64 -d | python3",
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


def _is_complete_torch_dist_checkpoint(path: str) -> bool:
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return "common.pt" in names and any(name.endswith(".distcp") for name in names)


def _checkpoint_conversion_cache_status(
    save_path: str, current_config: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Return the state of a cached HF-to-torch-dist conversion."""
    if not os.path.exists(save_path):
        return "missing", None
    if not has_torch_dist_checkpoint(
        save_path, is_complete=_is_complete_torch_dist_checkpoint
    ):
        return "incomplete", None

    import json

    config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
    if not os.path.isfile(config_path):
        return "stale", None
    try:
        with open(config_path) as f:
            stored_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "stale", None
    if stored_config != current_config:
        return "stale", stored_config
    return "hit", stored_config


_serialize_slime_params = serialize_recipe_params


def _preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Thin wrapper around :func:`~modal_training_gym.common.wandb.preflight_wandb`."""
    from modal_training_gym.common.wandb import preflight_wandb

    return preflight_wandb(wandb_cfg)


def aggregate_step_times(
    step_times_dict: Mapping[str, Any],
    run_id: str,
    num_steps: int,
    SUBSTEP_ORDER: list[str],
    OPTIONAL_SUBSTEPS: set[str],
) -> tuple[
    dict[str, dict[str, int | None]],
    dict[str, dict[str, dict[str, float | None]]],
]:
    """Organize step and substep timings from the Modal dict at the end of a run.

    Records each step's start/end/duration; missing timestamps become None.
    Each substep's duration is the gap from its start to the next recorded
    substep's start (or the step's end for the last one). Duration is None
    if a mandatory substep in between is missing. Step duration is computed
    independently, since substeps include work outside the step (evals,
    checkpointing).
    """
    step_times: dict[str, dict[str, int | None]] = {}
    substep_times: dict[str, dict[str, dict[str, float | None]]] = {}

    for current_step_num in range(1, num_steps + 1):
        start_key = f"{run_id}:{current_step_num}:start"
        finish_key = f"{run_id}:{current_step_num}:finish"

        raw_start_time = step_times_dict.get(start_key)
        raw_end_time = step_times_dict.get(finish_key)
        precise_start_time = (
            float(raw_start_time) if raw_start_time is not None else None
        )
        precise_end_time = float(raw_end_time) if raw_end_time is not None else None
        start_time = int(precise_start_time) if precise_start_time is not None else None
        end_time = int(precise_end_time) if precise_end_time is not None else None

        duration = None
        if start_time is not None and end_time is not None:
            duration = end_time - start_time

        step_times[f"{current_step_num}"] = {
            "start": start_time,
            "end": end_time,
            "duration_s": duration,
        }

        raw_step_window_start = step_times_dict.get(
            f"{run_id}:{current_step_num}:substep_start"
        )
        step_window_start = (
            float(raw_step_window_start) if raw_step_window_start is not None else None
        )
        full_step_start_time = (
            step_window_start if step_window_start is not None else precise_start_time
        )
        full_step_end_time = step_times_dict.get(
            f"{run_id}:{current_step_num}:substep_finish"
        )
        if full_step_end_time is not None:
            full_step_end_time = float(full_step_end_time)
            if step_window_start is not None and full_step_end_time < step_window_start:
                full_step_end_time = precise_end_time
        else:
            full_step_end_time = precise_end_time

        substep_times[f"{current_step_num}"] = {}
        eval_before = Substep.EVAL_BEFORE.value
        present: set[str] = set()
        recorded: list[tuple[float, int, str]] = []
        for order_idx, substep in enumerate(SUBSTEP_ORDER):
            substep_start = step_times_dict.get(
                f"{run_id}:{current_step_num}:substep:{substep}"
            )
            if substep_start is None:
                continue
            substep_start = float(substep_start)
            if (
                step_window_start is not None
                and substep_start < step_window_start
                and substep != eval_before
            ):
                continue
            if full_step_start_time is not None and substep != eval_before:
                substep_start = max(substep_start, full_step_start_time)
            if full_step_end_time is not None:
                substep_start = min(substep_start, full_step_end_time)
            present.add(substep)
            recorded.append((substep_start, order_idx, substep))
        recorded.sort()

        for idx, (substep_start, order_idx, substep) in enumerate(recorded):
            if idx + 1 < len(recorded):
                next_start, next_idx = recorded[idx + 1][0], recorded[idx + 1][1]
            else:
                next_start, next_idx = full_step_end_time, len(SUBSTEP_ORDER)

            gap = SUBSTEP_ORDER[order_idx + 1 : next_idx]
            dropped_mandatory = any(
                s not in OPTIONAL_SUBSTEPS and s not in present for s in gap
            )
            if next_start is None or dropped_mandatory:
                substep_duration = None
            else:
                substep_duration = round(max(next_start - substep_start, 0.0), 3)

            substep_times[f"{current_step_num}"][substep] = {
                "start": round(substep_start, 3),
                "duration_s": substep_duration,
            }

    return step_times, substep_times


def build_slime_app(
    *,
    training_run_id: str,
    slime: SlimeRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
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

    _caller_module, caller_script = resolve_caller_context()

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
        image = ship_callable(
            image,
            fn,
            caller_script=caller_script,
            fallback_name=fallback_name,
            set_path=set_path,
        )

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
        slime.custom_reward_post_process_function,
        fallback_name="custom_reward_post_process",
        set_path=_set_extra_config_path("custom_reward_post_process_path"),
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
    if slime.custom_reward_post_process_function is not None:
        object.__setattr__(slime, "custom_reward_post_process_function", None)

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
    checkpoints_volume_name, checkpoints_mount_path, checkpoints_volume = (
        resolve_checkpoint_volumes(
            checkpoint,
            volume_prefix=volume_prefix,
            default_mount_path=str(CHECKPOINTS_PATH),
        )
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
    tags = build_app_tags(
        framework="slime",
        model=model,
        recipe_app_tags=slime.app_tags,
        wandb=slime.wandb,
    )
    app = App(app_name, tags=tags)
    gpu_spec = f"{slime.gpu_type}:{slime.actor_num_gpus_per_node}"

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=6 * 60 * 60,
        secrets=[*hf_secrets(), *proxy_auth_secrets()],
        serialized=True,
        name="download",
    )
    def download(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        run_download_phase(
            training_run_id=training_run_id,
            phase=SlimeStatus.DOWNLOAD_MODEL.value,
            framework_status_url=framework_status_url,
            framework_status_token=framework_status_token,
            volumes=(hf_cache_volume, checkpoints_volume),
            download=model.download,
        )

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        run_prepare_dataset(dataset, data_volume, SlimeRecipe._resolve_data_paths)

    convert_nnodes = get_checkpoint_conversion_policy(slime, model=model)[0]

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
                SlimeStatus.CONVERT_MODEL.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )

        # Bridge mode loads HF weights directly into Megatron at train time.
        if getattr(slime, "megatron_to_hf_mode", None) == "bridge":
            print(
                "Bridge mode — HF weights loaded directly via AutoBridge; no conversion needed."
            )
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return None

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        save_path = str(slime.ref_load)
        current_config = _build_conversion_config(slime, model=model)
        cache_status, stored_config = _checkpoint_conversion_cache_status(
            save_path, current_config
        )
        if cache_status == "hit":
            print(f"Using existing torch_dist checkpoint at {save_path}.")
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return None

        if cache_status == "stale":
            if stored_config is None:
                print(
                    f"Checkpoint at {save_path} has missing or unreadable conversion "
                    "config metadata — reconverting."
                )
            else:
                print(
                    f"Checkpoint at {save_path} was built with different config:"
                    f"\n  stored: {stored_config}\n  current: {current_config}"
                )
        if cache_status in {"stale", "incomplete"}:
            print(f"Removing {cache_status} torch_dist checkpoint at {save_path}.")
            import shutil

            shutil.rmtree(save_path, ignore_errors=True)
            checkpoints_volume.commit()

        if slime.megatron_conversion_hf_checkpoint:
            return resolve_checkpoint_ref(slime.megatron_conversion_hf_checkpoint)
        if model.model_path:
            return str(model.model_path)

        from huggingface_hub import snapshot_download

        return snapshot_download(model.model_name, local_files_only=True)

    @app.function(
        image=image,
        gpu=gpu_spec,
        memory=slime.memory,
        cloud=slime.cloud,
        region=slime.region,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        secrets=proxy_auth_secrets() or None,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered_if(convert_nnodes > 1, convert_nnodes, gpu_type=slime.gpu_type)
    def convert_checkpoint(
        hf_path: str,
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from modal_training_gym.common.status_reporter import (
            flush as flush_status_reporter,
        )

        save_path = str(slime.ref_load)

        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            slime, model=model
        )
        node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

        import json

        current_config = _build_conversion_config(slime, model=model)

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
    # Proxy-auth tokens for any custom_rm / generate hook that calls a
    # CustomDeployment.launch() endpoint (teacher /generate, etc.).
    train_secrets.extend(proxy_auth_secrets())
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

    SUBSTEP_ORDER = [substep.value for substep in Substep]
    OPTIONAL_SUBSTEPS = {
        Substep.EVAL_BEFORE.value,
        Substep.OFFLOAD_ROLLOUT.value,
        Substep.CHECKPOINT_SAVE.value,
        Substep.OFFLOAD_TRAIN.value,
        Substep.EVAL_AFTER.value,
    }

    STEP_TIME_DICT_BATCH_SIZE = 128

    STEP_TIME_KEY_SUFFIXES = (
        "start",
        "finish",
        "substep_start",
        "substep_finish",
        *(f"substep:{s}" for s in SUBSTEP_ORDER),
    )

    async def write_step_times(
        run_id: str, num_steps: int
    ) -> tuple[
        dict[str, dict[str, float | None]],
        dict[str, dict[str, dict[str, float | None]]],
    ]:
        step_times_dict = ModalDict.from_name(
            "training-gym-step-times", create_if_missing=True
        )
        keys = [
            f"{run_id}:{step}:{suffix}"
            for step in range(1, num_steps + 1)
            for suffix in STEP_TIME_KEY_SUFFIXES
        ]
        event_times_by_key: dict[str, Any] = {}
        for offset in range(0, len(keys), STEP_TIME_DICT_BATCH_SIZE):
            batch = keys[offset : offset + STEP_TIME_DICT_BATCH_SIZE]
            values = await asyncio.gather(*(step_times_dict.get.aio(k) for k in batch))
            event_times_by_key.update(zip(batch, values))
        return aggregate_step_times(
            event_times_by_key,
            run_id,
            num_steps,
            SUBSTEP_ORDER,
            OPTIONAL_SUBSTEPS,
        )

    async def clear_step_times(run_id: str, num_steps: int) -> None:
        step_times_dict = ModalDict.from_name(
            "training-gym-step-times", create_if_missing=True
        )
        keys = [
            f"{run_id}:{step}:{suffix}"
            for step in range(1, num_steps + 1)
            for suffix in STEP_TIME_KEY_SUFFIXES
        ]
        for offset in range(0, len(keys), STEP_TIME_DICT_BATCH_SIZE):
            batch = keys[offset : offset + STEP_TIME_DICT_BATCH_SIZE]
            await asyncio.gather(*(step_times_dict.pop.aio(k, None) for k in batch))

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
        # Retries exist for transient failures (preemption/NCCL), where a retry
        # resumes from the last checkpoint. But a *deterministic* crash (esp.
        # before the first save_interval checkpoint) re-runs from scratch and
        # crashloops through every attempt — 10 wasted ~4h of a 40-GPU cluster on
        # a step-1 crash. Cap low so a persistent failure surfaces fast.
        retries=Retries(max_retries=3, initial_delay=0.0),
        single_use_containers=True,
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

        wandb_run_id = ""

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
        (
            run_record,
            wandb_run_id,
            framework_status_token,
        ) = await init_training_run_record(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url or modal_app_dashboard_url(modal_app_id),
            framework=Framework.SLIME,
            initializing_status=SlimeStatus.INITIALIZING,
            config_summary=config_summary,
            wandb_cfg=slime.wandb,
            wandb_entity=wandb_entity,
            framework_status_token=framework_status_token,
        )

        try:  # Wraps all post-setup work so any failure marks the run terminal.
            # In-flight status updates are fire-and-forget via the dashboard's
            # /api/framework-status endpoint so the training thread doesn't pay
            # the ~300ms volume-write latency on each transition. Terminal state
            # (COMPLETED/FAILED/STOPPED) still goes through
            # run_record.save(is_async=True) below to guarantee delivery
            # before the container exits.
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
                    dataset.validate_prepared(ep)

            await _set_framework_status_async(SlimeStatus.CONVERT_MODEL)
            prepare_slime_config(slime, model, tempfile.mkdtemp())

            if wandb_key := os.environ.get("WANDB_API_KEY", ""):
                if slime.wandb is not None:
                    slime.wandb.key = wandb_key

            save_root = compute_save_root(
                slime.save,
                recipe_default_save_root=str(CHECKPOINTS_PATH).rstrip("/"),
                mounted_save_root=checkpoints_mount_path,
                training_run_id=training_run_id,
            )

            original_save = slime.save
            original_load = slime.load
            original_ref_load = slime.ref_load
            original_no_load_optim = slime.no_load_optim
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

            resume_checkpoint = torch_dist_resume_checkpoint(
                save_root, is_complete=_is_complete_torch_dist_checkpoint
            )
            record_resume_checkpoint(run_record, resume_checkpoint)
            await run_record.save(is_async=True)

            if resume_checkpoint is not None:
                print(
                    f"WARNING: detected existing checkpoint in "
                    f"{resume_checkpoint['resume_checkpoint_path']}; "
                    "resuming training from last saved iteration."
                )
                object.__setattr__(slime, "load", save_root)
                # Weights-only checkpoints (``no_save_optim``) have no Adam state;
                # Megatron will KeyError on state_dict["optimizer"] unless we skip it.
                if slime.no_save_optim and not slime.no_load_optim:
                    print(
                        "WARNING: no_save_optim=True — enabling no_load_optim for resume."
                    )
                    object.__setattr__(slime, "no_load_optim", True)
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
                object.__setattr__(slime, "no_load_optim", original_no_load_optim)

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
            print(slime.gpu_allocation.summary())
            print(f"Command: {cmd}, runtime_env: {runtime_env}")

            await _set_framework_status_async(SlimeStatus.ROLLOUT_INITIALIZING)
            async with cluster.forward_dashboard() as tunnel:
                print(f"Ray dashboard: {tunnel.url}")
                result = await cluster.submit_and_tail(cmd, runtime_env=runtime_env)
                if not result.is_success:
                    run_record.error_message = result.message
                    raise RuntimeError(
                        result.message
                        or f"Ray job finished with status: {result.status}"
                    )
                print(f"Ray job completed: {result.status}")

            result = build_train_result(
                app_name=app_name,
                framework=Framework.SLIME,
                training_run_id=training_run_id,
                checkpoint_dir=save_root,
                model=model,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
                wandb_cfg=slime.wandb,
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

            step_times_read = False
            if not slime.async_mode:
                try:
                    (
                        latest_run_record.step_times,
                        latest_run_record.substep_times,
                    ) = await write_step_times(training_run_id, slime.num_rollout)
                    step_times_read = True
                except Exception as exc:
                    print(f"Failed to read step times: {exc}")

            try:
                await latest_run_record.save(is_async=True)
            except Exception as exc:
                print(f"Failed to save run record: {exc}")
            else:
                if step_times_read or slime.async_mode:
                    try:
                        await clear_step_times(training_run_id, slime.num_rollout)
                    except Exception as exc:
                        print(f"Failed to clear step times: {exc}")

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
