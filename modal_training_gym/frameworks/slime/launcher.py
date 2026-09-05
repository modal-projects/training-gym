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
from collections.abc import Callable
from modal import App, Image, Secret, Volume, Retries

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
    drop_materialized_config_key,
    serialize_recipe_params,
    timing_debug_env,
)
from modal_training_gym.common.metrics import (
    apply_metric_image,
    metric_metadata,
    metric_runtime_env,
    metric_secrets,
    preflight_metric,
)
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.common.status import SlimeStatus

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
from modal_training_gym.common.patches import _MEGATRON_PATCHES, encode_patch
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.framework import Framework


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


SLIME_ROOT = "/root/slime"
# Pin by digest to prevent mutable-tag drift.  Tag: nightly-dev-20260722a
SLIME_IMAGE = "slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342"
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"

_SLIME_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_VALIDATION_B64 = encode_patch("patch_validation", _MEGATRON_PATCHES)
_PATCH_MEGATRON_BRIDGE_B64 = encode_patch("patch_megatron_bridge", _SLIME_PATCHES)
_PATCH_TORCH_LOAD_B64 = encode_patch("patch_torch_load", _MEGATRON_PATCHES)
_PATCH_GLOBAL_PLAN_B64 = encode_patch("patch_global_plan", _SLIME_PATCHES)
_PATCH_CHECKPOINT_SAVE_B64 = encode_patch("patch_checkpoint_save", _MEGATRON_PATCHES)
_PATCH_ADVANTAGES_B64 = encode_patch("patch_advantages", _SLIME_PATCHES)
_PATCH_BRIDGE_NONE_TASK_B64 = encode_patch("patch_bridge_none_task", _SLIME_PATCHES)
_PATCH_GDN_PACKED_SEQ_B64 = encode_patch("patch_gdn_packed_seq", _MEGATRON_PATCHES)
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
_PATCH_SUBSTEP_TIMING_B64 = encode_patch("patch_substep_timing", _SLIME_PATCHES)
_PATCH_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", _SLIME_PATCHES)
_PATCH_LOG_ELIDE_B64 = encode_patch("patch_log_elide", _SLIME_PATCHES)
# Backport of NVIDIA/Megatron-LM #3845: dequantize quantized CUDA tensors in the
# async dist-checkpoint writer before serialization. slime pins a pre-#3845
# Megatron, so FP8/TE _extra_state tensors otherwise crash the torch_dist save
# with inline_container.cc "unexpected pos" (e.g. the GLM-5.2 convert). No-op for
# non-quantized tensors, so safe for every image.
_PATCH_DIST_CKPT_QUANTIZED_B64 = encode_patch(
    "patch_dist_ckpt_quantized", _MEGATRON_PATCHES
)
_PATCH_DIST_CKPT_NOFORK_B64 = encode_patch("patch_dist_ckpt_nofork", _MEGATRON_PATCHES)
# OPD / multi-turn: zero-std metrics must skip non-numeric rewards (dict/None).
_PATCH_ZERO_STD_METRICS_B64 = encode_patch("patch_zero_std_metrics", _SLIME_PATCHES)
_PATCH_SGLANG_PARALLEL_ALIASES_B64 = encode_patch(
    "patch_sglang_parallel_aliases", _SLIME_PATCHES
)

# Patches targeting /root/slime* — a git overlay replaces that directory, so
# these are skipped in the base image when an overlay is configured and applied
# after the replacement instead.
_SLIME_ROOT_PATCHES_B64 = (
    _PATCH_MEGATRON_BRIDGE_B64,
    _PATCH_ADVANTAGES_B64,
    _PATCH_STOP_TOKEN_DIAG_B64,
    _PATCH_QWEN3_ASR_EXPORT_B64,
    _PATCH_QWEN3_VL_EXPORT_B64,
    _PATCH_QWEN3_VL_TORCH_DIST_B64,
    _PATCH_ROLLOUT_STATUS_B64,
    _PATCH_ADVANTAGE_DIST_B64,
    _PATCH_ZERO_STD_METRICS_B64,
    _PATCH_SGLANG_PARALLEL_ALIASES_B64,
    _PATCH_SUBSTEP_TIMING_B64,
)

# Patches targeting Megatron-LM or site-packages — survive a git overlay.
_SLIME_EXTERNAL_PATCHES_B64 = (
    _PATCH_BRIDGE_NONE_TASK_B64,
    _PATCH_LOG_ELIDE_B64,
    _PATCH_DIST_CKPT_QUANTIZED_B64,
    _PATCH_DIST_CKPT_NOFORK_B64,
)


def _patch_commands(patches: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"echo {patch} | base64 -d | python3" for patch in patches)


def _build_slime_base_image(*, apply_root_patches: bool = True) -> "Image":
    patches = _SLIME_EXTERNAL_PATCHES_B64
    if apply_root_patches:
        patches = patches + _SLIME_ROOT_PATCHES_B64
    return (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands("rm -rf /root/.cache/huggingface", *_patch_commands(patches))
    )


def _slime_git_overlay_command(repository: str, revision: str) -> str:
    """Build the reproducible image command for a fork source overlay."""
    repo = shlex.quote(repository)
    sha = shlex.quote(revision)
    checkout = "/tmp/training-gym-slime"
    return (
        "set -eux; "
        "command -v git >/dev/null; "
        f"rm -rf {checkout}; "
        f"git init {checkout}; "
        f"git -C {checkout} remote add origin {repo}; "
        f"git -C {checkout} fetch --depth=1 origin {sha}; "
        f"git -C {checkout} checkout --detach FETCH_HEAD; "
        f'test "$(git -C {checkout} rev-parse HEAD)" = {sha}; '
        f"rm -rf {checkout}/.git {SLIME_ROOT}; "
        f"mv {checkout} {SLIME_ROOT}"
    )


def _overlay_slime_source(image: "Image", slime: SlimeRecipe) -> "Image":
    if slime.local_slime:
        # Preserve the local dev overlay's existing semantics: use the checkout
        # exactly as supplied rather than requiring it to match patch anchors.
        return image.add_local_dir(
            slime.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if not (slime.slime_git_repository and slime.slime_git_revision):
        return image

    image = image.run_commands(
        _slime_git_overlay_command(slime.slime_git_repository, slime.slime_git_revision)
    )

    # The pinned source replaced /root/slime after the base-image patches ran.
    # Fail if required patches no longer apply rather than run an incompatible
    # fork with silently missing Training Gym behavior.
    return image.run_commands(*_patch_commands(_SLIME_ROOT_PATCHES_B64))


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


_PIPELINE_SPLIT_FLAGS = (
    "--decoder-first-pipeline-num-layers",
    "--decoder-last-pipeline-num-layers",
)


def _conversion_config_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Whether a recorded conversion still describes the current layout.

    The record stores the emitted ``extra_args``, so a checkpoint converted before
    the pipeline-split flags stopped being emitted at conversion PP1 would otherwise
    read as stale and be re-converted for nothing. Dropping those flags is tolerated;
    changing their values is not, since at PP>1 they define the split.
    """
    if stored == current:
        return True
    stored_rest, current_rest = dict(stored), dict(current)
    stored_args = stored_rest.pop("extra_args", None)
    current_args = current_rest.pop("extra_args", None)
    if stored_rest != current_rest:
        return False
    if not isinstance(stored_args, list) or not isinstance(current_args, list):
        return False
    if [a for a in stored_args if not a.startswith(_PIPELINE_SPLIT_FLAGS)] != [
        a for a in current_args if not a.startswith(_PIPELINE_SPLIT_FLAGS)
    ]:
        return False
    return {a for a in current_args if a.startswith(_PIPELINE_SPLIT_FLAGS)} <= {
        a for a in stored_args if a.startswith(_PIPELINE_SPLIT_FLAGS)
    }


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
    if not _conversion_config_matches(stored_config, current_config):
        return "stale", stored_config
    return "hit", stored_config


_serialize_slime_params = serialize_recipe_params


def _preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Backward-compatible wrapper for the W&B preflight helper."""
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
                "use_dynamic_batch_size=False — or use Qwen3_ASR_1_7B_Recipe, which sets "
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
    # When a git overlay will replace /root/slime, skip root patches here so they
    # run once after the replacement instead of being applied and then discarded.
    _needs_git_overlay = bool(slime.slime_git_repository and slime.slime_git_revision)
    image = _build_slime_base_image(apply_root_patches=not _needs_git_overlay)

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

    image = _overlay_slime_source(image, slime)

    if slime.image_run_commands:
        image = image.run_commands(*slime.image_run_commands)
    if slime.image_env:
        image = image.env(slime.image_env)

    image = apply_metric_image(image, slime.metrics)
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
        metrics=slime.metrics,
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
        cpu=slime.cpu,
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

        spec = importlib.util.find_spec(
            "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
        )
        convert_script = spec.origin if spec is not None else None
        if not convert_script:
            raise RuntimeError(
                "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist not found"
            )
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
        if any(arg.startswith("--pipeline-model-parallel-size ") for arg in extra_args):
            env["SKIP_PP_AUTOINFLATE"] = "1"
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
    if slime.metrics is not None:
        train_secrets.extend(metric_secrets(slime.metrics))
        if (
            slime.metrics.provider == "trackio"
            and getattr(slime.metrics, "modal_secret_name", "") == "huggingface-secret"
        ):
            train_secrets.extend(hf_secrets())
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

    @app.function(
        image=train_image,
        gpu=gpu_spec,
        memory=slime.memory,
        cpu=slime.cpu,
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
        retries=Retries(max_retries=slime.max_retries, initial_delay=0.0),
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

        # Fail fast on tracker access before the framework starts training.
        metric_entity = preflight_metric(slime.metrics)

        metric_run_id = ""

        print(f"Training run id: {training_run_id}")
        config_summary: dict = {
            "model": {"model_name": model.model_name} if model else {},
            "recipe": _serialize_slime_params(slime, dataset=dataset, model=model),
            "metrics": metric_metadata(
                slime.metrics,
                entity=metric_entity,
                run_id=metric_run_id,
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
            metric_run_id,
            framework_status_token,
        ) = await init_training_run_record(
            training_run_id=training_run_id,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url or modal_app_dashboard_url(modal_app_id),
            framework=Framework.SLIME,
            initializing_status=SlimeStatus.INITIALIZING,
            config_summary=config_summary,
            metric_cfg=slime.metrics,
            metric_entity=metric_entity,
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
                if isinstance(slime.metrics, WandbConfig):
                    slime.metrics.key = wandb_key

            save_root = compute_save_root(
                slime.save,
                recipe_default_save_root=str(CHECKPOINTS_PATH).rstrip("/"),
                mounted_save_root=checkpoints_mount_path,
                training_run_id=training_run_id,
            )

            original_save = slime.save
            original_load = slime.load
            original_start_rollout_id = slime.start_rollout_id
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
                resume_from_iteration = resume_checkpoint.get("resume_from_iteration")
                _validate_resume_checkpoint(resume_from_iteration, slime.num_rollout)
                print(
                    f"WARNING: detected existing checkpoint in "
                    f"{resume_checkpoint['resume_checkpoint_path']}; "
                    "resuming training from last saved iteration."
                )
                object.__setattr__(slime, "load", save_root)
                # Continue from the iteration stored in the run's own checkpoint,
                # even for runs launched with an explicit start_rollout_id.
                object.__setattr__(slime, "start_rollout_id", None)
                drop_materialized_config_key(slime, "start_rollout_id")
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
                object.__setattr__(slime, "start_rollout_id", original_start_rollout_id)
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

            runtime_env = {
                "env_vars": {
                    "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                    "MASTER_ADDR": cluster.head_addr,
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
                    "TRAINING_GYM_SUBSTEP_TIMING": slime.substep_timing,
                    **metric_runtime_env(
                        slime.metrics,
                        run_id=metric_run_id,
                        entity=metric_entity,
                    ),
                    **slime.environment,
                    **timing_debug_env(),
                    # After **slime.environment: a recipe's env dict must not be
                    # able to reassign the run's identity, which keys metrics,
                    # the run record, and the framework status callbacks.
                    "TRAINING_GYM_TRAINING_RUN_ID": training_run_id,
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
                    message = (
                        result.message
                        or f"Ray job finished with status: {result.status}"
                    )
                    error = RuntimeError(
                        f"{message} (training_run_id={training_run_id})"
                    )
                    error.training_run_id = training_run_id  # pyright: ignore[reportAttributeAccessIssue]  # exception metadata is consumed by downstream callers
                    run_record.error_message = str(error)
                    raise error
                print(f"Ray job completed: {result.status}")

            result = build_train_result(
                app_name=app_name,
                framework=Framework.SLIME,
                training_run_id=training_run_id,
                checkpoint_dir=save_root,
                model=model,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
                metric_cfg=slime.metrics,
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
            except Exception as exc:
                print(f"Failed to save run record: {exc}")

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    return app
