"""Helper functions for Modal multi-node training infrastructure.

Shared implementations live in :mod:`modal_training_gym.common.launcher_utils`;
this module keeps slime's parametrization and the historical import path.
"""

from pathlib import Path
from typing import Any

from modal_training_gym.common.launcher_utils import (
    build_train_cmd as _build_train_cmd,
    get_checkpoint_conversion_policy as _get_checkpoint_conversion_policy,
    is_local_checkpoint_ref as is_local_checkpoint_ref,
    prepare_launch_config as _prepare_launch_config,
    resolve_checkpoint_ref as resolve_checkpoint_ref,
)
from modal_training_gym.common.patches import encode_patch

SLIME_ROOT = "/root/slime"


def _with_qwen35_vl_plugin(image: Any) -> Any:
    plugin = Path(__file__).parent / "patches" / "model_specific_patches" / "qwen3_5_vl"
    models = f"{SLIME_ROOT}/slime_plugins/models"
    for name in ("qwen3_5_vl.py", "qwen3_5_vl_utils.py"):
        image = image.add_local_file(
            str(plugin / name),
            remote_path=f"{models}/{name}",
            copy=True,
        )
    loader = f"{SLIME_ROOT}/slime/backends/megatron_utils/hf_to_megatron"
    for name in ("__init__.py", "common.py", "qwen3_5.py"):
        image = image.add_local_file(
            str(plugin / "hf_to_megatron" / name),
            remote_path=f"{loader}/{name}",
            copy=True,
        )
    patch_b64 = encode_patch("patch_qwen3_5_hf_to_megatron", plugin)
    return image.run_commands(f"echo {patch_b64} | base64 -d | python3")


def get_checkpoint_conversion_policy(
    slime_cfg, model=None
) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion."""
    return _get_checkpoint_conversion_policy(
        slime_cfg,
        model=model,
        single_rank_mtp=True,
        extended_arch_args=True,
        arch_args_model_script_attr="slime_model_script",
    )


def get_modal_cluster_context(n_nodes: int) -> tuple[int, str, str, int]:
    """Return (rank, master_addr, my_ip, n_nodes) for the current Modal cluster."""
    if n_nodes == 1:
        return 0, "127.0.0.1", "127.0.0.1", 1

    import modal.experimental

    info = modal.experimental.get_cluster_info()
    actual_nodes = len(info.container_ipv4_ips)
    if actual_nodes != n_nodes:
        raise RuntimeError(
            f"cluster size mismatch: expected {n_nodes} node(s), got {actual_nodes}"
        )
    return (
        info.rank,
        info.container_ipv4_ips[0],
        info.container_ipv4_ips[info.rank],
        actual_nodes,
    )


def prepare_slime_config(slime_cfg, model, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    from modal_training_gym.train_recipes.slime_recipe.recipe import YAML_CONFIG_FIELDS

    _prepare_launch_config(
        slime_cfg, model, tmpdir, yaml_config_fields=YAML_CONFIG_FIELDS
    )


def build_train_cmd(
    slime_cfg,
    slime_root: str,
    model=None,
    dataset=None,
    eval_dataset=None,
    dataset_path=None,
    eval_dataset_path=None,
) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    return _build_train_cmd(
        slime_cfg,
        slime_root,
        model=model,
        dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        model_script_attr="slime_model_script",
    )
