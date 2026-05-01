"""Helper functions for Modal multi-node training infrastructure."""

import os
import shlex

# (attr_name_on_slime_cfg, cli_flag) — optional per-rank conversion args
_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
]


def get_checkpoint_conversion_policy(slime_cfg) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion."""
    preset = slime_cfg.preset
    gpus_per_node = preset.actor_num_gpus_per_node
    actor_nodes = preset.actor_num_nodes
    tp = preset.tensor_model_parallel_size
    pp = getattr(slime_cfg, "pipeline_model_parallel_size", 1)

    world_size = tp * pp if (tp > 1 or pp > 1) else gpus_per_node
    max_world_size = actor_nodes * gpus_per_node
    if world_size > max_world_size:
        raise ValueError(
            f"checkpoint conversion world_size={world_size} exceeds actor cluster capacity "
            f"{actor_nodes}x{gpus_per_node}={max_world_size}"
        )

    for num_nodes in range(1, actor_nodes + 1):
        if world_size % num_nodes != 0:
            continue
        nproc_per_node = world_size // num_nodes
        if nproc_per_node > gpus_per_node:
            continue

        extra_args: list[str] = []
        if tp > 1 or pp > 1:
            extra_args += [
                f"--tensor-model-parallel-size {tp}",
                f"--pipeline-model-parallel-size {pp}",
            ]
        for attr, flag in _CONVERSION_EXTRA_ARGS:
            if x := getattr(slime_cfg, attr, None):
                extra_args.append(f"--{flag} {x}")

        return num_nodes, nproc_per_node, extra_args

    raise ValueError(
        f"cannot find checkpoint conversion layout for world_size={world_size} "
        f"with actor_num_nodes={actor_nodes}, actor_num_gpus_per_node={gpus_per_node}"
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


def prepare_slime_config(slime_cfg, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    from huggingface_hub import snapshot_download
    import yaml

    from ..config import YAML_CONFIG_FIELDS

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(slime_cfg, attr, None)) and not str(val).startswith("/"):
            setattr(slime_cfg, attr, snapshot_download(val, local_files_only=True))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(slime_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} → {path}")
            setattr(slime_cfg, field, path)


def build_train_cmd(slime_cfg, slime_root: str) -> str:
    """Build the Ray job entrypoint."""
    train_script = (
        f"{slime_root}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    )
    return f"python3 {train_script} {shlex.join(slime_cfg.cli_args())}"
