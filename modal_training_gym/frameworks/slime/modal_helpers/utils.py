"""Helper functions for Modal multi-node training infrastructure."""

import os
import shlex
from os import PathLike

# (attr_name_on_slime_cfg, cli_flag) — optional per-rank conversion args
_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
]


def is_local_checkpoint_ref(ref: str | PathLike) -> bool:
    """Return True when a checkpoint ref is already a mounted local path."""
    return str(ref).startswith("/")


def resolve_checkpoint_ref(
    ref: str | PathLike,
    *,
    local_files_only: bool = True,
) -> str:
    """Resolve an absolute path or Hugging Face repo ID to a local path."""
    ref_str = str(ref)
    if is_local_checkpoint_ref(ref_str):
        return ref_str

    from huggingface_hub import snapshot_download

    return snapshot_download(ref_str, local_files_only=local_files_only)


def get_checkpoint_conversion_policy(
    slime_cfg, model=None
) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion."""
    gpus_per_node = slime_cfg.actor_num_gpus_per_node
    actor_nodes = slime_cfg.actor_num_nodes
    tp = slime_cfg.tensor_model_parallel_size
    pp = getattr(slime_cfg, "pipeline_model_parallel_size", 1)

    needs_preconv = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "needs_pre_conversion", False)
    )
    ep = getattr(slime_cfg, "expert_model_parallel_size", 1) or 1
    etp = getattr(slime_cfg, "expert_tensor_parallel_size", 1) or 1

    if needs_preconv:
        # Match ALL training parallelism dims exactly so Megatron
        # doesn't attempt any re-sharding at load time (re-sharding
        # triggers BytesIO errors in dist_checkpointing).
        pp = 1
        world_size = actor_nodes * gpus_per_node
    else:
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
        conv_tp = tp
        if needs_preconv:
            pp = 1
        if conv_tp > 1 or pp > 1:
            extra_args += [
                f"--tensor-model-parallel-size {conv_tp}",
                f"--pipeline-model-parallel-size {pp}",
            ]
        if needs_preconv:
            extra_args += [
                f"--expert-model-parallel-size {ep}",
                f"--expert-tensor-parallel-size {etp}",
            ]
        for attr, flag in _CONVERSION_EXTRA_ARGS:
            if x := getattr(slime_cfg, attr, None):
                extra_args.append(f"--{flag} {x}")

        if model and getattr(model, "architecture", None):
            arch = model.architecture
            _arch_fields = [
                ("num_layers", "num-layers"),
                ("hidden_size", "hidden-size"),
                ("ffn_hidden_size", "ffn-hidden-size"),
                ("num_attention_heads", "num-attention-heads"),
                ("num_query_groups", "num-query-groups"),
                ("kv_channels", "kv-channels"),
                ("vocab_size", "vocab-size"),
                ("norm_epsilon", "norm-epsilon"),
                ("rotary_base", "rotary-base"),
            ]
            for attr, flag in _arch_fields:
                val = getattr(arch, attr, 0)
                if val:
                    extra_args.append(f"--{flag} {val}")
            if arch.group_query_attention:
                extra_args.append("--group-query-attention")
            if arch.swiglu:
                extra_args.append("--swiglu")
            if arch.disable_bias_linear:
                extra_args.append("--disable-bias-linear")
            if arch.qk_layernorm:
                extra_args.append("--qk-layernorm")
            if arch.untie_embeddings_and_output_weights:
                extra_args.append("--untie-embeddings-and-output-weights")
            if arch.normalization and arch.normalization != "LayerNorm":
                extra_args.append(f"--normalization {arch.normalization}")
            if arch.num_experts:
                extra_args.append(f"--num-experts {arch.num_experts}")
            if arch.moe_ffn_hidden_size:
                extra_args.append(f"--moe-ffn-hidden-size {arch.moe_ffn_hidden_size}")
            if arch.moe_shared_expert_intermediate_size:
                extra_args.append(
                    f"--moe-shared-expert-intermediate-size {arch.moe_shared_expert_intermediate_size}"
                )
            if arch.moe_grouped_gemm:
                extra_args.append("--moe-grouped-gemm")
            if arch.moe_shared_expert_gate:
                extra_args.append("--moe-shared-expert-gate")
            if arch.moe_router_topk:
                extra_args.append(f"--moe-router-topk {arch.moe_router_topk}")
            if arch.moe_router_score_function:
                extra_args.append(
                    f"--moe-router-score-function {arch.moe_router_score_function}"
                )
            if arch.moe_token_drop_policy:
                extra_args.append(
                    f"--moe-token-drop-policy {arch.moe_token_drop_policy}"
                )
            if arch.moe_router_dtype:
                extra_args.append(f"--moe-router-dtype {arch.moe_router_dtype}")
            if arch.moe_permute_fusion:
                extra_args.append("--moe-permute-fusion")
            if arch.moe_aux_loss_coeff is not None:
                extra_args.append(f"--moe-aux-loss-coeff {arch.moe_aux_loss_coeff}")
            if arch.megatron_spec:
                extra_args.append(f"--spec {' '.join(arch.megatron_spec)}")
            if arch.apply_layernorm_1p:
                extra_args.append("--apply-layernorm-1p")
            if arch.use_gated_attention:
                extra_args.append("--use-gated-attention")
            if arch.attention_output_gate:
                extra_args.append("--attention-output-gate")
            if arch.use_rotary_position_embeddings:
                extra_args.append("--use-rotary-position-embeddings")
                extra_args.append("--position-embedding-type rope")
                if arch.rotary_percent != 1.0:
                    extra_args.append(f"--rotary-percent {arch.rotary_percent}")

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


def prepare_slime_config(slime_cfg, model, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    import yaml

    from modal_training_gym.train_recipes.slime_recipe.recipe import YAML_CONFIG_FIELDS

    if (
        model
        and not model.model_path
        and model.model_name
        and not str(model.model_name).startswith("/")
    ):
        model.model_path = resolve_checkpoint_ref(model.model_name)

    for attr in ("load", "ref_load", "critic_load"):
        if (val := getattr(slime_cfg, attr, None)) and not str(val).startswith("/"):
            object.__setattr__(slime_cfg, attr, resolve_checkpoint_ref(val))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(slime_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} → {path}")
            object.__setattr__(slime_cfg, field, path)


def build_train_cmd(slime_cfg, slime_root: str, model=None, dataset=None) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    train_script = (
        f"{slime_root}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    )
    args = shlex.join(slime_cfg.cli_args(dataset=dataset, model=model))
    if getattr(slime_cfg, "slime_model_script", ""):
        inner = (
            f"source {slime_root}/{slime_cfg.slime_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {args}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {args}"
