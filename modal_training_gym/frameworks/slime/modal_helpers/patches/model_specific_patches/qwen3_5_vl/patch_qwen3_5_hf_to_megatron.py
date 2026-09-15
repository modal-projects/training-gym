"""image: nightly-dev-20260722a
commit: https://github.com/THUDM/slime/commit/f655e13
file: slime/utils/arguments.py, slime/backends/megatron_utils/checkpoint.py,
slime/backends/megatron_utils/megatron_to_hf/__init__.py,
slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py,
slime/backends/megatron_utils/update_weight/common.py
"""

from __future__ import annotations

from pathlib import Path

SLIME = Path("/root/slime/slime")
MARKER = "PATCHED_QWEN3_5_HF_TO_MEGATRON"

# validate_args: models the vendored loader supports take slime's direct path,
# so weight sync and HF export use global names instead of megatron-bridge.
ARGUMENTS = SLIME / "utils" / "arguments.py"
ARGUMENTS_OLD = """\
    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
"""
ARGUMENTS_NEW = f"""\
    if args.megatron_to_hf_mode == "bridge":  # {MARKER}
        from slime.backends.megatron_utils.hf_to_megatron import supports_hf_weight_loading

        if supports_hf_weight_loading(args.hf_checkpoint):
            args.megatron_to_hf_mode = "raw"
    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
"""

CHECKPOINT = SLIME / "backends" / "megatron_utils" / "checkpoint.py"
CHECKPOINT_OLD = """\
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)
"""
CHECKPOINT_NEW = f"""\
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={{load_path}})")
    from slime.backends.megatron_utils.hf_to_megatron import load_hf_weights, supports_hf_weight_loading

    if supports_hf_weight_loading(load_path):  # {MARKER}
        load_hf_weights(args, ddp_model, load_path)
    else:
        assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
        from megatron.bridge import AutoBridge

        import slime_plugins.megatron_bridge  # noqa: F401

        with megatron_bridge_utils.patch_megatron_model(ddp_model):
            bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
                AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
            )
            bridge.load_hf_weights(ddp_model)
"""

# Export: the plugin's Transformers ViT already carries HF names.
MEGATRON_TO_HF = (
    SLIME / "backends" / "megatron_utils" / "megatron_to_hf" / "__init__.py"
)
MEGATRON_TO_HF_OLD = """\
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)
"""
MEGATRON_TO_HF_NEW = f"""\
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    hf_name = name  # {MARKER}
    while hf_name.startswith("module."):
        hf_name = hf_name.removeprefix("module.")
    if hf_name.startswith("model.visual."):
        return [(hf_name, param)]

    param = remove_padding(name, param, args.vocab_size)
"""

QWEN3_5_EXPORT = SLIME / "backends" / "megatron_utils" / "megatron_to_hf" / "qwen3_5.py"
QWEN3_5_EXPORT_OLD = """\
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.
    \"\"\"
    # Handle MTP layers
"""
QWEN3_5_EXPORT_NEW = f"""\
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.
    \"\"\"
    if name.startswith("module.module.language_model."):  # {MARKER}
        name = "module.module." + name.removeprefix("module.module.language_model.")

    # Handle MTP layers
"""

UPDATE_WEIGHT_COMMON = (
    SLIME / "backends" / "megatron_utils" / "update_weight" / "common.py"
)
UPDATE_WEIGHT_COMMON_EDITS = (
    (
        """\
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\\.module\\.decoder\\.layers\\.(\\d+)\\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\\.module\\.mtp\\.layers\\.(\\d+)\\.(.+)"
""",
        f"""\
            if not name.startswith("module.module."):
                name = "module." + name
            prefix = "module.module.language_model." if ".language_model." in name else "module.module."  # {MARKER}

            decoder_layers_pattern = r"module\\.module\\.(?:language_model\\.)?decoder\\.layers\\.(\\d+)\\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\\.module\\.(?:language_model\\.)?mtp\\.layers\\.(\\d+)\\.(.+)"
""",
    ),
    (
        '                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.{param_type}{expert_idx}", param\n',
        '                yield f"{prefix}mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.{param_type}{expert_idx}", param\n',
    ),
    (
        '                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param\n',
        '                yield f"{prefix}decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param\n',
    ),
    (
        '                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param\n',
        '                yield f"{prefix}decoder.layers.{layer_idx}.{rest}", param\n',
    ),
    (
        """\
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\\.module\\.decoder\\.layers\\.(\\d+)\\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
""",
        """\
            if not name.startswith("module.module."):
                name = "module." + name
            prefix = "module.module.language_model." if ".language_model." in name else "module.module."

            decoder_layers_pattern = r"module\\.module\\.(?:language_model\\.)?decoder\\.layers\\.(\\d+)\\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
""",
    ),
    (
        '                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer\n',
        '                yield f"{prefix}decoder.layers.{layer_idx}.{rest}", buffer\n',
    ),
)

EDITS: dict[Path, tuple[tuple[str, str], ...]] = {
    ARGUMENTS: ((ARGUMENTS_OLD, ARGUMENTS_NEW),),
    CHECKPOINT: ((CHECKPOINT_OLD, CHECKPOINT_NEW),),
    MEGATRON_TO_HF: ((MEGATRON_TO_HF_OLD, MEGATRON_TO_HF_NEW),),
    QWEN3_5_EXPORT: ((QWEN3_5_EXPORT_OLD, QWEN3_5_EXPORT_NEW),),
    UPDATE_WEIGHT_COMMON: UPDATE_WEIGHT_COMMON_EDITS,
}


def _patch_file(path: Path, edits: tuple[tuple[str, str], ...]) -> None:
    source = path.read_text()
    if MARKER in source:
        print(f"{path} already patched for qwen3_5 hf_to_megatron")
        return
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(
                f"qwen3_5 hf_to_megatron anchor matched {source.count(old)} times in {path}:\n{old}"
            )
        source = source.replace(old, new, 1)
    path.write_text(source)
    print(f"Patched {path} for qwen3_5 hf_to_megatron")


def main() -> None:
    for path, edits in EDITS.items():
        _patch_file(path, edits)


if __name__ == "__main__":
    main()
