"""Checkpoint preparation for a stitch run: BF16 masters + the served baseline.

Vendored from the stitch cookbook (``cookbook/miles_disagg/prep.py``). A
quantized run needs *two* checkpoints, and the pair is the whole point:

- BF16 masters (``train.bf16_checkpoint_path``) — what megatron-bridge loads as
  the trainer's reference weights.
- the served baseline (``train.hf_checkpoint``) — what every replica boots from
  and what each sparse XOR delta is applied against.

The baseline must be byte-identical to what the trainer's live export produces
from unchanged weights, so it is built here with the *same* miles quantizer under
the *same* ``NVTE_*`` contract (``train.prep_env``) that the trainer exports
with — a mismatched quantizer turns every delta into a full-tensor rewrite at
best, and a checksum failure at worst.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from modal_training_gym.train_recipes.stitch_recipe.pins import MILES_ROOT


def prepare_checkpoints(train: Any, checkpoint_volume: Any) -> None:
    """Materialize the trainer's BF16 masters and the pool's served baseline.

    Idempotent and resumable: each directory is built into a ``.partial`` sibling
    and atomically renamed, so an interrupted (or preempted) prep never leaves a
    half-built checkpoint that the reuse check mistakes for a complete one.
    """
    checkpoint_volume.reload()
    os.environ.update({str(k): str(v) for k, v in (train.prep_env or {}).items()})
    source = train.source_hf_checkpoint or train.hf_checkpoint
    served_dir = str(train.hf_checkpoint)
    bf16_dir = str(train.bf16_checkpoint_path)
    if train.served_checkpoint_format == "bf16":
        # A BF16 source *is* the masters, so there is nothing to convert: either
        # the replicas serve the HF snapshot directly, or (when the recipe asks
        # for a Volume-resident copy) the same tree materialized on the Volume.
        if bf16_dir:
            _staged(
                bf16_dir, lambda out: _materialize_bf16_masters(_snapshot(source), out)
            )
            checkpoint_volume.commit()
        served = bf16_dir or _snapshot(source)
        print(f"Prepared served_base={served} (bf16 source)")
        return

    _staged(bf16_dir, lambda out: _materialize_bf16_masters(_snapshot(source), out))
    _staged(served_dir, lambda out: _convert_to_nvfp4(train, bf16_dir, out))
    checkpoint_volume.commit()
    print(f"Prepared masters={bf16_dir} served_base={served_dir}")


def _snapshot(model_ref: str) -> str:
    if str(model_ref).startswith("/"):
        return str(model_ref)
    from huggingface_hub import snapshot_download

    return snapshot_download(model_ref)


def _materialize_bf16_masters(src: str, out: str) -> None:
    """Copy the HF snapshot into the checkpoints Volume as the trainer's masters.

    The HF cache stores blobs as symlinks, so this dereferences into real files
    (the old ``cp -aL``) — the trainer and the converter both read the tree
    directly, and a symlink into a cache the container may not mount would dangle.
    """
    _copy_tree("bf16 masters", src, out)
    _strip_stale_quant_config(os.path.join(out, "config.json"))


def _convert_to_nvfp4(train: Any, bf16_dir: str, out: str) -> None:
    """Quantize the BF16 masters with miles' TE-direct NVFP4 converter.

    The BF16 carve-outs must match the trainer's
    ``--num-layers-at-start/end-in-bf16``, or the served baseline's layout differs
    from the export's and the first delta cannot apply.
    """
    carveouts: list[str] = []
    for attr, flag in (
        ("num_layers_at_start_in_bf16", "--num-layers-at-start-in-bf16"),
        ("num_layers_at_end_in_bf16", "--num-layers-at-end-in-bf16"),
    ):
        if (n := getattr(train, attr, None)) is not None:
            carveouts += [flag, str(n)]
    if layers := getattr(train, "extra_high_precision_layers_hf", None):
        carveouts += ["--extra-high-precision-layers-hf", *[str(x) for x in layers]]
    print("building nvfp4 served base from bf16 masters (GPU)...", flush=True)
    subprocess.run(
        [
            "python",
            f"{MILES_ROOT}/tools/convert_hf_to_nvfp4.py",
            "--model-dir",
            bf16_dir,
            "--save-dir",
            out,
            *carveouts,
        ],
        check=True,
    )


# A single cache→Volume stream is backend-fetch bound; ~8 parallel streams recover ~5x.
_COPY_WORKERS = int(os.environ.get("PREP_COPY_WORKERS", "8"))


def _copy_tree(label: str, src: str, dst: str) -> None:
    files = [p for p in Path(src).rglob("*") if p.is_file()]  # follows symlinks
    total_gb = sum(p.stat().st_size for p in files) / 1e9

    def copy_one(p: Path) -> None:
        out = Path(dst) / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)  # dereferences the HF cache's blob symlinks

    os.makedirs(dst, exist_ok=True)
    print(f"copying {label}: {total_gb:.0f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=min(_COPY_WORKERS, len(files) or 1)) as pool:
        list(pool.map(copy_one, files))
    print(f"copied {label}", flush=True)


def _staged(final_dir: str, build: Callable[[str], None]) -> None:
    if os.path.isdir(final_dir) and os.listdir(final_dir):
        print(f"reusing existing {final_dir}")
        return
    partial = f"{final_dir}.partial"
    subprocess.run(["rm", "-rf", partial], check=True)
    os.makedirs(partial, exist_ok=True)
    build(partial)
    os.rename(partial, final_dir)


def _strip_stale_quant_config(config_path: str) -> None:
    """Drop any ``quantization_config`` from the masters' HF config, so they don't
    claim the source's quant scheme."""
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    removed = bool(cfg.pop("quantization_config", None))
    if isinstance(cfg.get("text_config"), dict):
        removed = bool(cfg["text_config"].pop("quantization_config", None)) or removed
    if removed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"stripped stale quantization_config from {config_path}")
