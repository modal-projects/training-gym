"""Prepare SWE-rebench tasks and train/eval subsets for the coding tutorial.

Run with:
uv run -m tutorials.coding_agent.dataset prepare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import modal

from modal_training_gym.common import hf_secrets
from modal_training_gym.common.dataset_sampling import sample_rows, split_rows
from modal_training_gym.frameworks.slime.launcher import (
    SLIME_IMAGE,
    _slime_git_overlay_command,
)
from modal_training_gym.train_recipes.base import DATA_PATH
from modal_training_gym.train_recipes.slime_recipe import (
    Qwen3_6_27B_Recipe_Agentic,
)

# Fraction of tasks that go to eval. Remainder go to train.
EVAL_SPLIT_FRACTION = 0.2
# Subset for smoke tests
EVAL_SPLIT_SIZES = (4,)
TRAIN_SPLIT_SIZES = (4, 100, 300, 1000)
SPLIT_SEED = 0
# Train keeps at least this many task groups of each language, so moving a
# group to eval never leaves train without that language.
MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE = 2
MIXED_CRITERION = "fully_gradeable_and_0_lt_solved_lt_n_samples"
SOURCE_COLUMNS = ("repo", "language", "license", "created_at")
DEFAULT_MIXED_RECIPE_SLUG = "qwen3-6-27b-agentic"


HF_DATASET = "nebius/SWE-rebench-V2"
DATASET_ROOT = "swe_rebench_v2"


def data_volume_name(recipe: Qwen3_6_27B_Recipe_Agentic) -> str:
    return recipe.data_volume_name or f"slime-{type(recipe).__name__.lower()}-data"


def dataset_root_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"dataset root must be a single directory name, got {value!r}")
    return value


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = (row.get("metadata") or {}).get("source") or {}
    return value if isinstance(value, dict) else {}


def language(row: dict[str, Any]) -> str:
    metadata = source_metadata(row)
    return str(metadata.get("language_bucket") or metadata.get("language") or "?")


def task_group(row: dict[str, Any]) -> str:
    repo = source_metadata(row).get("repo")
    if not repo:
        raise ValueError("converted row is missing metadata.source.repo")
    return str(repo)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(
        path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_partitions(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, int]:
    train_rows, eval_rows = split_rows(
        rows,
        eval_fraction=EVAL_SPLIT_FRACTION,
        seed=seed,
        group_key=task_group,
        stratify_key=language,
        min_train_groups=MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE,
    )
    outputs = {"eval": eval_rows, "train-full": train_rows}
    for prefix, pool, sizes in (
        ("eval", eval_rows, EVAL_SPLIT_SIZES),
        ("train", train_rows, TRAIN_SPLIT_SIZES),
    ):
        pool_size = len(pool)
        for size in sizes:
            if size > pool_size:
                (root / f"{prefix}-{size}.jsonl").unlink(missing_ok=True)
                print(f"[swe] skipping {prefix}-{size}: only {pool_size} {prefix} rows")
                continue
            outputs[f"{prefix}-{size}"] = sample_rows(
                pool, size, seed=seed, stratify_key=language
            )
    staged: list[tuple[Path, Path]] = []
    for name, subset in outputs.items():
        final = root / f"{name}.jsonl"
        staging = final.with_name(f".{final.name}.tmp")
        write_jsonl(staging, subset)
        staged.append((staging, final))
    for staging, final in staged:
        os.replace(staging, final)
    for name in ("eval-100.jsonl", "eval-300.jsonl"):
        (root / name).unlink(missing_ok=True)
    print(
        f"[swe] eval: {len(eval_rows)}/{len(rows)} tasks ({len(eval_rows) / len(rows):.1%})"
    )
    return {name: len(subset) for name, subset in outputs.items()}


def aggregate_probe_samples(
    samples: list[Any], *, n_samples: int
) -> dict[str, dict[str, int]]:

    def value(sample: Any, key: str, default: Any = None) -> Any:
        return (
            sample.get(key, default)
            if isinstance(sample, dict)
            else getattr(sample, key, default)
        )

    episodes: dict[tuple[str, int], dict[str, bool]] = {}
    for sample in samples:
        metadata = value(sample, "metadata", {}) or {}
        instance_id = str(
            metadata.get("instance_id") or value(sample, "label", "") or ""
        )
        if not instance_id:
            raise ValueError("probe dump contains a sample without an instance id")
        index = int(value(sample, "index"))
        agentic = metadata.get("agentic") or {}
        gradeable = not bool(value(sample, "remove_sample", False)) and (
            "is_solved" in agentic
        )
        state = episodes.setdefault(
            (instance_id, index), {"gradeable": False, "solved": False}
        )
        state["gradeable"] = state["gradeable"] or gradeable
        state["solved"] = state["solved"] or (gradeable and bool(agentic["is_solved"]))

    totals: dict[str, list[int]] = {}
    for (instance_id, _), state in episodes.items():
        counts = totals.setdefault(instance_id, [0, 0, 0])
        counts[2] += 1
        if state["gradeable"]:
            counts[1] += 1
            counts[0] += int(state["solved"])
    result = {
        instance_id: {
            "solved": counts[0],
            "gradeable": counts[1],
            "total": counts[2],
        }
        for instance_id, counts in sorted(totals.items())
    }
    if any(counts["total"] != n_samples for counts in result.values()):
        raise ValueError(
            f"rollout dump is not a fixed-sample evaluation dump: "
            f"expected {n_samples} episodes per task"
        )
    return result


def mixed_subset_name(source: str, recipe: str, n_samples: int) -> str:
    recipe_slug = re.sub(r"[^a-z0-9]+", "-", recipe.lower()).strip("-")
    if not recipe_slug:
        raise ValueError("recipe name must contain a letter or number")
    return f"{source}-mixed-reward-{recipe_slug}-n{n_samples}"


def write_mixed_subset(
    root: Path,
    *,
    source: str,
    recipe: str,
    samples: list[Any],
    n_samples: int,
    checkpoint: str | None,
    probe_dump: str,
    replace: bool = False,
) -> tuple[Path, dict[str, Any]]:
    train_splits = {f"train-{size}" for size in TRAIN_SPLIT_SIZES} | {"train-full"}
    if source not in train_splits:
        raise ValueError(f"source must be one of {sorted(train_splits)}")
    source_path = root / f"{source}.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"source subset does not exist: {source_path}")

    name = mixed_subset_name(source, recipe, n_samples)
    output_path = root / f"{name}.jsonl"
    metadata_path = root / f"{name}.json"
    if (output_path.exists() or metadata_path.exists()) and not replace:
        raise FileExistsError(f"{name} already exists; pass --replace to overwrite it")

    source_rows = read_jsonl(source_path)
    indexed = {
        str((row.get("metadata") or {}).get("instance_id") or ""): row
        for row in source_rows
    }
    if "" in indexed:
        raise ValueError(f"{source_path} contains a row without metadata.instance_id")
    if len(indexed) != len(source_rows):
        raise ValueError(f"{source_path} contains duplicate instance ids")

    counts = aggregate_probe_samples(samples, n_samples=n_samples)
    absent = sorted(set(indexed) - set(counts))
    unexpected = sorted(set(counts) - set(indexed))
    if absent or unexpected:
        raise ValueError(
            f"probe dump does not cover {source}: {len(absent)} source tasks are "
            f"missing from the probe and {len(unexpected)} probe tasks are not in "
            f"{source}"
        )
    selected_ids = sorted(
        (
            instance_id
            for instance_id, values in counts.items()
            if values["gradeable"] == values["total"] == n_samples
            and 0 < values["solved"] < n_samples
        ),
        key=lambda instance_id: (
            abs(2 * counts[instance_id]["solved"] - n_samples),
            instance_id,
        ),
    )
    write_jsonl(output_path, [indexed[instance_id] for instance_id in selected_ids])
    provenance = {
        "subset": name,
        "source": source,
        "source_sha256": sha256(source_path),
        "recipe": recipe,
        "checkpoint": checkpoint,
        "n_samples": n_samples,
        "criterion": MIXED_CRITERION,
        "probe_dump": probe_dump,
        "selected_instance_ids": selected_ids,
        "sha256": sha256(output_path),
    }
    write_text(metadata_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return output_path, provenance


def convert_tasks(
    root: Path,
    *,
    hf_revision: str | None = None,
    min_grade: str | None = "A",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if "/root/slime" not in sys.path:
        sys.path.insert(0, "/root/slime")
    from agentic_rl.environment.convert2slime import harbor, swerebench

    source = load_dataset(
        HF_DATASET, split="train", revision=hf_revision, streaming=True
    )
    rows = []
    skipped: Counter[str] = Counter()
    for row in source:
        if limit is not None and len(rows) >= limit:
            break
        if not swerebench._passes_quality(row, min_grade):
            skipped["quality grade"] += 1
            continue
        task_dir = root / "tasks" / swerebench._safe_id(row["instance_id"])
        try:
            swerebench.build_task_dir(row, task_dir)
            converted = harbor.translate_task(task_dir, dataset=DATASET_ROOT)
        except (swerebench.SkipRow, harbor.SkipTask) as exc:
            skipped[str(exc)] += 1
            shutil.rmtree(task_dir, ignore_errors=True)
            continue
        metadata = converted.setdefault("metadata", {})
        metadata["task_path"] = f"{root.name}/tasks/{task_dir.name}"
        metadata["source"] = {key: row[key] for key in SOURCE_COLUMNS if row.get(key)}
        rows.append(converted)
        if len(rows) % 500 == 0:
            print(
                f"[swe] converted {len(rows)} tasks ({sum(skipped.values())} skipped)"
            )
    if not rows:
        raise RuntimeError("conversion produced no rows")
    write_jsonl(root / "all.converted.jsonl", rows)
    for reason, count in skipped.most_common():
        print(f"[swe] skipped {count} rows: {reason}")
    return rows


def prepare_dataset(
    root: Path,
    *,
    hf_revision: str | None = None,
    min_grade: str | None = "A",
    limit: int | None = None,
) -> dict[str, int]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root.parent, prefix=f".{root.name}-") as temporary:
        staging = Path(temporary) / root.name
        staging.mkdir()
        rows = convert_tasks(
            staging, hf_revision=hf_revision, min_grade=min_grade, limit=limit
        )
        counts = write_partitions(staging, rows)
        previous = Path(temporary) / "previous"
        if root.exists():
            root.rename(previous)
        try:
            staging.rename(root)
        except OSError:
            if previous.exists():
                previous.rename(root)
            raise
    return counts


def _image(recipe: Qwen3_6_27B_Recipe_Agentic) -> modal.Image:
    if not (recipe.slime_git_repository and recipe.slime_git_revision):
        raise ValueError(f"{type(recipe).__name__} does not pin a Slime fork")
    return (
        modal.Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands(
            _slime_git_overlay_command(
                recipe.slime_git_repository,
                recipe.slime_git_revision,
            ),
            "uv pip install --system modal datasets huggingface_hub",
        )
        .add_local_python_source(
            "modal_training_gym", "tutorials.coding_agent", copy=True
        )
    )


def _prepare_remote(
    root: str,
    *,
    hf_revision: str | None,
    min_grade: str | None,
    limit: int | None,
    volume_name: str,
):
    counts = prepare_dataset(
        Path(root), hf_revision=hf_revision, min_grade=min_grade, limit=limit
    )
    modal.Volume.from_name(volume_name).commit()
    return counts


def _mixed_remote(
    root: str,
    *,
    source: str,
    recipe: str,
    probe_dump: str,
    n_samples: int,
    checkpoint: str | None,
    replace: bool,
    volume_name: str,
):
    import torch

    dump = Path(probe_dump)
    if not dump.is_file():
        raise FileNotFoundError(f"probe dump does not exist: {dump}")
    payload = torch.load(dump, weights_only=True)
    path, provenance = write_mixed_subset(
        Path(root),
        source=source,
        recipe=recipe,
        samples=payload["samples"],
        n_samples=n_samples,
        checkpoint=checkpoint,
        probe_dump=probe_dump,
        replace=replace,
    )
    modal.Volume.from_name(volume_name).commit()
    return str(path), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=DATASET_ROOT,
        help="Directory under /data holding the prepared tasks and subsets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--hf-revision")
    prepare.add_argument(
        "--min-grade",
        default="A",
        help="Keep rows whose meta.llm_metadata.code grade is this or better "
        "(A is best); 'none' keeps every row.",
    )
    prepare.add_argument(
        "--limit", type=int, help="Stop after converting this many tasks."
    )

    mixed = subparsers.add_parser("mixed")
    mixed.add_argument("--source", required=True)
    mixed.add_argument("--recipe", default=DEFAULT_MIXED_RECIPE_SLUG)
    mixed.add_argument("--probe-dump", required=True)
    mixed.add_argument("--n-samples", type=int, default=4)
    mixed.add_argument("--checkpoint", default="base")
    mixed.add_argument("--checkpoints-volume", required=True)
    mixed.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    try:
        dataset_root = dataset_root_name(dataset_root)
    except ValueError as exc:
        parser.error(str(exc))
    root = f"{DATA_PATH}/{dataset_root}"

    training_recipe = Qwen3_6_27B_Recipe_Agentic()
    volume_name = data_volume_name(training_recipe)
    app = modal.App("partition-swe-dataset")
    volumes = {
        str(DATA_PATH): modal.Volume.from_name(volume_name, create_if_missing=True)
    }
    if args.command == "mixed":
        volumes["/checkpoints"] = modal.Volume.from_name(
            args.checkpoints_volume, create_if_missing=False
        )
    remote_options: dict[str, Any] = {
        "image": _image(training_recipe),
        "volumes": volumes,
        "timeout": 24 * 60 * 60,
    }
    if args.command == "prepare":
        remote_options["secrets"] = hf_secrets()
        remote = app.function(**remote_options)(_prepare_remote)
        with app.run():
            counts = remote.remote(
                root,
                hf_revision=args.hf_revision,
                min_grade=None if args.min_grade.lower() == "none" else args.min_grade,
                limit=args.limit,
                volume_name=volume_name,
            )
        print("\n".join(f"{name}: {count}" for name, count in counts.items()))
        return

    remote = app.function(**remote_options)(_mixed_remote)
    with app.run():
        path, provenance = remote.remote(
            root,
            source=args.source,
            recipe=args.recipe,
            probe_dump=args.probe_dump,
            n_samples=args.n_samples,
            checkpoint=args.checkpoint,
            replace=args.replace,
            volume_name=volume_name,
        )
    print(f"{path}: {len(provenance['selected_instance_ids'])} rows")


if __name__ == "__main__":
    main()
