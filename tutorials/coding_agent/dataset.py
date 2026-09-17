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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import modal

from modal_training_gym import DatasetConfig

from modal_training_gym.common import hf_secrets
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
# ``mixed`` keeps a task when all n_samples episodes were gradeable and the
# model solved it at least once but not every time.
MIXED_CRITERION = "fully_gradeable_and_0_lt_solved_lt_n_samples"
# Row columns kept under ``metadata.<namespace>`` for split balancing and analysis.
SOURCE_COLUMNS = ("repo", "language", "license", "created_at")
DEFAULT_MIXED_RECIPE_SLUG = "qwen3-6-27b-agentic"


@dataclass(frozen=True)
class SweDataset:
    hf_repo: str
    split: str
    key: str


SWE_DATASETS = {
    "swe-rebench-v2": SweDataset(
        hf_repo="nebius/SWE-rebench-V2", split="train", key="swe_rebench_v2"
    ),
}
DEFAULT_SWE_DATASET = "swe-rebench-v2"


def data_volume_name(recipe: Qwen3_6_27B_Recipe_Agentic) -> str:
    return recipe.data_volume_name or f"slime-{type(recipe).__name__.lower()}-data"


def dataset_root_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"dataset root must be a single directory name, got {value!r}")
    return value


class PreparedTaskSubset(DatasetConfig):
    def __init__(self, path: Path):
        self.path = path

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def apply_chat_template(self) -> bool:
        return False

    def rows(self):
        with self.path.open() as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def source_metadata(row: dict[str, Any], namespace: str) -> dict[str, Any]:
    value = (row.get("metadata") or {}).get(namespace) or {}
    return value if isinstance(value, dict) else {}


def language(row: dict[str, Any], namespace: str) -> str:
    metadata = source_metadata(row, namespace)
    return str(metadata.get("language_bucket") or metadata.get("language") or "?")


def task_group(row: dict[str, Any], namespace: str) -> str:
    repo = source_metadata(row, namespace).get("repo")
    if not repo:
        raise ValueError(f"converted row is missing metadata.{namespace}.repo")
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
    metadata_namespace: str = "source",
    seed: int = SPLIT_SEED,
) -> dict[str, int]:
    source = PreparedTaskSubset(root / "all.converted.jsonl")
    train, evaluation = source.snapshot(rows).split(
        eval_fraction=EVAL_SPLIT_FRACTION,
        seed=seed,
        group_key=lambda row: task_group(row, metadata_namespace),
        stratify_key=lambda row: language(row, metadata_namespace),
        min_train_groups=MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE,
    )
    train_rows, eval_rows = list(train.rows()), list(evaluation.rows())
    outputs = {"eval": eval_rows, "train-full": train_rows}
    for prefix, pool, sizes in (
        ("eval", evaluation, EVAL_SPLIT_SIZES),
        ("train", train, TRAIN_SPLIT_SIZES),
    ):
        pool_size = len(list(pool.rows()))
        available = [size for size in sizes if size <= pool_size]
        subsets = pool.nested_subsets(
            available,
            seed=seed,
            stratify_key=lambda row: language(row, metadata_namespace),
        )
        for size in sizes:
            if size > pool_size:
                (root / f"{prefix}-{size}.jsonl").unlink(missing_ok=True)
                print(f"[swe] skipping {prefix}-{size}: only {pool_size} {prefix} rows")
                continue
            outputs[f"{prefix}-{size}"] = list(subsets[size].rows())
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


class SweBenchSource:
    def __init__(
        self,
        *,
        dataset: SweDataset,
        hf_revision: str | None,
        metadata_namespace: str,
        translator_revision: str,
        min_grade: str | None,
        limit: int | None,
    ) -> None:
        self.dataset = dataset
        self.hf_revision = hf_revision
        self.metadata_namespace = metadata_namespace
        self.translator_revision = translator_revision
        self.min_grade = min_grade
        self.limit = limit

    def resolve_revision(self) -> str:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(self.dataset.hf_repo, revision=self.hf_revision).sha

    def rows(self, revision: str) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        for row in load_dataset(
            self.dataset.hf_repo,
            split=self.dataset.split,
            revision=revision,
            streaming=True,
        ):
            yield dict(row)

    def source_record(self, revision: str) -> dict[str, Any]:
        return {
            "hf_repo": self.dataset.hf_repo,
            "split": self.dataset.split,
            "revision": revision,
            "key": self.dataset.key,
            "metadata_namespace": self.metadata_namespace,
            "translator_revision": self.translator_revision,
            "min_grade": self.min_grade,
            "limit": self.limit,
        }

    @staticmethod
    def cached_rows(root: Path, source: dict[str, Any]) -> list[dict[str, Any]] | None:
        record_path = root / "all.converted.json"
        converted_path = root / "all.converted.jsonl"
        if not (record_path.is_file() and converted_path.is_file()):
            return None
        if json.loads(record_path.read_text(encoding="utf-8")) != source:
            return None
        return read_jsonl(converted_path)

    def convert(self, root: Path, revision: str) -> list[dict[str, Any]]:
        if "/root/slime" not in sys.path:
            sys.path.insert(0, "/root/slime")
        from agentic_rl.environment.convert2slime import (  # type: ignore[import-not-found]
            harbor,
            swerebench,
        )

        tasks_root = root / "tasks"
        rows: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        for row in self.rows(revision):
            if self.limit is not None and len(rows) >= self.limit:
                break
            if not swerebench._passes_quality(row, self.min_grade):
                skipped["quality grade"] += 1
                continue
            task_dir = tasks_root / swerebench._safe_id(row["instance_id"])
            try:
                swerebench.build_task_dir(row, task_dir)
                converted = harbor.translate_task(task_dir, dataset=self.dataset.key)
            except (swerebench.SkipRow, harbor.SkipTask) as exc:
                skipped[str(exc)] += 1
                shutil.rmtree(task_dir, ignore_errors=True)
                continue
            metadata = converted.setdefault("metadata", {})
            metadata["task_path"] = f"{root.name}/tasks/{task_dir.name}"
            if self.metadata_namespace in metadata:
                raise ValueError(
                    f"metadata namespace {self.metadata_namespace!r} conflicts with converted task metadata"
                )
            metadata[self.metadata_namespace] = {
                key: row[key] for key in SOURCE_COLUMNS if row.get(key)
            }
            rows.append(converted)
            if len(rows) % 500 == 0:
                print(
                    f"[swe] converted {len(rows)} tasks ({sum(skipped.values())} skipped)"
                )
        if not rows:
            raise RuntimeError("conversion produced no rows")
        write_jsonl(root / "all.converted.jsonl", rows)
        if skipped:
            print(f"[swe] skipped {sum(skipped.values())} rows:")
            for reason, count in skipped.most_common():
                print(f"{count:>8}  {reason}")
        return rows

    def partition(self, root: Path) -> dict[str, int]:
        root.mkdir(parents=True, exist_ok=True)
        revision = self.resolve_revision()
        source = self.source_record(revision)
        rows = self.cached_rows(root, source)
        if rows is None:
            with TemporaryDirectory(
                dir=root.parent, prefix=f".{root.name}-"
            ) as temporary:
                staging = Path(temporary) / root.name
                staging.mkdir()
                rows = self.convert(staging, revision)
                counts = write_partitions(
                    staging, rows, metadata_namespace=self.metadata_namespace
                )
                write_text(
                    staging / "all.converted.json",
                    json.dumps(source, indent=2, sort_keys=True) + "\n",
                )
                previous = Path(f"{temporary}.previous")
                root.rename(previous)
                try:
                    staging.rename(root)
                except OSError:
                    previous.rename(root)
                    raise
                shutil.rmtree(previous, ignore_errors=True)
                return counts
        return write_partitions(root, rows, metadata_namespace=self.metadata_namespace)


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


def _partition_remote(
    root: str, dataset: str, kwargs: dict[str, Any], volume_name: str
):
    counts = SweBenchSource(dataset=SWE_DATASETS[dataset], **kwargs).partition(
        Path(root)
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
        help="Directory under /data holding the subsets. prepare defaults this "
        "to the dataset's key.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--dataset", choices=sorted(SWE_DATASETS), default=DEFAULT_SWE_DATASET
    )
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
    prepare.add_argument("--metadata-namespace", default="source")

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
    if args.command == "prepare":
        dataset_root = dataset_root or SWE_DATASETS[args.dataset].key
    if dataset_root is None:
        parser.error("--dataset-root is required for mixed")
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
        remote = app.function(**remote_options)(_partition_remote)
        kwargs = {
            "hf_revision": args.hf_revision,
            "metadata_namespace": args.metadata_namespace,
            "translator_revision": training_recipe.slime_git_revision,
            "min_grade": None if args.min_grade.lower() == "none" else args.min_grade,
            "limit": args.limit,
        }
        with app.run():
            counts = remote.remote(root, args.dataset, kwargs, volume_name)
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
