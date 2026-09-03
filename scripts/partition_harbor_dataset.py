"""Partition an archived Harbor dataset into deterministic training subsets.

Runs on Modal against the data volume of ``Qwen3_6_27B_Recipe_Agentic`` so the
subsets land where that recipe reads them: a ``DatasetConfig`` with
``hf_repo=<dataset-root>`` and ``hf_split=<subset>`` resolves to
``/data/<dataset-root>/<subset>.jsonl``.

``prepare`` downloads the archived tasks from the Hugging Face Hub, unpacks and
converts them once with the pinned Slime fork's Harbor translator, and writes a
repository-disjoint 20% ``eval`` split plus nested, language-balanced
``train-100``, ``train-300``, ``train-1000``, and ``train-full`` subsets to
``/data/<dataset-root>/``.

``mixed`` filters one of those train subsets to tasks whose probe rollouts were
fully gradeable with both successes and failures. It reads the rollout dump a
training run wrote via ``save_debug_rollout_data``, so it is only exercisable
after a probe run has completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

from modal_training_gym.frameworks.slime.launcher import (
    SLIME_IMAGE,
    _slime_git_overlay_command,
)
from modal_training_gym.train_recipes.base import DATA_PATH
from modal_training_gym.train_recipes.slime_recipe import (
    Qwen3_6_27B_Recipe_Agentic,
)

TRAIN_SIZES = (100, 300, 1000)
EVAL_FRACTION = 0.2
MIXED_CRITERION = "fully_gradeable_and_0_lt_solved_lt_n_samples"
METADATA_COLUMNS = ("language", "language_bucket", "category", "difficulty")
DEFAULT_MIXED_RECIPE_SLUG = "qwen3-6-27b-agentic"


def data_volume_name(recipe: Qwen3_6_27B_Recipe_Agentic) -> str:
    """The data volume the Slime launcher mounts at ``/data`` for ``recipe``."""
    return recipe.data_volume_name or f"slime-{type(recipe).__name__.lower()}-data"


def _metadata(row: dict[str, Any], namespace: str) -> dict[str, Any]:
    value = (row.get("metadata") or {}).get(namespace) or {}
    return value if isinstance(value, dict) else {}


def _language(row: dict[str, Any], namespace: str) -> str:
    metadata = _metadata(row, namespace)
    return str(metadata.get("language_bucket") or metadata.get("language") or "?")


def _group(row: dict[str, Any], suffix_pattern: str) -> str:
    task_path = str((row.get("metadata") or {}).get("task_path") or "")
    if not task_path:
        raise ValueError("converted row is missing metadata.task_path")
    return re.sub(suffix_pattern, "", Path(task_path).name)


def nested_subset(
    rows: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    metadata_namespace: str,
) -> list[dict[str, Any]]:
    """Select a deterministic language-balanced prefix."""
    if count < 0 or count > len(rows):
        raise ValueError(f"sample count {count} is outside [0, {len(rows)}]")

    by_language: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_language.setdefault(_language(row, metadata_namespace), []).append(index)

    rng = random.Random(seed)
    for indices in by_language.values():
        rng.shuffle(indices)
    tie_break = {language: rng.random() for language in sorted(by_language)}

    selected_counts: Counter[str] = Counter()
    selected: set[int] = set()
    for position in range(1, count + 1):
        available = [
            language
            for language, indices in by_language.items()
            if selected_counts[language] < len(indices)
        ]
        language = max(
            available,
            key=lambda value: (
                position * len(by_language[value]) / len(rows) - selected_counts[value],
                tie_break[value],
                value,
            ),
        )
        selected.add(by_language[language][selected_counts[language]])
        selected_counts[language] += 1
    return [row for index, row in enumerate(rows) if index in selected]


def repo_disjoint_split(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
    metadata_namespace: str,
    group_suffix_pattern: str = r"__\d+$",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create an approximately language-balanced, repository-disjoint split."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    if not rows:
        raise ValueError("cannot split an empty dataset")

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(_group(row, group_suffix_pattern), []).append(index)
    group_languages = {
        group: Counter(_language(rows[index], metadata_namespace) for index in indices)
        for group, indices in groups.items()
    }
    language_totals = Counter(_language(row, metadata_namespace) for row in rows)
    language_groups = {
        language: {
            group for group, counts in group_languages.items() if counts[language]
        }
        for language in language_totals
    }
    target_total = round(len(rows) * eval_fraction)
    language_targets = {
        language: count * eval_fraction for language, count in language_totals.items()
    }

    rng = random.Random(seed)
    tie_break = {group: rng.random() for group in sorted(groups)}
    selected: set[str] = set()
    selected_total = 0
    eval_languages: Counter[str] = Counter()

    def can_select(group: str) -> bool:
        if group in selected:
            return False
        return all(
            len(language_groups[language]) > 1
            and len(language_groups[language] - selected) > 1
            for language in group_languages[group]
        )

    def add(group: str) -> None:
        nonlocal selected_total
        selected.add(group)
        selected_total += len(groups[group])
        eval_languages.update(group_languages[group])

    for language in sorted(
        language_totals, key=lambda value: (language_totals[value], value)
    ):
        if len(language_groups[language]) < 2 or eval_languages[language]:
            continue
        candidates = [group for group in language_groups[language] if can_select(group)]
        if candidates:
            add(
                min(
                    candidates,
                    key=lambda group: (
                        len(groups[group]),
                        tie_break[group],
                        group,
                    ),
                )
            )

    def objective(group: str) -> tuple[float, float, str]:
        total_error = abs(selected_total + len(groups[group]) - target_total) / max(
            target_total, 1
        )
        language_error = sum(
            abs(eval_languages[language] + group_languages[group][language] - target)
            / max(target, 1.0)
            for language, target in language_targets.items()
        ) / max(len(language_targets), 1)
        return total_error + language_error, tie_break[group], group

    while selected_total < target_total:
        remaining = target_total - selected_total
        candidates = [
            group
            for group in groups
            if can_select(group) and len(groups[group]) <= remaining
        ]
        if not candidates:
            candidates = [group for group in groups if can_select(group)]
        if not candidates:
            break
        add(min(candidates, key=objective))

    eval_indices = sorted(index for group in selected for index in groups[group])
    eval_index_set = set(eval_indices)
    train_rows = [row for index, row in enumerate(rows) if index not in eval_index_set]
    eval_rows = [rows[index] for index in eval_indices]
    train_groups = {_group(row, group_suffix_pattern) for row in train_rows}
    eval_groups = {_group(row, group_suffix_pattern) for row in eval_rows}
    if train_groups & eval_groups:
        raise RuntimeError("repository-disjoint split leaked groups")
    return train_rows, eval_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
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
    seed: int = 0,
) -> dict[str, int]:
    """Write ``eval`` and the nested ``train-*`` subsets as ``<name>.jsonl`` under ``root``."""
    train_rows, eval_rows = repo_disjoint_split(
        rows,
        eval_fraction=EVAL_FRACTION,
        seed=seed,
        metadata_namespace=metadata_namespace,
    )
    outputs = {"eval": eval_rows, "train-full": train_rows}
    for size in TRAIN_SIZES:
        if size > len(train_rows):
            raise ValueError(
                f"requested train-{size} from only {len(train_rows)} training rows"
            )
        outputs[f"train-{size}"] = nested_subset(
            train_rows,
            size,
            seed=seed,
            metadata_namespace=metadata_namespace,
        )
    for name, subset in outputs.items():
        write_jsonl(root / f"{name}.jsonl", subset)
    return {name: len(subset) for name, subset in outputs.items()}


def aggregate_probe_samples(
    samples: list[Any], *, n_samples: int
) -> dict[str, dict[str, int]]:
    """Count solved, gradeable, and total episodes for each task."""

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
    if any(counts["total"] > n_samples for counts in result.values()):
        raise ValueError(f"probe contains more than {n_samples} episodes for a task")
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
    """Filter a train subset to fully gradeable tasks with mixed outcomes."""
    if not re.fullmatch(r"train-(100|300|1000|full)", source):
        raise ValueError(
            "source must be train-100, train-300, train-1000, or train-full"
        )
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
    missing = [
        instance_id for instance_id in selected_ids if instance_id not in indexed
    ]
    if missing:
        raise ValueError(f"probe references {len(missing)} tasks absent from {source}")

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
    metadata_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, provenance


class ArchivedHarborSource:
    """Download, safely unpack, and convert a zipped Harbor dataset."""

    def __init__(
        self,
        *,
        hf_repo: str,
        dataset_key: str,
        hf_revision: str | None,
        metadata_namespace: str,
        unpack_workers: int,
        max_archives: int | None,
    ) -> None:
        self.hf_repo = hf_repo
        self.dataset_key = dataset_key
        self.hf_revision = hf_revision
        self.metadata_namespace = metadata_namespace
        self.unpack_workers = unpack_workers
        self.max_archives = max_archives

    def download(self, root: Path) -> None:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().dataset_info(self.hf_repo, revision=self.hf_revision).sha
        snapshot_download(
            self.hf_repo,
            repo_type="dataset",
            local_dir=str(root),
            revision=revision,
        )
        print(f"[harbor] downloaded {self.hf_repo}@{revision}")

    @staticmethod
    def safe_extract(bundle: Path, tasks_root: Path) -> int:
        destination = tasks_root.resolve()
        with zipfile.ZipFile(bundle) as archive:
            members = archive.infolist()
            task_roots = {
                member.filename.split("/", 1)[0]
                for member in members
                if "/" in member.filename
            }
            if task_roots and all(
                (tasks_root / task_root / "task.toml").is_file()
                for task_root in task_roots
            ):
                return len(task_roots)
            for member in members:
                target = (tasks_root / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(
                        f"archive {bundle.name} contains unsafe path {member.filename!r}"
                    )
            archive.extractall(tasks_root)
            return len(task_roots)

    def unpack(self, root: Path) -> None:
        tasks_root = root / "tasks"
        bundles = sorted(root.glob("tasks/batch_*.zip"))
        if self.max_archives is not None:
            bundles = bundles[: self.max_archives]
        if not bundles:
            raise FileNotFoundError(f"no task archives found under {tasks_root}")
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.unpack_workers) as pool:
            for completed, task_count in enumerate(
                pool.map(lambda bundle: self.safe_extract(bundle, tasks_root), bundles),
                start=1,
            ):
                print(
                    f"[harbor] unpacked {completed}/{len(bundles)} archives "
                    f"({task_count} tasks, {time.monotonic() - started:.0f}s)"
                )

    def convert(self, root: Path) -> list[dict[str, Any]]:
        converted_path = root / "all.converted.jsonl"
        if converted_path.is_file() and converted_path.stat().st_size:
            return read_jsonl(converted_path)
        if "/root/slime" not in sys.path:
            sys.path.insert(0, "/root/slime")
        from agentic_rl.environment.convert2slime.harbor import (  # type: ignore[import-not-found]
            SkipTask,
            translate_task,
        )

        metadata_path = root / "tasks.csv"
        metadata_index: dict[str, dict[str, str]] = {}
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8", newline="") as handle:
                metadata_index = {
                    row["task_id"]: row
                    for row in csv.DictReader(handle)
                    if row.get("task_id")
                }

        def convert_one(
            task_dir: Path,
        ) -> tuple[Path, dict[str, Any] | None, str | None]:
            try:
                return (
                    task_dir,
                    translate_task(task_dir, dataset=self.dataset_key),
                    None,
                )
            except SkipTask as exc:
                return task_dir, None, str(exc)

        task_dirs = sorted(path for path in (root / "tasks").iterdir() if path.is_dir())
        rows: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=self.unpack_workers) as pool:
            for task_dir, row, reason in pool.map(convert_one, task_dirs):
                if reason:
                    skipped[reason] += 1
                    continue
                assert row is not None
                metadata = row.setdefault("metadata", {})
                metadata["task_path"] = f"{root.name}/tasks/{task_dir.name}"
                source = metadata_index.get(task_dir.name)
                if source:
                    metadata[self.metadata_namespace] = {
                        key: source[key] for key in METADATA_COLUMNS if source.get(key)
                    }
                rows.append(row)
        if not rows:
            raise RuntimeError("Harbor conversion produced no rows")
        write_jsonl(converted_path, rows)
        if skipped:
            print(f"[harbor] skipped {sum(skipped.values())} tasks: {dict(skipped)}")
        return rows

    def partition(self, root: Path) -> dict[str, int]:
        root.mkdir(parents=True, exist_ok=True)
        self.download(root)
        self.unpack(root)
        return write_partitions(
            root,
            self.convert(root),
            metadata_namespace=self.metadata_namespace,
        )


def _image(recipe: Qwen3_6_27B_Recipe_Agentic) -> modal.Image:
    """The Slime image with the recipe's pinned fork, whose translator converts tasks."""
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
        .add_local_python_source("modal_training_gym", copy=True)
    )


def _partition_remote(root: str, kwargs: dict[str, Any], volume_name: str):
    counts = ArchivedHarborSource(**kwargs).partition(Path(root))
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
    payload = torch.load(dump, weights_only=False)
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
        help="Directory under /data holding the subsets; `prepare` defaults it "
        "to the Hugging Face repo id with '/' replaced by '_'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--hf-repo", required=True)
    prepare.add_argument("--dataset-key", required=True)
    prepare.add_argument("--hf-revision")
    prepare.add_argument("--metadata-namespace", default="source")
    prepare.add_argument("--unpack-workers", type=int, default=32)
    prepare.add_argument("--max-archives", type=int)

    mixed = subparsers.add_parser("mixed")
    mixed.add_argument("--source", required=True)
    mixed.add_argument("--recipe", default=DEFAULT_MIXED_RECIPE_SLUG)
    mixed.add_argument("--probe-dump", required=True)
    mixed.add_argument("--n-samples", type=int, default=8)
    mixed.add_argument("--checkpoint", default="base")
    mixed.add_argument("--checkpoints-volume", required=True)
    mixed.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if args.command == "prepare":
        dataset_root = dataset_root or args.hf_repo.replace("/", "_")
    if not dataset_root:
        parser.error("--dataset-root is required for mixed")
    root = f"{DATA_PATH}/{dataset_root}"

    training_recipe = Qwen3_6_27B_Recipe_Agentic()
    volume_name = data_volume_name(training_recipe)
    app = modal.App("partition-harbor-dataset")
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
        remote_options["secrets"] = [modal.Secret.from_name("huggingface-secret")]
        remote = app.function(**remote_options)(_partition_remote)
        kwargs = {
            "hf_repo": args.hf_repo,
            "dataset_key": args.dataset_key,
            "hf_revision": args.hf_revision,
            "metadata_namespace": args.metadata_namespace,
            "unpack_workers": args.unpack_workers,
            "max_archives": args.max_archives,
        }
        with app.run():
            counts = remote.remote(root, kwargs, volume_name)
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
