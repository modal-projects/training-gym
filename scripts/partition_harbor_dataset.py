"""Partition an archived Harbor dataset into deterministic training subsets.

Runs on Modal against the data volume of ``Qwen3_6_27B_Recipe_Agentic`` so the
subsets land where that recipe reads them, as ``/data/<dataset-root>/<subset>.jsonl``.
The tutorial selects a subset by that path.

``prepare`` downloads the archived tasks from the Hugging Face Hub, unpacks and
converts them once with the pinned Slime fork's Harbor translator, and writes
the splits below to ``/data/<dataset-root>/``.

``mixed`` filters one of the train splits to tasks whose probe rollouts were
fully gradeable with both successes and failures. It reads the rollout dump a
training run wrote via ``save_debug_rollout_data``, so it is only exercisable
after a probe run has completed.

Split design
------------
Each converted row is one Harbor task. Tasks belong to a *task group*, in
practice the GitHub repository they were mined from, and carry a *language*
from the dataset's metadata sheet. The splits make these guarantees:

* ``eval`` holds ``EVAL_SPLIT_FRACTION`` of the rows and is task-group
  disjoint from every train split: no repository appears on both sides.
* ``eval`` covers every language that appears in at least two task groups,
  and its language mix tracks the full dataset's as closely as disjointness
  allows. Every language keeps at least two task groups in train.
* ``train-full`` is everything not in ``eval``. ``train-<N>`` for each
  ``TRAIN_SPLIT_SIZES`` entry is drawn from ``train-full`` with the full
  dataset's language mix, and the splits nest: ``train-100`` is a subset of
  ``train-300``, which is a subset of ``train-1000``.
* Everything is deterministic under a fixed seed. Rerunning ``prepare`` on the
  same source produces byte-identical files, and ``all.converted.json``
  records the inputs the conversion depended on.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import stat
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

from modal_training_gym.common import hf_secrets
from modal_training_gym.frameworks.slime.launcher import (
    SLIME_IMAGE,
    _slime_git_overlay_command,
)
from modal_training_gym.train_recipes.base import DATA_PATH
from modal_training_gym.train_recipes.slime_recipe import (
    Qwen3_6_27B_Recipe_Agentic,
)

# The eval split takes this fraction of all rows. The train splits below are
# drawn from the remaining 1 - EVAL_SPLIT_FRACTION, so with 0.2 they come from
# the other 80% of the dataset.
EVAL_SPLIT_FRACTION = 0.2
TRAIN_SPLIT_SIZES = (100, 300, 1000)
SPLIT_SEED = 0
# Every language keeps at least this many task groups in train, so moving a
# group to eval never leaves train without examples of a language.
MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE = 2
# A mixed-reward subset keeps tasks whose n_samples probe episodes were all
# gradeable and were neither all solved nor all failed.
MIXED_CRITERION = "fully_gradeable_and_0_lt_solved_lt_n_samples"
METADATA_COLUMNS = ("language", "language_bucket", "category", "difficulty")
DEFAULT_MIXED_RECIPE_SLUG = "qwen3-6-27b-agentic"


def data_volume_name(recipe: Qwen3_6_27B_Recipe_Agentic) -> str:
    """The data volume the Slime launcher mounts at ``/data`` for ``recipe``."""
    return recipe.data_volume_name or f"slime-{type(recipe).__name__.lower()}-data"


def dataset_root_name(value: str) -> str:
    """Validate the directory name under ``/data`` that holds one dataset's subsets."""
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"dataset root must be a single directory name, got {value!r}")
    return value


def source_metadata(row: dict[str, Any], namespace: str) -> dict[str, Any]:
    """The columns copied from the dataset's ``tasks.csv`` sheet, if the task had a row."""
    value = (row.get("metadata") or {}).get(namespace) or {}
    return value if isinstance(value, dict) else {}


def language(row: dict[str, Any], namespace: str) -> str:
    """The task's language for balancing: the coarse bucket, else the raw language, else ``?``."""
    metadata = source_metadata(row, namespace)
    return str(metadata.get("language_bucket") or metadata.get("language") or "?")


def task_group(row: dict[str, Any], suffix_pattern: str) -> str:
    """The task's group, in practice its source repository.

    Task directories are named ``<owner>_<repo>__<issue>``, so stripping the
    ``__<issue>`` suffix maps ``aws_aws-cli__2819`` to ``aws_aws-cli``.
    """
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
    """Select ``count`` rows whose language mix matches ``rows``, as a nested prefix.

    Args:
        rows: The pool to draw from, normally the ``train-full`` split.
        count: How many rows to select.
        seed: Seeds the within-language shuffle and the tie-breaks. The same
            seed and pool give the same selection.
        metadata_namespace: The ``metadata`` key holding the ``tasks.csv``
            columns that carry each task's language.

    Rows are picked one position at a time. At each position the language that
    is furthest below its share of the full pool gets the next row, so the
    selection tracks the pool's language proportions at every prefix length.
    Because the choice at a position never depends on ``count``, a smaller
    ``count`` yields a prefix of a larger one: ``train-100`` is a subset of
    ``train-300``, which is a subset of ``train-1000``.
    """
    if count < 0 or count > len(rows):
        raise ValueError(f"sample count {count} is outside [0, {len(rows)}]")

    rows_by_language: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        rows_by_language.setdefault(language(row, metadata_namespace), []).append(index)

    # Shuffle within each language so the selection is not biased by conversion
    # order. The seeded per-language tie-break keeps exact deficit ties from
    # systematically favoring alphabetically earlier languages.
    rng = random.Random(seed)
    for indices in rows_by_language.values():
        rng.shuffle(indices)
    tie_break = {name: rng.random() for name in sorted(rows_by_language)}
    language_share = {
        name: len(indices) / len(rows) for name, indices in rows_by_language.items()
    }

    selected_counts: Counter[str] = Counter()

    def language_deficit(name: str, position: int) -> tuple[float, float, str]:
        """How far a language is below its expected count after ``position`` picks."""
        expected = position * language_share[name]
        return expected - selected_counts[name], tie_break[name], name

    selected: set[int] = set()
    for position in range(1, count + 1):
        languages_with_rows_left = [
            name
            for name, indices in rows_by_language.items()
            if selected_counts[name] < len(indices)
        ]
        name = max(
            languages_with_rows_left,
            key=lambda value: language_deficit(value, position),
        )
        selected.add(rows_by_language[name][selected_counts[name]])
        selected_counts[name] += 1
    return [row for index, row in enumerate(rows) if index in selected]


def repo_disjoint_split(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
    metadata_namespace: str,
    group_suffix_pattern: str = r"__\d+$",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``rows`` into ``(train, eval)`` with no task group on both sides.

    Args:
        rows: Every converted task.
        eval_fraction: Share of rows to move to eval.
        seed: Seeds the tie-breaks between otherwise equal task groups.
        metadata_namespace: The ``metadata`` key holding each task's language.
        group_suffix_pattern: Regex stripped from a task directory name to get
            its task group; the default removes the ``__<issue>`` suffix.

    The first priority is disjointness: eval is built from whole task groups,
    so a repository never contributes tasks to both train and eval. In
    practice task groups are GitHub repositories, and leaking one across the
    split would let a model score on eval tasks by recognizing code it
    trained on.

    The second priority is language balance. Eval should contain every
    language and match the full dataset's language proportions, and train must
    keep ``MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE`` groups of every language.
    Whole-group moves make exact proportions impossible, so the eval size and
    per-language counts are approximate targets.
    """
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    if not rows:
        raise ValueError("cannot split an empty dataset")

    task_groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        task_groups.setdefault(task_group(row, group_suffix_pattern), []).append(index)
    task_group_language_counts = {
        group: Counter(language(rows[index], metadata_namespace) for index in indices)
        for group, indices in task_groups.items()
    }
    language_counts = Counter(language(row, metadata_namespace) for row in rows)
    task_groups_by_language = {
        name: {
            group
            for group, counts in task_group_language_counts.items()
            if counts[name]
        }
        for name in language_counts
    }
    eval_target_num_rows = round(len(rows) * eval_fraction)
    eval_target_language_counts = {
        name: count * eval_fraction for name, count in language_counts.items()
    }

    # Seeded per-group tie-break so equal candidates are not always resolved in
    # alphabetical order.
    rng = random.Random(seed)
    tie_break = {group: rng.random() for group in sorted(task_groups)}
    eval_selected: set[str] = set()
    eval_num_rows = 0
    eval_language_counts: Counter[str] = Counter()

    def can_move_to_eval(group: str) -> bool:
        """Whether moving ``group`` to eval keeps every one of its languages in train.

        A language must keep ``MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE`` groups out
        of eval, so a language that only exists in one group stays train-only.
        """
        if group in eval_selected:
            return False
        return all(
            len(task_groups_by_language[name] - eval_selected - {group})
            >= MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE
            for name in task_group_language_counts[group]
        )

    def move_to_eval(group: str) -> None:
        nonlocal eval_num_rows
        eval_selected.add(group)
        eval_num_rows += len(task_groups[group])
        eval_language_counts.update(task_group_language_counts[group])

    # Ensure eval represents every language by preselecting one task group per
    # language, rarest language first so it is not crowded out later. Languages
    # confined to a single group are skipped: they cannot leave train.
    for name in sorted(
        language_counts, key=lambda value: (language_counts[value], value)
    ):
        if (
            len(task_groups_by_language[name]) < MIN_TRAIN_TASK_GROUPS_PER_LANGUAGE
            or eval_language_counts[name]
        ):
            continue
        eval_candidates = [
            group for group in task_groups_by_language[name] if can_move_to_eval(group)
        ]
        if eval_candidates:
            # Pick the task group with the fewest tasks, so this coverage pass
            # spends as little of the eval budget as possible.
            move_to_eval(
                min(
                    eval_candidates,
                    key=lambda group: (
                        len(task_groups[group]),
                        tie_break[group],
                        group,
                    ),
                )
            )

    def eval_selection_error(group: str) -> tuple[float, float, str]:
        """How far eval would be from its targets after adding ``group``.

        Sums the relative error in total eval size with the mean relative error
        across per-language counts, so a group is preferred when it moves both
        toward target. Lower is better.
        """
        size_error = abs(
            eval_num_rows + len(task_groups[group]) - eval_target_num_rows
        ) / max(eval_target_num_rows, 1)
        language_error = sum(
            abs(
                eval_language_counts[name]
                + task_group_language_counts[group][name]
                - target
            )
            / max(target, 1.0)
            for name, target in eval_target_language_counts.items()
        ) / max(len(eval_target_language_counts), 1)
        return size_error + language_error, tie_break[group], group

    # Fill eval greedily up to its target size. Prefer groups that fit in the
    # remaining budget; if none fit, allow any movable group so the loop can
    # still reach the target, then stop once nothing can move.
    while eval_num_rows < eval_target_num_rows:
        remaining = eval_target_num_rows - eval_num_rows
        eval_candidates = [
            group
            for group in task_groups
            if can_move_to_eval(group) and len(task_groups[group]) <= remaining
        ]
        if not eval_candidates:
            eval_candidates = [
                group for group in task_groups if can_move_to_eval(group)
            ]
        if not eval_candidates:
            break
        move_to_eval(min(eval_candidates, key=eval_selection_error))

    eval_indices = sorted(
        index for group in eval_selected for index in task_groups[group]
    )
    eval_index_set = set(eval_indices)
    train_rows = [row for index, row in enumerate(rows) if index not in eval_index_set]
    eval_rows = [rows[index] for index in eval_indices]
    train_groups = {task_group(row, group_suffix_pattern) for row in train_rows}
    eval_groups = {task_group(row, group_suffix_pattern) for row in eval_rows}
    if train_groups & eval_groups:
        raise RuntimeError("repository-disjoint split leaked groups")
    return train_rows, eval_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text(path: Path, text: str) -> None:
    """Replace ``path`` atomically so readers never observe a partial file."""
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
    """Write ``eval.jsonl``, ``train-full.jsonl``, and the nested ``train-<N>.jsonl`` splits under ``root``.

    ``eval`` is task-group disjoint from ``train-full``; each ``train-<N>`` is
    a language-balanced subset of ``train-full`` and of every larger ``train-<M>``.
    """
    train_rows, eval_rows = repo_disjoint_split(
        rows,
        eval_fraction=EVAL_SPLIT_FRACTION,
        seed=seed,
        metadata_namespace=metadata_namespace,
    )
    outputs = {"eval": eval_rows, "train-full": train_rows}
    for size in TRAIN_SPLIT_SIZES:
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
    """Count solved, gradeable, and total probe episodes per task.

    An episode is one ``(instance_id, sample index)`` pair; a dump may hold
    several records for one episode, which are merged. An episode is gradeable
    when the rollout kept it and the environment reported ``is_solved``, and
    solved when that report was true.
    """

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
    """Filter a train split to tasks that are neither too easy nor too hard for the probed model.

    A task is kept when all ``n_samples`` probe episodes were gradeable and it
    was solved at least once but not every time (``MIXED_CRITERION``). With
    binary rewards, tasks solved always or never give GRPO zero advantage
    signal. The output is ordered by closeness to a 50% solve rate, where that
    signal is strongest, and a JSON sidecar records the provenance.
    """
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
    write_text(metadata_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
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
        translator_revision: str,
        unpack_workers: int,
        max_archives: int | None,
    ) -> None:
        self.hf_repo = hf_repo
        self.dataset_key = dataset_key
        self.hf_revision = hf_revision
        self.metadata_namespace = metadata_namespace
        self.translator_revision = translator_revision
        self.unpack_workers = unpack_workers
        self.max_archives = max_archives

    def download(self, root: Path) -> str:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().dataset_info(self.hf_repo, revision=self.hf_revision).sha
        snapshot_download(
            self.hf_repo,
            repo_type="dataset",
            local_dir=str(root),
            revision=revision,
        )
        print(f"[harbor] downloaded {self.hf_repo}@{revision}")
        return revision

    def bundles(self, root: Path) -> list[Path]:
        bundles = sorted(root.glob("tasks/batch_*.zip"))
        if self.max_archives is not None:
            bundles = bundles[: self.max_archives]
        return bundles

    def source_record(self, root: Path, revision: str) -> dict[str, Any]:
        """Every input the converted rows depend on; a change invalidates them."""
        return {
            "hf_repo": self.hf_repo,
            "revision": revision,
            "dataset_key": self.dataset_key,
            "metadata_namespace": self.metadata_namespace,
            "translator_revision": self.translator_revision,
            "archives": [bundle.name for bundle in self.bundles(root)],
        }

    @staticmethod
    def cached_rows(root: Path, source: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Rows converted from exactly ``source``, or ``None`` when they must be rebuilt."""
        record_path = root / "all.converted.json"
        converted_path = root / "all.converted.jsonl"
        if not (record_path.is_file() and converted_path.is_file()):
            return None
        if json.loads(record_path.read_text(encoding="utf-8")) != source:
            return None
        return read_jsonl(converted_path)

    @staticmethod
    def clear_extracted(root: Path) -> None:
        """Drop extracted tasks, their markers, and conversion outputs; keep the archives."""
        tasks_root = root / "tasks"
        for entry in tasks_root.iterdir() if tasks_root.is_dir() else ():
            if entry.is_dir():
                shutil.rmtree(entry)
            elif entry.suffix == ".extracted":
                entry.unlink()
        for name in ("all.converted.json", "all.converted.jsonl"):
            (root / name).unlink(missing_ok=True)

    @staticmethod
    def safe_extract(bundle: Path, tasks_root: Path) -> int:
        """Extract one archive's task directories, or skip it if a previous run finished it."""
        marker = tasks_root / f".{bundle.stem}.extracted"
        staging = tasks_root / f".{bundle.stem}.partial"
        with zipfile.ZipFile(bundle) as archive:
            members = archive.infolist()
            task_roots = {
                member.filename.split("/", 1)[0]
                for member in members
                if "/" in member.filename
            }
            if marker.is_file():
                return len(task_roots)
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            destination = staging.resolve()
            for member in members:
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(
                        f"archive {bundle.name} contains a link {member.filename!r}"
                    )
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(
                        f"archive {bundle.name} contains unsafe path {member.filename!r}"
                    )
            archive.extractall(staging)
        for task_root in task_roots:
            final = tasks_root / task_root
            shutil.rmtree(final, ignore_errors=True)
            os.replace(staging / task_root, final)
        shutil.rmtree(staging, ignore_errors=True)
        marker.touch()
        return len(task_roots)

    def unpack(self, root: Path) -> None:
        tasks_root = root / "tasks"
        bundles = self.bundles(root)
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

        task_dirs = sorted(
            path
            for path in (root / "tasks").iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
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
        write_jsonl(root / "all.converted.jsonl", rows)
        if skipped:
            print(f"[harbor] skipped {sum(skipped.values())} tasks: {dict(skipped)}")
        return rows

    def partition(self, root: Path) -> dict[str, int]:
        root.mkdir(parents=True, exist_ok=True)
        revision = self.download(root)
        source = self.source_record(root, revision)
        rows = self.cached_rows(root, source)
        if rows is None:
            self.clear_extracted(root)
            self.unpack(root)
            rows = self.convert(root)
            write_text(
                root / "all.converted.json",
                json.dumps(source, indent=2, sort_keys=True) + "\n",
            )
        return write_partitions(root, rows, metadata_namespace=self.metadata_namespace)


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
    if dataset_root is None:
        parser.error("--dataset-root is required for mixed")
    try:
        dataset_root = dataset_root_name(dataset_root)
    except ValueError as exc:
        parser.error(str(exc))
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
        remote_options["secrets"] = hf_secrets()
        remote = app.function(**remote_options)(_partition_remote)
        kwargs = {
            "hf_repo": args.hf_repo,
            "dataset_key": args.dataset_key,
            "hf_revision": args.hf_revision,
            "metadata_namespace": args.metadata_namespace,
            "translator_revision": training_recipe.slime_git_revision,
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
