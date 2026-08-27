"""Dataset config + row materialization, shared across training frameworks.

Pure data — each framework config writes its own converter from a
``DatasetConfig`` instance to its specific CLI flags (e.g. SlimeRecipe emits
``--prompt-data``, ``--input-key``, …).

Subclass and implement ``rows()``; launchers call ``write()`` to materialize
the data into a shared volume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Any, Literal
import json
import random
import shutil
import tomllib
import uuid
from pathlib import Path

from datasets import Dataset as HFDataset

from modal_training_gym.common.errors import TrainingGymConfigError

DatasetRow = dict[str, Any]


class DatasetType(Enum):
    DEFAULT = "default"
    HUGGING_FACE = "hugging_face"
    HARBOR = "harbor"


class DatasetConfig(ABC):
    """Dataset configuration shared across training frameworks.

    A dataset describes *what* the data is and a way to iterate over its rows.
    The recipe/launcher layer then uses the dataset to materialize the data on disk.

    output_format : str
        On-disk format ``write()`` uses, ``"parquet"`` (default) or
        ``"jsonl"``. The format also selects the extension of the path that
        the launcher passes to ``write()``. Parquet is compact, carries a
        typed schema, and stores binary media without base64 inflation. Choose
        ``"jsonl"`` for small datasets that
        need to stay greppable or tolerate rows with different schemas.

    You can implement your own dataset by subclassing DatasetConfig and overriding the
    necessary methods, but you can also use one of our built-in datasets to pull from
    common sources like HuggingFace or Harbor.
    """

    type: DatasetType = DatasetType.DEFAULT
    id: str = ""
    input_key: str = "input"
    needs_chat_template: bool = True
    needs_refresh: bool = False
    output_format: Literal["parquet", "jsonl"] = "parquet"

    @property
    @abstractmethod
    def label_key(self) -> str:
        raise NotImplementedError("Datasets must set a label_key.")

    @abstractmethod
    def rows(self) -> Iterable[DatasetRow]:
        pass

    @property
    def name(self) -> str:
        return self.id

    def __init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        self._validate()

    def _validate(self) -> None:
        if self.output_format not in ("parquet", "jsonl"):
            raise TrainingGymConfigError(
                f"{type(self).__name__} has output_format="
                f"{self.output_format!r}; expected 'parquet' or 'jsonl'."
            )
        if not self.label_key:
            raise TrainingGymConfigError(
                f"{type(self).__name__} requires `label_key` to be set. "
                "Declare it as a class attribute on the dataset subclass."
            )

    def _expected_columns(self) -> set[str]:
        cols: set[str] = set()
        if self.input_key:
            cols.add(self.input_key)
        if self.label_key:
            cols.add(self.label_key)
        return cols

    def write(self, path: str) -> None:
        if self.output_format == "parquet":
            HFDataset.from_list(list(self.rows())).to_parquet(path)
        else:
            with open(path, "w") as f:
                for row in self.rows():
                    f.write(json.dumps(row) + "\n")

    def validate_write(self, path: str) -> None:
        """Sniff what ``write()`` wrote and confirm the columns the framework will index.

        Catches the common ``KeyError: 'label'`` (and friends) that otherwise
        only fire deep inside a Ray actor on a remote container, after image
        build and cluster bringup.
        """
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{type(self).__name__}.write() did not produce {path!r}. "
                "Ensure your write(path, ...) override writes to the `path` arg."
            )

        expected = self._expected_columns()
        if not expected:
            return

        try:
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq

                cols = set(pq.read_schema(path).names)
            elif path.endswith((".jsonl", ".json")):
                with open(path) as f:
                    first = f.readline().strip()
                if not first:
                    raise TrainingGymConfigError(f"{path!r} is empty")
                cols = set(json.loads(first).keys())
            else:
                return
        except Exception as e:  # don't shadow the user's real bug with a sniff bug
            print(
                f"[{type(self).__name__}.validate_write] could not sniff "
                f"{path!r} ({e!r}); skipping schema check."
            )
            return

        missing = expected - cols
        if missing:
            raise TrainingGymConfigError(
                f"{type(self).__name__}.write() wrote {path!r} but it is "
                f"missing required column(s) {sorted(missing)} "
                f"(input_key={self.input_key!r}, label_key={self.label_key!r}). "
                f"Columns present: {sorted(cols)}. "
                "Either rename the column(s) your write() writes, or set "
                "input_key/label_key on your DatasetConfig subclass to match."
            )


class HuggingFaceDataset(DatasetConfig):
    """Dataset backed by a HuggingFace ``datasets`` repo.

    Subclass and set ``hf_repo`` plus column mappings. When
    ``input_column`` and ``output_column`` are set, ``rows()`` formats
    each row as a chat message list plus a separate label field:
    ``{"messages": [{"role": "user", ...}], <label_key>: ...}``.
    A leading ``{"role": "system", ...}`` message is included when
    ``system_prompt`` is set. No assistant turn is emitted. The target
    from ``output_column`` is stored under ``label_key``.

    With the columns unmapped, rows from the Hub are yielded as they come and
    ``input_key`` names the prompt column directly.
    """

    type: DatasetType = DatasetType.HUGGING_FACE
    hf_repo: str = ""
    hf_split: str = "train"
    hf_config: str | None = None
    input_column: str = ""
    output_column: str = ""
    system_prompt: str = ""
    prompt_template: str = "{input}"
    n_rows: int = 0
    label_key: str = "label"

    def __init__(
        self, *, hf_split: str | None = None, n_rows: int | None = None
    ) -> None:
        if hf_split is not None:
            self.hf_split = hf_split
        if n_rows is not None:
            self.n_rows = n_rows
        if not self.id:
            self.id = f"{self.hf_repo}-{self.hf_split}-{uuid.uuid4()}"
        super().__init__()

    @property
    def name(self) -> str:
        return self.hf_repo

    @property
    def _emits_chat(self) -> bool:
        return bool(self.input_column and self.output_column)

    @property
    def input_key(self) -> str:
        if self._emits_chat:
            return "messages"
        return self.input_column or super().input_key

    def _load_hf_dataset(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_repo,
            self.hf_config,
            split=self.hf_split,
        )

        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))

        if self._emits_chat:
            ds = ds.map(self._to_chat)

        return ds

    def _to_chat(self, row: DatasetRow) -> DatasetRow:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = self.prompt_template.format(input=row[self.input_column])
        messages.append({"role": "user", "content": user_content})
        return {
            self.input_key: messages,
            self.label_key: str(row[self.output_column]),
        }

    def rows(self) -> Iterable[DatasetRow]:
        for row in self._load_hf_dataset():
            yield dict(row)

    def to_pandas(self):
        ds = self._load_hf_dataset()
        return ds.to_pandas()


class HarborDataset(DatasetConfig):
    """Dataset backed by a Harbor task directory structure.

    Each task folder contains an instruction file and optional label metadata.
    Tasks are discovered by globbing the task_root directory.

    Each instance represents one split. ``train_size`` / ``eval_size`` carve the
    task list into train and eval subsets, while ``train_repeats`` /
    ``eval_repeats`` duplicate rows within the selected split.
    """

    type: DatasetType = DatasetType.HARBOR
    dataset_name: str = ""
    path: str | None = None
    task_root: str = ""
    task_glob: str = "*"
    task_names: list[str] | None = None
    instruction_path: str = "instruction.md"
    label_metadata_path: str | None = None
    test_data_dir: str | None = None
    prompt_template: str = "{instruction}"
    system_prompt: str = ""
    train_size: int | None = None
    eval_size: int | None = None
    train_repeats: int = 1
    eval_repeats: int = 1
    shuffle_tasks: bool = False
    shuffle_seed: int = 0
    input_key: str = "messages"
    label_key: str = "label"

    def __init__(
        self,
        split: Literal["all", "train", "eval"] = "train",
        *,
        dataset_name: str | None = None,
        path: str | None = None,
        task_root: str | None = None,
        task_glob: str | None = None,
        task_names: list[str] | None = None,
        instruction_path: str | None = None,
        label_metadata_path: str | None = None,
        test_data_dir: str | None = None,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        train_size: int | None = None,
        eval_size: int | None = None,
        train_repeats: int | None = None,
        eval_repeats: int | None = None,
        shuffle_tasks: bool | None = None,
        shuffle_seed: int | None = None,
        needs_refresh: bool = False,
    ) -> None:
        self.split = split
        # Keep this while framework path resolution still uses ``hf_split``.
        self.hf_split = split
        if dataset_name is not None:
            self.dataset_name = dataset_name
        if path is not None:
            self.path = path
        if task_root is not None:
            self.task_root = task_root
        if task_glob is not None:
            self.task_glob = task_glob
        if task_names is not None:
            self.task_names = task_names
        if instruction_path is not None:
            self.instruction_path = instruction_path
        if label_metadata_path is not None:
            self.label_metadata_path = label_metadata_path
        if test_data_dir is not None:
            self.test_data_dir = test_data_dir
        if prompt_template is not None:
            self.prompt_template = prompt_template
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if train_size is not None:
            self.train_size = train_size
        if eval_size is not None:
            self.eval_size = eval_size
        if train_repeats is not None:
            self.train_repeats = train_repeats
        if eval_repeats is not None:
            self.eval_repeats = eval_repeats
        if shuffle_tasks is not None:
            self.shuffle_tasks = shuffle_tasks
        if shuffle_seed is not None:
            self.shuffle_seed = shuffle_seed
        self._needs_refresh = needs_refresh
        if not self.id:
            self.id = f"{self._id_slug()}-{split}-{uuid.uuid4()}"
        super().__init__()

    def _id_slug(self) -> str:
        if self.dataset_name:
            return self.dataset_name.replace("/", "-")
        if self.path:
            return self.path.replace("/", "_")
        if self.task_root:
            return self.task_root.replace("/", "_")
        return "harbor"

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def needs_refresh(self) -> bool:
        return self._needs_refresh

    def _harbor_dataset_ref(self) -> str:
        if "@" in self.dataset_name:
            return self.dataset_name
        return f"{self.dataset_name}@latest"

    def _harbor_cache_dir(self) -> Path:
        slug = self._harbor_dataset_ref().replace("/", "--").replace("@", "--")
        return Path.home() / ".cache" / "harbor" / "datasets" / slug

    def _download_harbor_dataset(self, cache_dir: Path) -> None:
        import subprocess

        ref = self._harbor_dataset_ref()
        harbor_bin = shutil.which("harbor")
        if harbor_bin is not None:
            cmd = [
                harbor_bin,
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        else:
            uvx_bin = shutil.which("uvx")
            if uvx_bin is None:
                raise FileNotFoundError(
                    "Harbor CLI not found. Install `harbor` or `uvx` to download "
                    f"{self.dataset_name!r}."
                )
            cmd = [
                uvx_bin,
                "harbor",
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(cache_dir),
            ]
        subprocess.run(cmd, check=True)

    def _pull_harbor_dataset(self) -> Path:
        cache_dir = self._harbor_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not any(cache_dir.rglob(self.instruction_path)):
            self._download_harbor_dataset(cache_dir)
        task_root = self._discover_task_root(cache_dir)
        if not any(task_root.rglob(self.instruction_path)):
            raise FileNotFoundError(f"No Harbor tasks found under {cache_dir}")
        return task_root

    def _discover_task_root(self, search_root: Path) -> Path:
        task_dirs = sorted(
            {
                instruction_file.parent
                for instruction_file in search_root.rglob(self.instruction_path)
                if instruction_file.is_file()
            }
        )
        if not task_dirs:
            return search_root
        if len(task_dirs) == 1:
            return task_dirs[0].parent
        import os

        return Path(os.path.commonpath([str(path) for path in task_dirs]))

    def _resolve_task_root(self) -> Path:
        if self.path:
            task_root = Path(self.path).resolve()
        elif self.dataset_name:
            task_root = self._pull_harbor_dataset()
        elif self.task_root:
            task_root = Path(self.task_root).resolve()
        else:
            raise TrainingGymConfigError(
                f"{type(self).__name__} requires dataset_name, path, or task_root"
            )
        if not task_root.exists():
            raise FileNotFoundError(f"task root does not exist: {task_root}")
        if not task_root.is_dir():
            raise TrainingGymConfigError(f"task root is not a directory: {task_root}")
        return task_root

    def _candidate_task_dirs(self, task_root: Path) -> list[Path]:
        if self.task_names is not None:
            return [
                (task_root / name).resolve()
                for name in self.task_names
                if (task_root / name).is_dir()
            ]
        return sorted(
            path.resolve() for path in task_root.glob(self.task_glob) if path.is_dir()
        )

    def _iter_task_dirs(self) -> list[Path]:
        task_root = self._resolve_task_root()
        task_dirs = self._candidate_task_dirs(task_root)
        if not task_dirs:
            discovered_root = self._discover_task_root(task_root)
            if discovered_root != task_root:
                task_root = discovered_root
                task_dirs = self._candidate_task_dirs(task_root)
        if self.shuffle_tasks:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(task_dirs)
        if not task_dirs:
            raise TrainingGymConfigError(f"No Harbor tasks found under {task_root}")
        if self.train_size is not None:
            max_tasks = self.train_size + (self.eval_size or 0)
            task_dirs = task_dirs[:max_tasks]
        return task_dirs

    def _read_label_metadata(self, task_dir: Path) -> dict[str, Any]:
        if not self.label_metadata_path:
            return {}
        metadata_path = task_dir / self.label_metadata_path
        if not metadata_path.exists():
            return {}
        if metadata_path.suffix == ".json":
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif metadata_path.suffix == ".toml":
            data = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            raise TrainingGymConfigError(
                f"Unsupported label metadata file type for {metadata_path}; expected .json or .toml"
            )
        if not isinstance(data, dict):
            raise TrainingGymConfigError(
                f"Label metadata must decode to an object: {metadata_path}"
            )
        return data

    def _read_test_data(self, task_dir: Path) -> list[dict[str, str]]:
        assert self.test_data_dir is not None
        tests_dir = task_dir / self.test_data_dir
        test_cases: list[dict[str, str]] = []
        if not tests_dir.is_dir():
            return test_cases
        for in_file in sorted(tests_dir.glob("*.in")):
            out_file = in_file.with_suffix(".out")
            if out_file.exists():
                test_cases.append(
                    {
                        "input": in_file.read_text(encoding="utf-8"),
                        "expected_output": out_file.read_text(encoding="utf-8"),
                    }
                )
        return test_cases

    def _build_label(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        rel = task_dir.relative_to(task_root)
        rel_with_root = (Path(task_root.name) / rel).as_posix()
        label: dict[str, Any] = {
            "harbor_task_name": task_dir.name,
            "harbor_task_path": task_dir.as_posix(),
            "harbor_task_rel": rel_with_root,
        }
        label.update(self._read_label_metadata(task_dir))
        if self.test_data_dir:
            label["test_cases"] = self._read_test_data(task_dir)
        return label

    def _format_prompt(
        self, *, instruction: str, task_dir: Path, label: dict[str, Any]
    ) -> str:
        context = {
            "instruction": instruction,
            "task_name": task_dir.name,
            "task_path": task_dir.as_posix(),
            **label,
        }
        return self.prompt_template.format(**context).strip()

    def _build_row(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        instruction_file = task_dir / self.instruction_path
        if not instruction_file.exists():
            raise FileNotFoundError(
                f"instruction file does not exist for Harbor task {task_dir.name}: {instruction_file}"
            )
        instruction = instruction_file.read_text(encoding="utf-8").strip()
        label = self._build_label(task_root, task_dir)
        user_prompt = self._format_prompt(
            instruction=instruction, task_dir=task_dir, label=label
        )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return {
            self.input_key: messages,
            self.label_key: json.dumps(label, separators=(",", ":")),
        }

    @staticmethod
    def _repeat_rows(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
        repeats = max(1, repeats)
        return [row for row in rows for _ in range(repeats)]

    def _split_rows(
        self, base_rows: list[DatasetRow]
    ) -> tuple[list[DatasetRow], list[DatasetRow]]:
        """Carve the task list into train / eval halves.

        Without a ``train_size`` there is nothing to hold out, so both splits get
        every task. An ``eval_size`` that runs past the end of the task list also
        falls back to the full list rather than yielding an empty eval split.
        """
        if self.train_size is None:
            return base_rows, base_rows
        train_size = max(1, min(int(self.train_size), len(base_rows)))
        eval_rows = (
            base_rows[train_size : train_size + (self.eval_size or 0)] or base_rows
        )
        return base_rows[:train_size], eval_rows

    def rows(self) -> Iterable[DatasetRow]:
        task_root = self._resolve_task_root()
        base_rows = [
            self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()
        ]
        if self.split == "all":
            return base_rows
        train_rows, eval_rows = self._split_rows(base_rows)
        if self.split == "train":
            return self._repeat_rows(train_rows, int(self.train_repeats))
        return self._repeat_rows(eval_rows, int(self.eval_repeats))

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.rows())


class MultimodalDataset(DatasetConfig):
    """Modality-agnostic dataset for image / audio / video RL.

    Each row pairs a text ``prompt`` with one or more ``media`` items and a
    ``label``. ``rows()`` emits the media verbatim under a column named by
    ``media_column`` (default ``"<modality>s"``), and the column is surfaced to
    the trainer/rollout via ``multimodal_keys`` (``{modality: media_column}``,
    e.g. slime's ``--multimodal-keys``). Media items may be URLs, local paths,
    or base64 data — whatever the serving engine accepts; the gym never
    inspects them.

    Pass ``rows=[{"prompt": str, "media": list, "label": Any}, ...]`` to the
    constructor, or subclass and override ``source_rows()`` to generate them.
    """

    modality: Literal["image", "audio", "video"] = "audio"
    # TODO(ben/joy): gate-check media at this boundary so the evals dashboard can
    # reliably visualize it. Two parts: (1) normalize each emitted media item to a
    # canonical, browser-renderable container per modality (audio->wav, image->png/
    # jpeg) instead of trusting whatever format the user brings — the audio tutorial's
    # dataset already re-encodes to wav, make it the convention here; (2) validate the
    # data-URI
    # MIME matches `modality`. Pairs with the dashboard fallback in EvalsPage.svelte.
    media_column: str = ""
    input_key: str = "prompt"
    label_key: str = "label"
    output_format: Literal["jsonl"] = "jsonl"

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        if self.modality not in ("image", "audio", "video"):
            raise TrainingGymConfigError(
                f"modality must be one of image/audio/video, got {self.modality!r}"
            )
        if not self.media_column:
            self.media_column = f"{self.modality}s"
        if self.input_key == self.media_column or self.label_key == self.media_column:
            raise TrainingGymConfigError(
                "media_column must differ from input_key and label_key"
            )
        self._source_rows = list(rows or [])
        if not self.id:
            self.id = f"mm-{self.modality}-{uuid.uuid4()}"
        super().__init__()

    @property
    def multimodal_keys(self) -> dict[str, str]:
        """The whole feature in one line: name the media column for the framework."""
        return {self.modality: self.media_column}

    def source_rows(self) -> Iterable[dict[str, Any]]:
        """Raw ``{"prompt", "media", "label"}`` triples; override to generate them."""
        return self._source_rows

    def _to_row(self, row: dict[str, Any]) -> DatasetRow:
        media = row["media"]
        return {
            self.input_key: row["prompt"],
            self.media_column: list(media)
            if isinstance(media, (list, tuple))
            else [media],
            self.label_key: row["label"],
        }

    def rows(self) -> Iterable[DatasetRow]:
        for row in self.source_rows():
            yield self._to_row(row)
