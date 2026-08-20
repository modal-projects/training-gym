"""Dataset config + ``prepare()`` hook, shared across training frameworks.

Pure data — each framework config writes its own converter from a
``DatasetConfig`` instance to its specific CLI flags (e.g. SlimeRecipe emits
``--prompt-data``, ``--input-key``, …).

Subclass and override ``prepare()`` to materialize the data into a shared
volume.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
import json
import random
import shutil
import tomllib
import uuid
from pathlib import Path

from modal_training_gym.common.errors import TrainingGymConfigError

DatasetRow = dict[str, Any]


class DatasetType(Enum):
    DEFAULT = "default"
    HUGGING_FACE = "hugging_face"
    HARBOR = "harbor"


class DatasetConfig:
    """Dataset configuration shared across training frameworks.

    Describes *what* the data is. Where it gets written on disk is decided
    by the recipe/launcher layer, not by the dataset itself.
    """

    _type: DatasetType = DatasetType.DEFAULT
    dataset_id: str = ""
    input_key: str = ""
    label_key: str = ""
    apply_chat_template: bool = True
    always_prepare: bool = False
    # When True (default), ``prepare()`` is expected to materialize every path in
    # ``eval_paths`` and the launcher validates them strictly. Datasets that use
    # a separate DatasetConfig instance for offline eval (Toolathlon, BFCL) set
    # this False so resolvers don't invent a companion ``eval.*`` file.
    writes_eval_paths: bool = True

    def __init__(self, **kwargs: Any) -> None:
        if not self.dataset_id:
            self.dataset_id = str(uuid.uuid4())
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()

    def _validate(self) -> None:
        """Required-field check; subclasses call this at the end of their own ``__init__``."""
        if not self.label_key:
            raise TrainingGymConfigError(
                f"{type(self).__name__} requires `label_key` to be set. "
                "It names the column on the materialized dataset that holds "
                "per-sample ground-truth / reward-function input. "
                'Declare it as a class attribute (`label_key = "label"`) on '
                "your subclass, or pass `label_key=...` as a kwarg. Frameworks "
                "like slime index `data[label_key]` at load time, so an unset "
                "value reliably crashes deep in a remote Ray actor."
            )

    @property
    def name(self) -> str:
        return self.dataset_id

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        """Materialize training data to ``path`` (and eval splits to ``eval_paths``)."""
        raise NotImplementedError(f"{type(self).__name__} has no prepare()")

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        """Load raw examples, optionally filtered by split."""
        raise NotImplementedError(f"{type(self).__name__} has no load()")

    def _expected_columns(self) -> set[str]:
        cols: set[str] = set()
        if self.input_key:
            cols.add(self.input_key)
        if self.label_key:
            cols.add(self.label_key)
        return cols

    def validate_prepared(self, path: str) -> None:
        """Sniff what ``prepare()`` wrote and confirm the columns the framework will index.

        Catches the common ``KeyError: 'label'`` (and friends) that otherwise
        only fire deep inside a Ray actor on a remote container, after image
        build and cluster bringup.
        """
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{type(self).__name__}.prepare() did not produce {path!r}. "
                "Ensure your prepare(path, ...) override writes to the `path` arg."
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
                f"[{type(self).__name__}.validate_prepared] could not sniff "
                f"{path!r} ({e!r}); skipping schema check."
            )
            return

        missing = expected - cols
        if missing:
            raise TrainingGymConfigError(
                f"{type(self).__name__}.prepare() wrote {path!r} but it is "
                f"missing required column(s) {sorted(missing)} "
                f"(input_key={self.input_key!r}, label_key={self.label_key!r}). "
                f"Columns present: {sorted(cols)}. "
                "Either rename the column(s) your prepare() writes, or set "
                "input_key/label_key on your DatasetConfig subclass to match."
            )


class HuggingFaceDataset(DatasetConfig):
    """Dataset backed by a HuggingFace ``datasets`` repo.

    Subclass and set ``hf_repo`` plus column mappings. When
    ``input_column`` and ``output_column`` are set, ``prepare()`` wraps
    each row into a prompt-only chat message list plus a separate label
    field: ``{"messages": [{"role": "user", ...}], <label_key>: ...}``.
    A leading ``{"role": "system", ...}`` message is included when
    ``system_prompt`` is set. No assistant turn is emitted — the target
    from ``output_column`` is stored under ``label_key``.
    """

    _type: DatasetType = DatasetType.HUGGING_FACE
    hf_repo: str = ""
    hf_split: str = "train"
    hf_config: str | None = None
    output_format: str = "parquet"
    input_column: str = ""
    output_column: str = ""
    system_prompt: str = ""
    prompt_template: str = "{input}"
    n_rows: int = 0
    label_key: str = "label"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.input_key and self.input_column and self.output_column:
            self.input_key = "messages"
        if "dataset_id" not in kwargs:
            self.dataset_id = f"{self.hf_repo}-{self.hf_split}-{uuid.uuid4()}"
        self._validate()

    @property
    def name(self) -> str:
        return self.hf_repo

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_repo,
            self.hf_config,
            split=self.hf_split,
        )
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds

    def _format_for_training(self, ds):
        if not (self.input_column and self.output_column):
            return ds

        in_col, out_col = self.input_column, self.output_column
        sys_prompt = self.system_prompt
        template = self.prompt_template
        label_key = self.label_key

        def _to_chat(row: dict) -> dict:
            user_content = template.format(input=row[in_col])
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": user_content})
            return {"messages": msgs, label_key: str(row[out_col])}

        return ds.map(_to_chat, remove_columns=ds.column_names)

    def to_pandas(self, *, formatted: bool = False):
        ds = self.load()
        if formatted:
            ds = self._format_for_training(ds)
        return ds.to_pandas()

    def _write_split(self, ds, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.output_format == "jsonl":
            ds.to_json(path, orient="records", lines=True)
        else:
            ds.to_parquet(path)

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        ds = self._format_for_training(self.load())
        self._write_split(ds, path)

        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_split(ds, eval_path)


class HarborDataset(DatasetConfig):
    """Dataset backed by a Harbor task directory structure.

    Each task folder contains an instruction file and optional label metadata.
    Tasks are discovered by globbing the task_root directory.
    """

    _type: DatasetType = DatasetType.HARBOR
    dataset_name: str = ""
    path: str | None = None
    task_root: str = ""
    task_glob: str = "*"
    task_names: list[str] | None = None
    instruction_path: str = "instruction.md"
    label_metadata_path: str | None = None
    test_data_dir: str | None = None
    output_format: str = "parquet"
    prompt_template: str = "{instruction}"
    system_prompt: str = ""
    train_size: int | None = None
    eval_size: int | None = None
    train_repeats: int = 1
    eval_repeats: int = 1
    shuffle_tasks: bool = False
    shuffle_seed: int = 0

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.input_key:
            self.input_key = "messages"
        if not self.label_key:
            self.label_key = "label"
        if "dataset_id" not in kwargs:
            if self.dataset_name:
                slug = self.dataset_name.replace("/", "-")
            elif self.path:
                slug = self.path.replace("/", "_")
            elif self.task_root:
                slug = self.task_root.replace("/", "_")
            else:
                slug = "harbor"
            self.dataset_id = f"{slug}-{uuid.uuid4()}"
        self._validate()

    @property
    def name(self) -> str:
        return self.dataset_name

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

    def _write_split(self, rows: list[dict[str, Any]], path: str) -> None:
        import os

        from datasets import Dataset

        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds = Dataset.from_list(rows)
        if self.output_format == "jsonl":
            ds.to_json(path, orient="records", lines=True)
            return
        ds.to_parquet(path)

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
            "messages": messages,
            "label": json.dumps(label, separators=(",", ":")),
        }

    @staticmethod
    def _repeat_rows(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
        repeats = max(1, repeats)
        return [row for row in rows for _ in range(repeats)]

    def load(self, split: Literal["all", "train", "eval"] = "all") -> Any:
        task_root = self._resolve_task_root()
        out = []
        for task_dir in self._iter_task_dirs():
            instruction_file = task_dir / self.instruction_path
            if not instruction_file.exists():
                raise FileNotFoundError(
                    f"instruction file does not exist for Harbor task {task_dir.name}: {instruction_file}"
                )
            label = self._build_label(task_root, task_dir)
            out.append(
                {
                    "task_name": task_dir.name,
                    "task_path": task_dir.as_posix(),
                    "instruction": instruction_file.read_text(encoding="utf-8").strip(),
                    "label": label,
                }
            )
        if self.train_size is not None:
            if split == "train":
                return out[: self.train_size]
            if split == "eval":
                return out[self.train_size : self.train_size + (self.eval_size or 0)]
        return out

    def to_pandas(self, *, formatted: bool = False):
        import pandas as pd

        if not formatted:
            return pd.DataFrame(self.load())

        task_root = self._resolve_task_root()
        rows = [
            self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()
        ]
        return pd.DataFrame(rows)

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        task_root = self._resolve_task_root()
        base_rows = [
            self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()
        ]

        if self.train_size is None:
            train_base = base_rows
            eval_base = base_rows
        else:
            train_size = max(1, min(int(self.train_size), len(base_rows)))
            train_base = base_rows[:train_size]
            eval_base = (
                base_rows[train_size : train_size + (self.eval_size or 0)] or base_rows
            )

        train_rows = self._repeat_rows(train_base, int(self.train_repeats))
        eval_rows = self._repeat_rows(eval_base, int(self.eval_repeats))

        self._write_split(train_rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_split(eval_rows, eval_path)


class MultimodalDataset(DatasetConfig):
    """Modality-agnostic dataset for image / audio / video RL.

    Each row pairs a text ``prompt`` with one or more ``media`` items and a
    ``label``. ``prepare()`` writes the media verbatim into a column named by
    ``media_column`` (default ``"<modality>s"``), and the column is surfaced to
    the trainer/rollout via ``multimodal_keys`` (``{modality: media_column}``,
    e.g. slime's ``--multimodal-keys``). Media items may be URLs, local paths,
    or base64 data — whatever the serving engine accepts; the gym never
    inspects them.

    Pass ``rows=[{"prompt": str, "media": list, "label": Any}, ...]`` or
    subclass and override the ``rows`` property.
    """

    input_key: str = "prompt"
    label_key: str = "label"
    modality: Literal["image", "audio", "video"] = "audio"
    # TODO(ben/joy): gate-check media at this boundary so the evals dashboard can
    # reliably visualize it. Two parts: (1) normalize each emitted media item to a
    # canonical, browser-renderable container per modality (audio->wav, image->png/
    # jpeg) instead of trusting whatever format the user brings — the audio tutorial's
    # dataset already re-encodes to wav, make it the convention here; (2) validate the
    # data-URI
    # MIME matches `modality`. Pairs with the dashboard fallback in EvalsPage.svelte.
    media_column: str = ""
    output_format: str = "jsonl"

    def __init__(self, rows: list[dict[str, Any]] | None = None, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
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
        # The whole feature in one line: name the media column for the framework.
        self.multimodal_keys = {self.modality: self.media_column}
        self._rows = list(rows or [])
        if not self.dataset_id:
            self.dataset_id = f"mm-{self.modality}-{uuid.uuid4()}"
        self._validate()

    @property
    def rows(self) -> list[DatasetRow]:
        return self._rows

    def _to_row(self, r: dict[str, Any]) -> DatasetRow:
        media = r["media"]
        return {
            self.input_key: r["prompt"],
            self.media_column: list(media)
            if isinstance(media, (list, tuple))
            else [media],
            self.label_key: r["label"],
        }

    def load(self) -> list[DatasetRow]:
        return [self._to_row(r) for r in self.rows]

    def _write_jsonl(self, rows: list[dict[str, Any]], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        rows = self.load()
        self._write_jsonl(rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_jsonl(rows, eval_path)


class EmbeddingProjectorDataset(DatasetConfig):
    """Supervised dataset for projector training: messages plus token embeddings.

    Each row is a chat conversation together with the embeddings an external
    encoder produced and the token positions they occupy::

        EmbeddingProjectorDataset(
            rows=[
                {
                    "messages": [
                        {"role": "user", "content": "<emb> what does this bind?"},
                        {"role": "assistant", "content": "..."},
                    ],
                    "embeddings": [[0.1, ...], [0.2, ...]],  # [n_tokens, input_dim]
                    "positions": [3, 4],                     # positions in the token ids
                },
            ]
        )

    The embeddings ride in the ``metadata`` column, which miles carries onto
    ``Sample.metadata`` untouched, and the projector rollout function turns them
    into the tensors the model's forward takes. Positions index the row's own
    token sequence; miles offsets them when it packs samples together.

    Targets come from the conversation itself, so ``label_key`` holds nothing a
    reward function would read — supervised training has no reward — but the
    column exists because the loaders index it.
    """

    input_key: str = "messages"
    label_key: str = "label"
    embeddings_key: str = "projector_embeddings"
    positions_key: str = "projector_positions"
    output_format: str = "jsonl"
    # The conversation reaches miles as a list of messages, which the loss-mask
    # generator needs; a rendered string would leave it nothing to split on.
    apply_chat_template: bool = False
    # Synthetic rows are described, not stored: see ``synthetic()``.
    synthetic_rows: int = 0
    synthetic_input_dim: int = 0
    synthetic_seed: int = 0
    synthetic_tokens: int = 1600

    def __init__(self, rows: list[dict[str, Any]] | None = None, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._rows = list(rows or [])
        if not self.dataset_id:
            self.dataset_id = f"projector-{uuid.uuid4()}"
        self._validate()
        for row in self._rows:
            self._check_row(row)

    @staticmethod
    def _check_row(row: dict[str, Any]) -> None:
        embeddings, positions = row.get("embeddings"), row.get("positions")
        if embeddings is None or positions is None:
            raise TrainingGymConfigError(
                "each EmbeddingProjectorDataset row needs 'embeddings' and "
                f"'positions'; got keys {sorted(row)}"
            )
        if len(embeddings) != len(positions):
            raise TrainingGymConfigError(
                f"{len(embeddings)} embedding row(s) for {len(positions)} "
                "position(s): every embedding occupies exactly one token"
            )

    @classmethod
    def synthetic(
        cls, n_rows: int, input_dim: int, seed: int = 0, tokens: int = 1600
    ) -> "EmbeddingProjectorDataset":
        """Random embeddings on a fixed conversation, for wiring validation.

        A projector-only run needs an encoder's embeddings, and no public dataset
        ships them, so a smoke test that only has to prove the plumbing (data →
        rollout → forward → loss → projector gradient) generates them. Loss goes
        down on noise no more than it should — read this as a wiring check, not a
        learning curve.

        The rows are generated at prepare time from the seed rather than held in
        the instance: the dataset is cloudpickled into the container, and a
        thousand 1536-wide vectors do not belong in that payload.

        ``always_prepare`` is on: the on-volume path is derived from the class
        name, so without it a later run with a different row count or embedding
        width would silently train on the file an earlier run left behind.

        The filler is one repeated pattern, so it is also nearly free for a base
        model to predict: expect a training loss two orders of magnitude below
        what real prose gives (~0.006 rather than ~1.4 on Qwen3.6-35B-A3B). That
        says nothing about the projector, whose inputs are noise either way.

        ``tokens`` is a rough per-sample length, long by default so a
        validation step exercises a sequence a real projector run would see —
        a two-sentence conversation is a few dozen tokens, well under the
        sparse-attention regime a DSA model like GLM-5.2 trains in.
        """
        return cls(
            synthetic_rows=n_rows,
            synthetic_input_dim=input_dim,
            synthetic_seed=seed,
            synthetic_tokens=tokens,
            always_prepare=True,
        )

    def _synthetic_rows(self) -> list[dict[str, Any]]:
        rng = random.Random(self.synthetic_seed)
        # ~1 token per word, split so both the prompt and the response the loss
        # is taken over are long.
        filler = " ".join(
            f"token{n}" for n in range(max(self.synthetic_tokens, 8) // 2)
        )
        return [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<emb> describe sample {i}. {filler}",
                    },
                    {
                        "role": "assistant",
                        "content": f"Sample {i} is a placeholder. {filler}",
                    },
                ],
                "embeddings": [
                    [rng.gauss(0.0, 1.0) for _ in range(self.synthetic_input_dim)]
                ],
                "positions": [1],
            }
            for i in range(self.synthetic_rows)
        ]

    @property
    def rows(self) -> list[DatasetRow]:
        return self._rows or self._synthetic_rows()

    def _to_row(self, r: dict[str, Any]) -> DatasetRow:
        return {
            self.input_key: r["messages"],
            self.label_key: r.get("label", ""),
            "metadata": {
                # Normalized element-wise: encoder output arrives as a numpy
                # array or a torch tensor as often as a list, and neither's
                # scalars are JSON-serializable.
                self.embeddings_key: [[float(v) for v in e] for e in r["embeddings"]],
                self.positions_key: [int(p) for p in r["positions"]],
            },
        }

    def load(self) -> list[DatasetRow]:
        return [self._to_row(r) for r in self.rows]

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        rows = self.load()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        for target in [path, *(eval_paths or {}).values()]:
            with open(target, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
