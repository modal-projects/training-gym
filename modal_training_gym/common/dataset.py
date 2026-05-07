"""Dataset config + ``prepare()`` hook, shared across training frameworks.

Pure data — each framework config writes its own converter from a
``DatasetConfig`` instance to its specific CLI flags (e.g. SlimeRecipe emits
``--prompt-data``, ``--input-key``, …).

Subclass and override ``prepare()`` to materialize the data into a shared
volume.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any
import json
import random
import tomllib
import uuid
from pathlib import Path

DatasetRow = dict[str, Any]

class DatasetConfig:
    """Dataset configuration shared across training frameworks.

    Describes *what* the data is. Where it gets written on disk is decided
    by the recipe/launcher layer, not by the dataset itself.
    """

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_key: str = ""
    label_key: str = ""
    apply_chat_template: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        """Materialize training data to ``path`` (and eval splits to ``eval_paths``)."""
        raise NotImplementedError(f"{type(self).__name__} has no prepare()")

    def load(self) -> Any:
        """Load raw examples for local inspection or evaluation."""
        raise NotImplementedError(f"{type(self).__name__} has no load()")


class HuggingFaceDataset(DatasetConfig):
    """Dataset backed by a HuggingFace ``datasets`` repo.

    Subclass and set ``hf_repo`` plus column mappings. When
    ``input_column`` and ``output_column`` are set, ``prepare()``
    auto-wraps rows into chat-message format
    (``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``).
    """

    hf_repo: str = ""
    hf_split: str = "train"
    hf_config: str | None = None
    output_format: str = "parquet"
    input_column: str = ""
    output_column: str = ""
    system_prompt: str = ""
    prompt_template: str = ""
    n_rows: int = 0

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.input_key and self.input_column and self.output_column:
            self.input_key = "messages"
        if "dataset_id" not in kwargs:
            self.dataset_id = f"{self.hf_repo}-{self.hf_split}-{uuid.uuid4()}"

    def load(self) -> Any:
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

        def _to_chat(row: dict) -> dict:
            user_content = row[in_col]
            if template:
                user_content = template.format(input=user_content)
            msgs = []
            if sys_prompt:
                msgs.append({"role": "system", "content": sys_prompt})
            msgs.append({"role": "user", "content": user_content})
            msgs.append({"role": "assistant", "content": row[out_col]})
            return {"messages": msgs}

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
                eval_ds = self._format_for_training(self.load())
                self._write_split(eval_ds, eval_path)


class HarborDataset(DatasetConfig):
    task_root: str = ""
    task_glob: str = "*"
    task_names: list[str] | None = None
    instruction_path: str = "instruction.md"
    label_metadata_path: str | None = None
    output_format: str = "parquet"
    prompt_template: str = "{instruction}"
    system_prompt: str = ""
    train_size: int | None = None
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
            root_slug = self.task_root.replace("/", "_") if self.task_root else "harbor"
            self.dataset_id = f"{root_slug}-{uuid.uuid4()}"

    def _write_split(self, rows: list[dict[str, Any]], path: str) -> None:
        import os

        from datasets import Dataset

        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds = Dataset.from_list(rows)
        if self.output_format == "jsonl":
            ds.to_json(path, orient="records", lines=True)
            return
        ds.to_parquet(path)

    def _resolve_task_root(self) -> Path:
        if not self.task_root:
            raise ValueError(f"{type(self).__name__} requires task_root")
        task_root = Path(self.task_root).resolve()
        if not task_root.exists():
            raise FileNotFoundError(f"task_root does not exist: {task_root}")
        if not task_root.is_dir():
            raise ValueError(f"task_root is not a directory: {task_root}")
        return task_root

    def _iter_task_dirs(self) -> list[Path]:
        task_root = self._resolve_task_root()
        if self.task_names is not None:
            task_dirs = [
                (task_root / name).resolve()
                for name in self.task_names
                if (task_root / name).is_dir()
            ]
        else:
            task_dirs = sorted(path.resolve() for path in task_root.glob(self.task_glob) if path.is_dir())
        if self.shuffle_tasks:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(task_dirs)
        if not task_dirs:
            raise ValueError(f"No Harbor tasks found under {task_root}")
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
            raise ValueError(
                f"Unsupported label metadata file type for {metadata_path}; expected .json or .toml"
            )
        if not isinstance(data, dict):
            raise ValueError(f"Label metadata must decode to an object: {metadata_path}")
        return data

    def _build_label(self, task_root: Path, task_dir: Path) -> dict[str, Any]:
        rel = task_dir.relative_to(task_root)
        rel_with_root = (Path(task_root.name) / rel).as_posix()
        label = {
            "harbor_task_name": task_dir.name,
            "harbor_task_path": task_dir.as_posix(),
            "harbor_task_rel": rel_with_root,
        }
        label.update(self._read_label_metadata(task_dir))
        return label

    def _format_prompt(self, *, instruction: str, task_dir: Path, label: dict[str, Any]) -> str:
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
        user_prompt = self._format_prompt(instruction=instruction, task_dir=task_dir, label=label)
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

    def load(self) -> Any:
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
        return out

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        task_root = self._resolve_task_root()
        base_rows = [self._build_row(task_root, task_dir) for task_dir in self._iter_task_dirs()]

        if self.train_size is None:
            train_base = base_rows
            eval_base = base_rows
        else:
            train_size = max(1, min(int(self.train_size), len(base_rows)))
            train_base = base_rows[:train_size]
            eval_base = base_rows[train_size:] or base_rows

        train_rows = self._repeat_rows(train_base, int(self.train_repeats))
        eval_rows = self._repeat_rows(eval_base, int(self.eval_repeats))

        self._write_split(train_rows, path)
        if eval_paths:
            for eval_path in eval_paths.values():
                self._write_split(eval_rows, eval_path)
