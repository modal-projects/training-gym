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
import uuid

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
