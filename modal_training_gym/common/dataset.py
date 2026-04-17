"""Dataset config + `prepare()` hook, shared across training frameworks.

Pure data — each framework config writes its own converter from a
`DatasetConfig` instance to its specific CLI flags (e.g. SlimeConfig emits
`--prompt-data`, `--input-key`, …; ms-swift emits only `--dataset <path>`).

Subclass and override `prepare()` to materialize the data into a shared
volume.
"""

from __future__ import annotations

from typing import Any


class DatasetConfig:
    prompt_data: str = ""
    eval_prompt_data: list[str] | str | None = None
    input_key: str = ""
    label_key: str = ""
    apply_chat_template: bool = True
    rollout_shuffle: bool = True
    rm_type: str = ""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def prepare(self) -> None:
        """Download and/or preprocess the dataset into the data volume."""
        raise NotImplementedError(f"{type(self).__name__} has no prepare()")


class _GSM8KDataset(DatasetConfig):
    """Example: Zhuzilin's GSM8K parquet dump (grade-school math)."""

    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    def __init__(self, data_path: Any) -> None:
        self._data_path = str(data_path)
        self.prompt_data = f"{self._data_path}/gsm8k/train.parquet"
        self.eval_prompt_data = ["gsm8k", f"{self._data_path}/gsm8k/test.parquet"]

    def prepare(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("zhuzilin/gsm8k")
        ds["train"].to_parquet(f"{self._data_path}/gsm8k/train.parquet")
        ds["test"].to_parquet(f"{self._data_path}/gsm8k/test.parquet")
