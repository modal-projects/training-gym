"""Dataset configs shared across training frameworks.

`DatasetConfig` owns the knobs a framework needs to locate + interpret a
dataset plus a `prepare()` method that materializes the data into a volume.
Framework configs (e.g. `SlimeConfig`) accept a `DatasetConfig` instance as a
field; the dataset's fields get merged into the framework's CLI args.

Conversion goes both ways:
  - `DatasetConfig.to_fields()`          → dict of dataset field values
  - `DatasetConfig.from_fields(flat)`    → DatasetConfig from a framework's
                                           flat field dict (reverse direction)
"""

from __future__ import annotations

from typing import Any

# Field names a DatasetConfig contributes to a framework's flat field dict.
# Extending this exposes a new dataset-scoped knob to the framework.
DATASET_FIELDS: tuple[str, ...] = (
    "prompt_data",
    "eval_prompt_data",
    "input_key",
    "label_key",
    "apply_chat_template",
    "rollout_shuffle",
    "rm_type",
)


class DatasetConfig:
    """Base configuration for a training dataset.

    Subclass, set path + interpretation fields, and override `prepare()` to
    download/preprocess into a shared volume.
    """

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

    def to_fields(self) -> dict[str, Any]:
        """Flat dict of this dataset's values, keyed by framework field name."""
        return {k: getattr(self, k) for k in DATASET_FIELDS if hasattr(self, k)}

    @classmethod
    def from_fields(cls, fields: dict[str, Any], **extra: Any) -> "DatasetConfig":
        """Build a bare `DatasetConfig` from a flat dict of dataset fields.

        Used for the reverse direction — pulling the dataset portion out of a
        framework config that has the fields set directly. The returned
        instance has no `prepare()`; use a concrete subclass for that.
        """
        ds_kwargs = {k: fields[k] for k in DATASET_FIELDS if k in fields}
        ds_kwargs.update(extra)
        return cls(**ds_kwargs)


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
