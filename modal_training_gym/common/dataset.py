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
    """Dataset configuration shared across training frameworks.

    Subclass this and override ``prepare()`` to materialize training data
    into the shared volume. Each framework config converts these fields
    into its own CLI flags (e.g. SlimeConfig emits ``--prompt-data``,
    ``--input-key``, etc.).

    ## Fields

    prompt_data : str
        Path to the training data file (e.g. a ``.parquet`` file on the
        data volume). Default ``""``.
    eval_prompt_data : list[str] | str | None
        Evaluation data path(s). Can be a single path, a list of paths,
        or ``None`` to skip evaluation. Default ``None``.
    input_key : str
        Column/key name for model input in the dataset. Default ``""``.
    label_key : str
        Column/key name for labels/targets in the dataset. Default ``""``.
    apply_chat_template : bool
        Whether to apply the model's chat template to inputs. Default ``True``.
    rollout_shuffle : bool
        Whether to shuffle data during rollout generation. Default ``True``.
    rm_type : str
        Reward model type identifier (e.g. ``"math"``). Default ``""``.
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
