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
    """

    prompt_data: str = ""
    eval_prompt_data: list[str] | str | None = None
    input_key: str = ""
    label_key: str = ""
    apply_chat_template: bool = True
    rollout_shuffle: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def prepare(self) -> None:
        """Download and/or preprocess the dataset into the data volume."""
        raise NotImplementedError(f"{type(self).__name__} has no prepare()")


class HuggingFaceDataset(DatasetConfig):
    """Dataset backed by a HuggingFace ``datasets`` repo.

    Subclass and set ``hf_repo`` plus column mappings. When
    ``input_column`` and ``output_column`` are set, ``prepare()``
    auto-wraps rows into chat-message format
    (``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``).

    ## Fields

    hf_repo : str
        HuggingFace dataset repo ID (e.g. ``"openai/gsm8k"``).
    hf_split : str
        Dataset split to load. Default ``"train"``.
    hf_config : str | None
        Dataset config name (for multi-config datasets). Default ``None``.
    data_root : str
        Root directory on the data volume. Default ``"/data"``.
    output_format : str
        Output format: ``"parquet"`` or ``"jsonl"``. Default ``"parquet"``.
    input_column : str
        Source column for user messages. When set with ``output_column``,
        rows are auto-converted to chat format. Default ``""``.
    output_column : str
        Source column for assistant messages. Default ``""``.
    system_prompt : str
        Optional system message prepended to every conversation.
        Default ``""``.
    """

    hf_repo: str = ""
    hf_split: str = "train"
    hf_config: str | None = None
    data_root: str = "/data"
    output_format: str = "parquet"
    input_column: str = ""
    output_column: str = ""
    system_prompt: str = ""

    def __init__(self, data_root: str = "/data", **kwargs: Any) -> None:
        self.data_root = str(data_root)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not self.prompt_data:
            name = self.hf_repo.replace("/", "_")
            ext = "jsonl" if self.output_format == "jsonl" else "parquet"
            self.prompt_data = f"{self.data_root}/{name}/train.{ext}"

    def prepare(self) -> None:
        import os

        from datasets import load_dataset

        os.makedirs(os.path.dirname(self.prompt_data), exist_ok=True)
        ds = load_dataset(
            self.hf_repo,
            self.hf_config,
            split=self.hf_split,
        )

        if self.input_column and self.output_column:
            in_col, out_col = self.input_column, self.output_column
            sys_prompt = self.system_prompt

            def _to_chat(row: dict) -> dict:
                msgs = []
                if sys_prompt:
                    msgs.append({"role": "system", "content": sys_prompt})
                msgs.append({"role": "user", "content": row[in_col]})
                msgs.append({"role": "assistant", "content": row[out_col]})
                return {"messages": msgs}

            ds = ds.map(_to_chat, remove_columns=ds.column_names)

        if self.output_format == "jsonl":
            ds.to_json(self.prompt_data, orient="records", lines=True)
        else:
            ds.to_parquet(self.prompt_data)
