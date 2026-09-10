import json
from types import SimpleNamespace

import pytest

from modal_training_gym.common.dataset import (
    DatasetConfig,
    HarborDataset,
    HuggingFaceDataset,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.launcher_helpers import (
    run_prepare_dataset,
    write_dataset_if_needed,
)
from modal_training_gym.train_recipes.base import BaseTrainRecipe


class RowsDataset(DatasetConfig):
    def __init__(self, key: str | None, value: str = "row") -> None:
        self.key = key
        self.value = value
        self.write_count = 0

    def cache_key(self) -> str | None:
        return self.key

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        yield {"prompt": self.value, "label": self.value}

    def write(self, path: str) -> None:
        self.write_count += 1
        super().write(path)


class FakeVolume:
    def __init__(self) -> None:
        self.reload_count = 0
        self.commit_count = 0

    def reload(self) -> None:
        self.reload_count += 1

    def commit(self) -> None:
        self.commit_count += 1


def test_resolve_data_paths_uses_cache_key():
    dataset = RowsDataset("org/data-train")
    path = BaseTrainRecipe._resolve_data_paths(dataset)
    assert path.startswith("/data/org-data-train-")
    assert path.endswith(".jsonl")
    assert "%2F" not in path
    assert path != BaseTrainRecipe._resolve_data_paths(RowsDataset("org_data-train"))


def test_resolve_data_paths_generates_fresh_random_id():
    dataset = RowsDataset(None)
    first = BaseTrainRecipe._resolve_data_paths(dataset)
    assert BaseTrainRecipe._resolve_data_paths(dataset) != first
    assert BaseTrainRecipe._resolve_data_paths(RowsDataset(None)) != first
    assert not hasattr(dataset, "_materialization_id")


def test_hugging_face_dataset_pins_latest_revision(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.dataset_info",
        lambda repo: SimpleNamespace(sha=f"{repo}-sha"),
    )

    dataset = HuggingFaceDataset(
        hf_repo="org/data",
        input_column="prompt",
        output_column="answer",
    )
    explicitly_pinned = HuggingFaceDataset(
        hf_repo="org/data",
        hf_revision="org/data-sha",
        input_column="prompt",
        output_column="answer",
    )

    assert dataset.hf_revision == "org/data-sha"
    assert dataset.cache_key() == explicitly_pinned.cache_key()


def test_hugging_face_revision_controls_cache_and_loading(monkeypatch):
    loaded = object()
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    first = HuggingFaceDataset(
        hf_repo="org/data",
        hf_revision="revision-a",
        input_column="prompt",
        output_column="answer",
        input_format="raw",
    )
    second = HuggingFaceDataset(
        hf_repo="org/data",
        hf_revision="revision-b",
        input_column="prompt",
        output_column="answer",
        input_format="raw",
    )

    assert first.cache_key() != second.cache_key()
    assert first._load_hf_dataset() is loaded
    assert calls == [
        (
            ("org/data", "default"),
            {"split": "train", "revision": "revision-a"},
        )
    ]


def test_dataset_fields_use_discrete_eval_dataset():
    train_dataset = RowsDataset("train")
    eval_dataset = RowsDataset("eval")
    fields = BaseTrainRecipe._dataset_to_fields(
        train_dataset,
        eval_dataset,
        dataset_path="/data/train.jsonl",
        eval_dataset_path="/data/eval.jsonl",
    )
    assert fields["prompt_data"] == "/data/train.jsonl"
    assert fields["eval_prompt_data"] == ["eval", "/data/eval.jsonl"]
    assert fields["input_key"] == "prompt"
    assert fields["label_key"] == "label"


def test_write_dataset_if_needed_reuses_valid_file(tmp_path):
    dataset = RowsDataset("train")
    path = str(tmp_path / "train.jsonl")
    assert write_dataset_if_needed(dataset, path)
    assert not write_dataset_if_needed(dataset, path)
    assert dataset.write_count == 1
    assert json.loads((tmp_path / "train.jsonl").read_text()) == {
        "prompt": "row",
        "label": "row",
    }


def test_write_caller_creates_parent_directory(tmp_path):
    dataset = RowsDataset("train")
    path = str(tmp_path / "nested" / "train.jsonl")
    with pytest.raises(FileNotFoundError):
        dataset.write(path)
    assert write_dataset_if_needed(dataset, path)


def test_run_prepare_dataset_writes_train_and_eval(tmp_path):
    train_dataset = RowsDataset("train", "training")
    eval_dataset = RowsDataset("eval", "evaluation")
    volume = FakeVolume()

    run_prepare_dataset(
        train_dataset,
        eval_dataset,
        volume,
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "eval.jsonl"),
    )

    assert train_dataset.write_count == 1
    assert eval_dataset.write_count == 1
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_eval_dataset_fields_must_match_training_dataset():
    class OtherInputDataset(RowsDataset):
        def input_key(self) -> str:
            return "messages"

    class OtherLabelDataset(RowsDataset):
        def label_key(self) -> str:
            return "answer"

    class OtherChatTemplateDataset(RowsDataset):
        def apply_chat_template(self) -> bool:
            return False

    with pytest.raises(TrainingGymConfigError, match="same input_key"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherInputDataset("eval")
        )
    with pytest.raises(TrainingGymConfigError, match="same label_key"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherLabelDataset("eval")
        )
    with pytest.raises(TrainingGymConfigError, match="same apply_chat_template"):
        BaseTrainRecipe._validate_datasets(
            RowsDataset("train"), OtherChatTemplateDataset("eval")
        )


def test_harbor_instances_select_discrete_splits(tmp_path):
    for name in ("one", "two", "three"):
        task = tmp_path / name
        task.mkdir()
        (task / "instruction.md").write_text(f"solve {name}")

    train = HarborDataset(
        task_root=str(tmp_path),
        split="train",
        train_size=2,
        eval_size=1,
        train_repeats=2,
    )
    evaluation = HarborDataset(
        task_root=str(tmp_path),
        split="eval",
        train_size=2,
        eval_size=1,
    )

    train_rows = list(train.rows())
    eval_rows = list(evaluation.rows())
    assert len(train_rows) == 4
    assert len(eval_rows) == 1
    train_prompts = {row["messages"][0]["content"] for row in train_rows}
    assert eval_rows[0]["messages"][0]["content"] not in train_prompts
    assert train.cache_key() != evaluation.cache_key()


def test_harbor_dataset_pins_latest_version(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "modal_training_gym.common.dataset.shutil.which",
        lambda executable: "/usr/bin/harbor" if executable == "harbor" else None,
    )

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps({"version": "1.2.3"}))

    monkeypatch.setattr("subprocess.run", fake_run)

    dataset = HarborDataset(dataset_name="harbor/example")
    explicitly_pinned = HarborDataset(dataset_name="harbor/example@1.2.3")

    assert dataset._latest_version == "1.2.3"
    assert dataset._harbor_dataset_ref() == "harbor/example@1.2.3"
    assert dataset.cache_key() == explicitly_pinned.cache_key()
    assert calls == [
        (
            [
                "/usr/bin/harbor",
                "version",
                "show",
                "harbor/example@latest",
                "--json",
            ],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_harbor_dataset_uses_latest_content_hash(monkeypatch):
    monkeypatch.setattr(
        "modal_training_gym.common.dataset.shutil.which",
        lambda executable: "/usr/bin/harbor" if executable == "harbor" else None,
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps({"version": None, "content_hash": "sha256:abc"})
        ),
    )

    dataset = HarborDataset(dataset_name="harbor/example")

    assert dataset._latest_version == "sha256:abc"
    assert dataset._harbor_dataset_ref() == "harbor/example@sha256:abc"


def test_harbor_always_download_disables_materialization_reuse():
    dataset = HarborDataset(
        dataset_name="harbor/example@1.2.3",
        always_download=True,
    )

    assert dataset.cache_key() is None
    assert BaseTrainRecipe._resolve_data_paths(
        dataset
    ) != BaseTrainRecipe._resolve_data_paths(dataset)
