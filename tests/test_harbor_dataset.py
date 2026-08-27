"""HarborDataset's encouraged path: subclass and declare the task source."""

import json

import pytest

from modal_training_gym import HarborDataset
from modal_training_gym.common.errors import TrainingGymConfigError


def _write_task(root, name, instruction):
    task_dir = root / name
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "task.toml").write_text('difficulty = "easy"\n')
    return task_dir


def test_subclass_declares_task_source(tmp_path):
    _write_task(tmp_path, "hello", "write hello.txt")

    class HelloWorld(HarborDataset):
        task_root = str(tmp_path)
        label_metadata_path = "task.toml"
        train_repeats = 2
        system_prompt = "be brief"
        output_format = "jsonl"

    dataset = HelloWorld()
    assert dataset.input_key == "messages"
    assert dataset.label_key == "label"

    path = tmp_path / "out" / "train.jsonl"
    dataset.prepare(str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert json.loads(rows[0]["label"])["difficulty"] == "easy"

    dataset.validate_prepared(str(path))


def test_kwargs_override_class_attrs(tmp_path):
    _write_task(tmp_path, "hello", "write hello.txt")

    class HelloWorld(HarborDataset):
        task_root = str(tmp_path)
        train_repeats = 2

    assert HelloWorld(train_repeats=5).train_repeats == 5


def test_missing_task_source_fails_at_config_time():
    class NoSource(HarborDataset):
        label_metadata_path = "task.toml"

    with pytest.raises(TrainingGymConfigError, match="requires a task source"):
        NoSource()
