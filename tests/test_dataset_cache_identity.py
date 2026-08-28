import pytest

from modal_training_gym.common.dataset import HarborDataset, HuggingFaceDataset
from modal_training_gym.train_recipes.base import BaseTrainRecipe


class ExampleHuggingFaceDataset(HuggingFaceDataset):
    hf_repo = "example/dataset"
    input_column = "question"
    output_column = "answer"


class AlternateHuggingFaceDataset(ExampleHuggingFaceDataset):
    def rows(self):
        return []


class AlternateHarborDataset(HarborDataset):
    def rows(self):
        return []


def _data_path(dataset):
    return BaseTrainRecipe._resolve_data_path(dataset, "train")


@pytest.mark.parametrize(
    ("baseline", "changed"),
    [
        (
            ExampleHuggingFaceDataset(hf_split="train"),
            ExampleHuggingFaceDataset(hf_split="validation"),
        ),
        (
            ExampleHuggingFaceDataset(n_rows=10),
            ExampleHuggingFaceDataset(n_rows=20),
        ),
    ],
)
def test_huggingface_materialization_path_tracks_row_selection(
    baseline, changed
) -> None:
    assert _data_path(baseline) != _data_path(changed)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("hf_config", "alternate"),
        ("hf_repo", "example/alternate"),
        ("input_column", "prompt"),
        ("label_key", "target"),
        ("output_column", "target"),
        ("output_format", "jsonl"),
        ("prompt_template", "Question: {input}"),
        ("system_prompt", "Answer the question."),
    ],
)
def test_huggingface_materialization_path_tracks_row_shape(attribute, value) -> None:
    baseline = ExampleHuggingFaceDataset()
    changed = ExampleHuggingFaceDataset()
    setattr(changed, attribute, value)

    assert _data_path(baseline) != _data_path(changed)


def test_huggingface_materialization_path_tracks_input_key() -> None:
    class InputKeyHuggingFaceDataset(ExampleHuggingFaceDataset):
        input_key = "messages"

    baseline = InputKeyHuggingFaceDataset()
    changed = InputKeyHuggingFaceDataset()
    changed.input_key = "prompt"

    assert _data_path(baseline) != _data_path(changed)


@pytest.mark.parametrize(
    ("baseline", "changed"),
    [
        (
            HarborDataset(task_root="/tasks", split="train"),
            HarborDataset(task_root="/tasks", split="eval"),
        ),
        (
            HarborDataset(task_root="/tasks", train_size=10),
            HarborDataset(task_root="/tasks", train_size=20),
        ),
    ],
)
def test_harbor_materialization_path_tracks_row_selection(baseline, changed) -> None:
    assert _data_path(baseline) != _data_path(changed)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dataset_name", "example/tasks"),
        ("eval_repeats", 2),
        ("eval_size", 5),
        ("input_key", "prompt"),
        ("instruction_path", "prompt.md"),
        ("label_key", "target"),
        ("label_metadata_path", "metadata.json"),
        ("output_format", "jsonl"),
        ("path", "/alternate-tasks"),
        ("prompt_template", "Task: {instruction}"),
        ("shuffle_seed", 42),
        ("shuffle_tasks", True),
        ("system_prompt", "Solve the task."),
        ("task_files_dir", "staged-tasks"),
        ("task_glob", "task-*"),
        ("task_names", ["task-a"]),
        ("task_root", "/alternate-tasks"),
        ("candidate_path", "/tmp/alternate-candidate.py"),
        ("candidate_command", "python3 {candidate_path}"),
        ("train_repeats", 2),
    ],
)
def test_harbor_materialization_path_tracks_row_inputs(attribute, value) -> None:
    baseline = HarborDataset(task_root="/tasks")
    changed = HarborDataset(task_root="/tasks")
    setattr(changed, attribute, value)

    assert _data_path(baseline) != _data_path(changed)


@pytest.mark.parametrize(
    "datasets",
    [
        (
            ExampleHuggingFaceDataset(hf_split="validation", n_rows=10),
            ExampleHuggingFaceDataset(hf_split="validation", n_rows=10),
        ),
        (
            HarborDataset(
                task_root="/tasks",
                split="eval",
                train_size=10,
                eval_size=5,
            ),
            HarborDataset(
                task_root="/tasks",
                split="eval",
                train_size=10,
                eval_size=5,
            ),
        ),
    ],
)
def test_materialization_path_is_stable_for_identical_configs(datasets) -> None:
    first, second = datasets
    assert _data_path(first) == _data_path(second)


def test_dataset_class_identity_changes_materialization_path() -> None:
    assert _data_path(ExampleHuggingFaceDataset()) != _data_path(
        AlternateHuggingFaceDataset()
    )
    assert _data_path(HarborDataset(task_root="/tasks")) != _data_path(
        AlternateHarborDataset(task_root="/tasks")
    )


def test_non_materialization_controls_do_not_change_paths() -> None:
    baseline_hf = ExampleHuggingFaceDataset()
    changed_hf = ExampleHuggingFaceDataset()
    changed_hf.needs_chat_template = not baseline_hf.needs_chat_template
    changed_hf.needs_refresh = not baseline_hf.needs_refresh

    baseline_harbor = HarborDataset(task_root="/tasks")
    changed_harbor = HarborDataset(task_root="/tasks", needs_refresh=True)
    changed_harbor.needs_chat_template = not baseline_harbor.needs_chat_template

    assert _data_path(baseline_hf) == _data_path(changed_hf)
    assert _data_path(baseline_harbor) == _data_path(changed_harbor)
