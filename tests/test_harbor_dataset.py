from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from modal_training_gym.common.dataset import HarborDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.harbor import resolve_harbor_task_path


def _write_task(task_dir: Path, *, marker: bytes) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Write the candidate.", encoding="utf-8")
    (task_dir / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n",
        encoding="utf-8",
    )
    (task_dir / "environment" / "fixtures").mkdir()
    (task_dir / "environment" / "fixtures" / "payload.bin").write_bytes(marker)
    (task_dir / "tests").mkdir()
    test_script = task_dir / "tests" / "test.sh"
    test_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    test_script.chmod(0o755)
    (task_dir / "solution").mkdir()
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )


def test_prepare_stages_complete_task_trees_with_portable_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = tmp_path / "source"
    _write_task(task_root / "task-a", marker=b"\x00task-a\xff")
    _write_task(task_root / "task-b", marker=b"\x00task-b\xff")
    output_path = tmp_path / "prepared" / "train.parquet"
    written_rows: dict[str, list[dict[str, object]]] = {}

    dataset = HarborDataset(path=str(task_root))
    output_path.parent.mkdir(parents=True)
    output_path.touch()
    assert not dataset.is_prepared(str(output_path))
    monkeypatch.setattr(
        dataset,
        "_write_split",
        lambda rows, path: written_rows.setdefault(path, rows),
    )

    dataset.prepare(str(output_path))

    assert dataset.is_prepared(str(output_path))
    rows = written_rows[str(output_path)]
    assert len(rows) == 2
    labels = {
        json.loads(row["label"])["harbor_task_name"]: json.loads(row["label"])
        for row in rows
    }
    assert (
        labels["task-a"]["harbor_task_data_rel"]
        == "prepared/harbor_tasks/source/task-a"
    )
    assert (
        labels["task-b"]["harbor_task_data_rel"]
        == "prepared/harbor_tasks/source/task-b"
    )

    staged_a = resolve_harbor_task_path(
        labels["task-a"],
        data_root=tmp_path,
    )
    staged_b = resolve_harbor_task_path(
        labels["task-b"],
        data_root=tmp_path,
    )
    assert (staged_a / "environment" / "fixtures" / "payload.bin").read_bytes() == (
        b"\x00task-a\xff"
    )
    assert (staged_b / "environment" / "fixtures" / "payload.bin").read_bytes() == (
        b"\x00task-b\xff"
    )
    assert not (staged_a / "environment" / "fixtures" / "payload.bin").samefile(
        staged_b / "environment" / "fixtures" / "payload.bin"
    )
    mode = (staged_a / "tests" / "test.sh").stat().st_mode
    assert mode & stat.S_IXUSR
    assert (staged_a / "solution" / "solve.sh").is_file()


def test_prepare_only_stages_selected_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_root = tmp_path / "source"
    _write_task(task_root / "task-a", marker=b"a")
    _write_task(task_root / "task-b", marker=b"b")
    output_path = tmp_path / "prepared" / "train.parquet"
    dataset = HarborDataset(path=str(task_root), task_names=["task-b"])
    monkeypatch.setattr(dataset, "_write_split", lambda rows, path: None)

    dataset.prepare(str(output_path))

    staged_root = output_path.parent / "harbor_tasks" / "source"
    assert not (staged_root / "task-a").exists()
    assert (staged_root / "task-b").is_dir()


def test_prepare_reports_missing_task_source(tmp_path: Path) -> None:
    dataset = HarborDataset(path=str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="task root does not exist"):
        dataset.prepare(str(tmp_path / "prepared" / "train.parquet"))


def test_resolve_harbor_task_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(
        TrainingGymConfigError,
        match="Harbor task path escapes data root",
    ):
        resolve_harbor_task_path(
            {"harbor_task_data_rel": "../outside"},
            data_root=tmp_path,
        )
