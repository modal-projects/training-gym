"""A torch_dist save that died mid-write must not be treated as resumable."""

from pathlib import Path

from modal_training_gym.common.run import torch_dist_resume_checkpoint
from modal_training_gym.frameworks.slime.launcher import (
    _is_complete_torch_dist_checkpoint,
)


def _write(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_finished_save_is_complete(tmp_path: Path) -> None:
    _write(tmp_path, ".metadata", "metadata.json", "common.pt", "__0_0.distcp")
    assert _is_complete_torch_dist_checkpoint(str(tmp_path))


def test_save_that_died_before_metadata_is_incomplete(tmp_path: Path) -> None:
    _write(tmp_path, "common.pt", "__0_0.distcp", "__1_0.distcp")
    assert not _is_complete_torch_dist_checkpoint(str(tmp_path))


def test_missing_directory_is_incomplete(tmp_path: Path) -> None:
    assert not _is_complete_torch_dist_checkpoint(str(tmp_path / "absent"))


def test_partial_iter_dir_is_not_offered_for_resume(tmp_path: Path) -> None:
    _write(tmp_path / "iter_0000001", "common.pt", "__0_0.distcp")
    assert (
        torch_dist_resume_checkpoint(
            str(tmp_path), is_complete=_is_complete_torch_dist_checkpoint
        )
        is None
    )


def test_finished_iter_dir_is_offered_for_resume(tmp_path: Path) -> None:
    _write(tmp_path / "iter_0000001", ".metadata", "common.pt", "__0_0.distcp")
    resume = torch_dist_resume_checkpoint(
        str(tmp_path), is_complete=_is_complete_torch_dist_checkpoint
    )
    assert resume is not None
    assert resume["resume_from_iteration"] == 1
