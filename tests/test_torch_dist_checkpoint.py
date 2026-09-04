from __future__ import annotations

from pathlib import Path

import pytest

from modal_training_gym.common import torch_dist_checkpoint as checkpoint


class _DistWrapper:
    def __init__(
        self,
        events: list[str],
        *,
        is_coordinator: bool,
        gathered: list[list[str | None]] | None = None,
    ) -> None:
        self.events = events
        self.is_coordinator = is_coordinator
        self.gathered = list(gathered or [])

    def all_gather_object(self, error: str | None) -> list[str | None]:
        self.events.append(f"gather:{error}")
        return self.gathered.pop(0) if self.gathered else [error]


class _Volume:
    def __init__(self, events: list[str], *, fail_commit: bool = False) -> None:
        self.events = events
        self.fail_commit = fail_commit

    def commit(self) -> None:
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("commit")


def _patch_volume(monkeypatch: pytest.MonkeyPatch, volume: _Volume) -> None:
    def from_name(name: str, *, create_if_missing: bool) -> _Volume:
        assert name == "checkpoints"
        assert not create_if_missing
        return volume

    monkeypatch.setattr(checkpoint.Volume, "from_name", from_name)


def _write_complete_checkpoint(path: Path) -> None:
    path.mkdir()
    for name in (".metadata", "common.pt", "shard.distcp"):
        (path / name).touch()


def test_checkpoint_completeness_requires_full_layout(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "iter_0000001"

    assert not checkpoint.is_complete_torch_dist_checkpoint(
        {"common.pt", "shard.distcp"}
    )
    assert not checkpoint.is_complete_torch_dist_checkpoint_dir(checkpoint_dir)

    _write_complete_checkpoint(checkpoint_dir)

    assert checkpoint.is_complete_torch_dist_checkpoint(
        {".metadata", "common.pt", "shard.distcp"}
    )
    assert checkpoint.is_complete_torch_dist_checkpoint_dir(checkpoint_dir)


def test_checkpoint_completeness_treats_unreadable_directory_as_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_permission_error(_path: Path):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "iterdir", raise_permission_error)

    assert not checkpoint.is_complete_torch_dist_checkpoint_dir("/checkpoints/run")


def test_parse_tracker_and_iteration() -> None:
    assert checkpoint.parse_torch_dist_tracker("1\n") == 1
    assert checkpoint.parse_torch_dist_tracker("release") is None
    assert checkpoint.parse_torch_dist_iteration("iter_0000007") == 7
    assert checkpoint.parse_torch_dist_iteration("iter_0000007_hf") is None


def test_commit_across_ranks_commits_noncoordinator_leaders_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    _patch_volume(monkeypatch, _Volume(events))

    checkpoint.commit_checkpoint_volume_across_ranks(
        "checkpoints",
        _DistWrapper(events, is_coordinator=False),
        lambda: events.append("barrier"),
    )

    assert events == ["commit", "gather:None", "barrier", "gather:None", "barrier"]


def test_commit_across_ranks_commits_coordinator_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    _patch_volume(monkeypatch, _Volume(events))

    checkpoint.commit_checkpoint_volume_across_ranks(
        "checkpoints",
        _DistWrapper(events, is_coordinator=True),
        lambda: events.append("barrier"),
    )

    assert events == ["gather:None", "barrier", "commit", "gather:None", "barrier"]


def test_commit_across_ranks_skips_non_leaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("LOCAL_RANK", "1")

    checkpoint.commit_checkpoint_volume_across_ranks(
        "checkpoints",
        _DistWrapper(events, is_coordinator=False),
        lambda: events.append("barrier"),
    )

    assert events == ["gather:None", "barrier", "gather:None", "barrier"]


def test_commit_across_ranks_noop_without_volume_name() -> None:
    events: list[str] = []

    checkpoint.commit_checkpoint_volume_across_ranks(
        None,
        _DistWrapper(events, is_coordinator=True),
        lambda: events.append("barrier"),
    )
    checkpoint.commit_checkpoint_volume_across_ranks(
        "",
        _DistWrapper(events, is_coordinator=True),
        lambda: events.append("barrier"),
    )

    assert events == []


def test_commit_across_ranks_stops_after_shard_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    _patch_volume(monkeypatch, _Volume(events, fail_commit=True))

    with pytest.raises(RuntimeError, match="shard commit"):
        checkpoint.commit_checkpoint_volume_across_ranks(
            "checkpoints",
            _DistWrapper(events, is_coordinator=False),
            lambda: events.append("barrier"),
        )

    assert events == ["commit", "gather:RuntimeError: commit"]


def test_commit_across_ranks_propagates_peer_failure() -> None:
    events: list[str] = []
    dist_wrapper = _DistWrapper(
        events,
        is_coordinator=True,
        gathered=[["RuntimeError: peer commit"]],
    )

    with pytest.raises(RuntimeError, match="shard commit"):
        checkpoint.commit_checkpoint_volume_across_ranks(
            "checkpoints",
            dist_wrapper,
            lambda: events.append("barrier"),
        )

    assert events == ["gather:None"]
