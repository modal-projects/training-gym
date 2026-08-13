"""The cross-rank barrier that gates publishing a converted checkpoint's ``.metadata``.

A multi-node conversion has every rank move its own shards onto the Volume
independently. ``.metadata`` is what makes a directory look finished, so rank 0 may
only publish it once every peer's shards are committed — otherwise an interrupted
conversion leaves a directory that passes the completeness check with shards missing
and the next run trains on partial weights.
"""

import os

import pytest

from modal_training_gym.frameworks.miles import launcher


class FakeVolume:
    """Stands in for a ``modal.Volume``; the barrier only commits and reloads."""

    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


@pytest.fixture(autouse=True)
def _fast_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "_CONVERT_BARRIER_TIMEOUT_S", 0.3)
    monkeypatch.setattr(launcher, "_CONVERT_BARRIER_POLL_S", 0.01)


def test_await_returns_once_every_rank_has_signalled(tmp_path) -> None:
    volume = FakeVolume()
    save_path = str(tmp_path)
    for rank in range(4):
        launcher._signal_convert_rank_moved(volume, save_path, rank)

    launcher._await_convert_ranks_moved(volume, save_path, 4)

    assert volume.reloads >= 1


def test_await_times_out_while_a_peer_is_still_moving(tmp_path) -> None:
    volume = FakeVolume()
    save_path = str(tmp_path)
    for rank in (0, 1, 3):
        launcher._signal_convert_rank_moved(volume, save_path, rank)

    with pytest.raises(RuntimeError, match=r"rank2\.moved"):
        launcher._await_convert_ranks_moved(volume, save_path, 4)


def test_await_times_out_when_no_rank_has_signalled(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="refusing to publish .metadata"):
        launcher._await_convert_ranks_moved(FakeVolume(), str(tmp_path), 2)


def test_markers_live_outside_the_checkpoint_scan(tmp_path) -> None:
    """The barrier directory must not read as an ``iter_*``/``release`` checkpoint."""
    save_path = str(tmp_path)
    launcher._signal_convert_rank_moved(FakeVolume(), save_path, 0)

    name = os.path.basename(launcher._convert_barrier_dir(save_path))
    assert name.startswith(".")
    assert name != "release" and not name.startswith("iter_")
    assert not launcher._is_complete_torch_dist_checkpoint(
        launcher._convert_barrier_dir(save_path)
    )


def test_withholding_metadata_leaves_the_mount_incomplete(tmp_path) -> None:
    """The direct-to-Volume path has to move ``.metadata`` off the mount entirely.

    ``Volume.reload`` may implicitly commit whatever is still pending, so leaving the
    file in place but uncommitted would publish it during rank 0's wait.
    """
    save_path = tmp_path / "ckpt"
    iter_dir = save_path / "iter_0000001"
    iter_dir.mkdir(parents=True)
    for name in (".metadata", "common.pt", "__0_0.distcp"):
        (iter_dir / name).touch()
    stash = tmp_path / "stash"
    stash.mkdir()

    assert launcher._is_complete_torch_dist_checkpoint(str(iter_dir))

    withheld = launcher._withhold_metadata(str(save_path), str(stash))

    assert len(withheld) == 1
    assert not (iter_dir / ".metadata").exists()
    assert not launcher._is_complete_torch_dist_checkpoint(str(iter_dir))

    for stashed, original in withheld:
        os.makedirs(os.path.dirname(original), exist_ok=True)
        os.replace(stashed, original)

    assert launcher._is_complete_torch_dist_checkpoint(str(iter_dir))


def test_withholding_metadata_is_a_noop_without_one(tmp_path) -> None:
    iter_dir = tmp_path / "ckpt" / "iter_0000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "__1_0.distcp").touch()
    stash = tmp_path / "stash"
    stash.mkdir()

    assert launcher._withhold_metadata(str(tmp_path / "ckpt"), str(stash)) == []


def test_signalling_is_idempotent(tmp_path) -> None:
    volume = FakeVolume()
    save_path = str(tmp_path)
    launcher._signal_convert_rank_moved(volume, save_path, 0)
    launcher._signal_convert_rank_moved(volume, save_path, 0)

    launcher._await_convert_ranks_moved(volume, save_path, 1)
    assert os.listdir(launcher._convert_barrier_dir(save_path)) == ["rank0.moved"]
