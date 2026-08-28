"""_is_resumable_checkpoint must reject saves whose committed .metadata
references shards that never landed on the volume (a node killed mid-commit
leaves exactly that shape), while keeping the existing acceptances."""

import pickle

import pytest

from modal_training_gym.frameworks.miles.launcher import _is_resumable_checkpoint


def _write_metadata(path, shard_names):
    # The predicate scans the raw pickle bytes for shard names, so a plain
    # pickled structure carrying the same literal strings stands in for torch
    # DCP's Metadata object.
    (path / ".metadata").write_bytes(
        pickle.dumps({"storage_data": {name: name for name in shard_names}})
    )


def _write_shards(path, shard_names, size=8):
    for name in shard_names:
        (path / name).write_bytes(b"x" * size)


SHARDS = [f"__{r}_{w}.distcp" for r in range(4) for w in (0, 1)]


def test_complete_torch_dist_save_is_resumable(tmp_path):
    _write_metadata(tmp_path, SHARDS)
    _write_shards(tmp_path, SHARDS)
    (tmp_path / "common.pt").write_bytes(b"c")
    assert _is_resumable_checkpoint(str(tmp_path))


def test_missing_referenced_shard_is_rejected(tmp_path):
    _write_metadata(tmp_path, SHARDS)
    _write_shards(tmp_path, SHARDS[:-1])
    assert not _is_resumable_checkpoint(str(tmp_path))


def test_empty_referenced_shard_is_rejected(tmp_path):
    _write_metadata(tmp_path, SHARDS)
    _write_shards(tmp_path, SHARDS[:-1])
    (tmp_path / SHARDS[-1]).write_bytes(b"")
    assert not _is_resumable_checkpoint(str(tmp_path))


def test_shards_without_metadata_are_rejected(tmp_path):
    _write_shards(tmp_path, SHARDS)
    assert not _is_resumable_checkpoint(str(tmp_path))


def test_extra_unreferenced_shards_are_fine(tmp_path):
    _write_metadata(tmp_path, SHARDS)
    _write_shards(tmp_path, SHARDS + ["__9_0.distcp"])
    assert _is_resumable_checkpoint(str(tmp_path))


def test_lora_adapter_save_is_resumable(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "rank0.pt").write_bytes(b"a")
    assert _is_resumable_checkpoint(str(tmp_path))


def test_unrecognized_metadata_keeps_old_behavior(tmp_path):
    _write_shards(tmp_path, SHARDS)
    (tmp_path / ".metadata").write_bytes(b"no shard names in here")
    assert _is_resumable_checkpoint(str(tmp_path))


@pytest.mark.parametrize("contents", [None, []])
def test_missing_or_empty_dir_is_not_resumable(tmp_path, contents):
    target = tmp_path / "iter_0000001"
    if contents is not None:
        target.mkdir()
    assert not _is_resumable_checkpoint(str(target))
