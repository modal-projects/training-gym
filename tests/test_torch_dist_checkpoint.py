import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from modal_training_gym.common import torch_dist_checkpoint as ckpt
from modal_training_gym.common.launcher_utils import pin_resume_iteration
from modal_training_gym.common.run import torch_dist_resume_checkpoint
from tests.checkpoint_fixtures import metadata_bytes, write_checkpoint


def test_all_referenced_files_and_offsets_are_required(tmp_path):
    root = write_checkpoint(
        tmp_path / "iter_0000449",
        [
            ("__0_0.distcp", 0, 8),
            ("__1_0.distcp", 0, 16),
            ("__1_0.distcp", 16, 32),
        ],
    )
    assert ckpt.is_complete_torch_dist_checkpoint_dir(root)
    (root / "__1_0.distcp").write_bytes(b"x" * 47)
    assert not ckpt.is_complete_torch_dist_checkpoint_dir(root)
    (root / "__1_0.distcp").unlink()
    assert not ckpt.is_complete_torch_dist_checkpoint_dir(root)


def test_broad_layover_missing_shard_pattern(tmp_path):
    root = write_checkpoint(
        tmp_path,
        [(f"__{rank}_{part}.distcp", 0, 8) for rank in range(64) for part in range(2)],
    )
    for rank in range(40, 56):
        (root / f"__{rank}_1.distcp").unlink()
    assert not ckpt.is_complete_torch_dist_checkpoint_dir(root)
    assert not ckpt.is_complete_torch_dist_checkpoint([p.name for p in root.iterdir()])


@pytest.mark.parametrize(
    "entry",
    [
        ("../outside.distcp", 0, 1),
        ("/outside.distcp", 0, 1),
        ("shard.distcp", -1, 1),
        ("shard.distcp", 0, 0),
    ],
)
def test_invalid_storage_entries_are_rejected(entry):
    with pytest.raises(ValueError):
        ckpt.checkpoint_shard_sizes(metadata_bytes([entry]))


def test_metadata_does_not_execute_pickle_globals(tmp_path):
    class Payload:
        def __reduce__(self):
            return (eval, ("1+1",))

    with pytest.raises(pickle.UnpicklingError, match="Unsupported"):
        ckpt.checkpoint_shard_sizes(pickle.dumps(Payload()))
    root = write_checkpoint(tmp_path)
    (root / ".metadata").write_bytes(b"broken")
    assert not ckpt.is_complete_torch_dist_checkpoint_dir(root)


def test_resume_falls_back_when_tracker_checkpoint_is_truncated(tmp_path):
    write_checkpoint(tmp_path / "iter_0000399")
    bad = write_checkpoint(tmp_path / "iter_0000449")
    (bad / "shard.distcp").write_bytes(b"short")
    (tmp_path / ckpt.TORCH_DIST_TRACKER_NAME).write_text("449\n")
    result = torch_dist_resume_checkpoint(
        str(tmp_path), is_complete=ckpt.is_complete_torch_dist_checkpoint_dir
    )
    assert result["resume_from_iteration"] == 399
    (tmp_path / "iter_0000399" / "common.pt").unlink()
    assert (
        torch_dist_resume_checkpoint(
            str(tmp_path), is_complete=ckpt.is_complete_torch_dist_checkpoint_dir
        )
        is None
    )


@pytest.mark.parametrize("reference", [None, 17])
def test_resume_pins_actor_without_reusing_actor_step_for_reference(
    tmp_path, reference
):
    source = tmp_path / "config.yaml"
    data = {"ckpt_step": 449, "qkv_format": "thd"}
    if reference is not None:
        data["ref_ckpt_step"] = reference
    source.write_text(yaml.safe_dump(data))
    cfg = SimpleNamespace(extra_config=str(source))
    pin_resume_iteration(cfg, 399)
    actual = yaml.safe_load(Path(cfg.extra_config).read_text())
    assert actual == {
        **data,
        "ckpt_step": 399,
        "ref_ckpt_step": reference or 0,
        "exit_on_missing_checkpoint": True,
    }
    assert yaml.safe_load(source.read_text()) == data
    Path(cfg.extra_config).unlink()


def test_completion_marker_requires_durable_complete_view(tmp_path, monkeypatch):
    root = write_checkpoint(tmp_path / "iter_0000449")
    (tmp_path / ckpt.TORCH_DIST_TRACKER_NAME).write_text("449\n")
    calls = []
    volume = SimpleNamespace(
        reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
    )
    monkeypatch.setattr(ckpt.Volume, "from_name", lambda *a, **k: volume)
    (root / "shard.distcp").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        ckpt._validate_committed_checkpoint("volume", str(tmp_path))
    assert not (root / "training_gym_complete.json").exists()
    assert calls == ["reload"]
    (root / "shard.distcp").write_bytes(b"x" * 32)
    ckpt._validate_committed_checkpoint("volume", str(tmp_path))
    assert (root / "training_gym_complete.json").exists()
    assert calls == ["reload", "reload", "commit"]


@pytest.mark.parametrize("coordinator", [False, True])
def test_each_node_commits_before_coordinator(coordinator, monkeypatch):
    events = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(ckpt, "_commit_volume", lambda v: events.append("commit"))

    def gather(error):
        events.append("gather")
        return [error]

    wrapper = SimpleNamespace(is_coordinator=coordinator, all_gather_object=gather)
    ckpt.commit_checkpoint_volume_across_ranks(
        "volume", wrapper, lambda: events.append("barrier")
    )
    assert events == (
        ["gather", "barrier", "commit", "gather", "barrier"]
        if coordinator
        else ["commit", "gather", "barrier", "gather", "barrier"]
    )


def test_failed_shard_commit_stops_before_coordinator_publish(monkeypatch):
    wrapper = SimpleNamespace(
        is_coordinator=True, all_gather_object=lambda error: [None, "disk failed"]
    )
    monkeypatch.setattr(
        ckpt, "_commit_volume", lambda v: pytest.fail("must not publish")
    )
    with pytest.raises(RuntimeError, match="shard commit failed"):
        ckpt.commit_checkpoint_volume_across_ranks("volume", wrapper, lambda: None)


def test_save_coordination_uses_gloo_and_cleans_up(monkeypatch):
    events = []
    group = object()

    def new_group(*, backend):
        assert backend == "gloo"
        events.append("gloo")
        return group

    def gather(output, value, **kwargs):
        assert kwargs["group"] is group
        output[:] = [value]

    def barrier(**kwargs):
        assert kwargs["group"] is group

    dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        new_group=new_group,
        all_gather_object=gather,
        barrier=barrier,
        destroy_process_group=lambda g: events.append("destroy"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TRAINING_GYM_CHECKPOINTS_VOLUME_NAME", "volume")
    monkeypatch.setattr(ckpt, "_commit_volume", lambda v: events.append("commit"))
    monkeypatch.setattr(
        ckpt, "_validate_committed_checkpoint", lambda *a: events.append("validate")
    )
    ckpt.commit_latest_checkpoint_volume("/checkpoints/run")
    assert events == ["gloo", "commit", "validate", "destroy"]


def test_save_patch_upgrades_old_wrappers_and_is_idempotent(tmp_path, monkeypatch):
    from modal_training_gym.common.megatron_patches import (
        patch_checkpoint_commit as patcher,
    )

    saving = tmp_path / "checkpointing.py"
    finalize = tmp_path / "async_utils.py"
    saving.write_text("def save_checkpoint(*args, **kwargs):\n    return 7\n")
    finalize.write_text(
        "def maybe_finalize_async_save(*args, **kwargs):\n    return 8\n"
    )
    monkeypatch.setattr(patcher, "_CHECKPOINTING", saving)
    monkeypatch.setattr(patcher, "_ASYNC_UTILS", finalize)
    patcher.main()
    for path in (saving, finalize):
        path.write_text(
            path.read_text().replace(
                "commit_latest_checkpoint_volume(get_args().save)",
                "commit_latest_checkpoint_volume()",
            )
        )
    finalize.write_text(
        finalize.read_text().replace("    from megatron.training import get_args\n", "")
    )
    patcher.main()
    first = [path.read_text() for path in (saving, finalize)]
    patcher.main()
    assert first == [path.read_text() for path in (saving, finalize)]
    for source in first:
        compile(source, "patched", "exec")
        assert "commit_latest_checkpoint_volume(get_args().save)" in source
        assert "from megatron.training import get_args" in source
