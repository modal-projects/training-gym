from __future__ import annotations

import sys
import types

import pytest

from modal_training_gym.frameworks.slime.modal_helpers.patches.patch_volume_commit_on_save import (
    commit_after_save,
)


def test_commit_after_save_skips_empty_volume(monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_CHECKPOINTS_VOLUME", "")

    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            raise AssertionError("modal imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    assert commit_after_save(lambda: "ok")() == "ok"


def test_commit_after_save_barriers_when_commit_raises(monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_CHECKPOINTS_VOLUME", "ckpts")
    monkeypatch.setenv("LOCAL_RANK", "0")

    barriers: list[int] = []
    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.is_initialized = lambda: True
    dist_mod.barrier = lambda: barriers.append(1)
    dist_mod.get_rank = lambda: 0
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)

    class BoomVolume:
        @staticmethod
        def from_name(name):
            raise RuntimeError("commit failed")

    modal_mod = types.ModuleType("modal")
    modal_mod.Volume = BoomVolume
    monkeypatch.setitem(sys.modules, "modal", modal_mod)

    with pytest.raises(RuntimeError, match="commit failed"):
        commit_after_save(lambda: "ok")()

    assert barriers == [1, 1]
