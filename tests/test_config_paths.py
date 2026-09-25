from __future__ import annotations

from pathlib import Path

import pytest

from modal_training_dojo.common import config as config_module


@pytest.fixture
def config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    new = tmp_path / ".training-dojo.toml"
    legacy = tmp_path / ".training-gym.toml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", new)
    monkeypatch.setattr(config_module, "LEGACY_CONFIG_PATH", legacy)
    return new, legacy


def test_load_config_prefers_new_path(config_paths):
    new, legacy = config_paths
    new.write_text('name = "dojo"\n')
    legacy.write_text('name = "gym"\n')

    assert config_module.load_config() == {"name": "dojo"}


def test_load_config_falls_back_to_legacy_path(config_paths):
    _, legacy = config_paths
    legacy.write_text('name = "gym"\n')

    assert config_module.load_config() == {"name": "gym"}


def test_load_config_missing_both(config_paths):
    assert config_module.load_config() == {}
