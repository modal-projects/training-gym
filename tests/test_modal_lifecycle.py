import pytest

from modal_training_gym.common import modal_lifecycle


def test_stop_app_or_raise_propagates_modal_errors(monkeypatch):
    def fail_stop(_app_id):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(modal_lifecycle, "_stop_app", fail_stop)

    with pytest.raises(RuntimeError, match="stop failed"):
        modal_lifecycle.stop_app_or_raise("ap-1")


def test_stop_app_remains_best_effort(monkeypatch, capsys):
    def fail_stop(_app_id):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(modal_lifecycle, "_stop_app", fail_stop)

    modal_lifecycle.stop_app("ap-1")

    assert "WARNING: could not auto-stop app ap-1" in capsys.readouterr().out
