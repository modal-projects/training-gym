"""Tests for the torch_dist checkpoint-writer fork-retry patch.

The patch itself edits a file inside the miles/slime image, so these tests run it
against a faithful copy of the upstream snippet in a tmp dir, then *execute* the
patched loop against a fake ``Process`` to prove the retry actually works. That
matters more than a source diff: the bug it fixes is intermittent
(``BlockingIOError: [Errno 11]`` from ``os.fork()`` on 2 of 3 checkpoint saves),
so a live run passing once would not have demonstrated anything.
"""

from __future__ import annotations

import errno
import runpy
from pathlib import Path

import pytest

PATCH = (
    Path(__file__).parent.parent
    / "modal_training_gym"
    / "common"
    / "megatron_patches"
    / "patch_dist_ckpt_fork_retry.py"
)

# The shape of the upstream loop the patch targets, at its real indentation
# (inside `def write_preloaded_data_multiproc` -> `if not isinstance(...)`).
UPSTREAM = """\
import logging


class FileSystemWriterAsync:
    @staticmethod
    def write_preloaded_data_multiproc(p_list, results):
        logger = logging.getLogger(__name__)
        write_results_or_exc = dict()
        if not isinstance(write_results_or_exc, Exception):
            for p in p_list:
                p.start()

            results.append("collected")
"""


def _run_patch(target: Path, monkeypatch) -> str:
    """Execute the patch script with its hardcoded target redirected to ``target``."""
    src = PATCH.read_text()
    real = (
        '"/root/Megatron-LM/megatron/core/dist_checkpointing/strategies'
        '/filesystem_async.py"'
    )
    src = src.replace(real, repr(str(target)))
    script = target.parent / "_patch_under_test.py"
    script.write_text(src)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit:
        pass
    return target.read_text() if target.exists() else ""


class _FlakyProcess:
    """Raises EAGAIN on the first ``fail_times`` starts, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.attempts = 0
        self.started = False

    def start(self) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        self.started = True


def _exec_patched(patched_src: str, p_list, sleeps: list[float]):
    """Run the patched loop, stubbing sleep so the test doesn't actually wait."""
    ns: dict = {}
    # The patched source is the thing under test, so it must actually run.
    exec(compile(patched_src, "patched", "exec"), ns)  # noqa: S102
    import time as real_time

    orig_sleep = real_time.sleep
    real_time.sleep = sleeps.append
    try:
        results: list[str] = []
        ns["FileSystemWriterAsync"].write_preloaded_data_multiproc(p_list, results)
        return results
    finally:
        real_time.sleep = orig_sleep


@pytest.fixture
def patched(tmp_path, monkeypatch) -> str:
    target = tmp_path / "filesystem_async.py"
    target.write_text(UPSTREAM)
    out = _run_patch(target, monkeypatch)
    assert "PATCHED_FORK_RETRY" in out
    return out


def test_patch_applies_and_compiles(patched):
    compile(patched, "patched", "exec")
    # The original bare `p.start()` must no longer be the only call site.
    assert "for _mtg_try in range(_mtg_attempts):" in patched


def test_patch_is_idempotent(tmp_path, monkeypatch, capsys):
    target = tmp_path / "filesystem_async.py"
    target.write_text(UPSTREAM)
    first = _run_patch(target, monkeypatch)
    second = _run_patch(target, monkeypatch)
    assert first == second, "re-applying the patch must not change the file again"
    # The marker appears in the comment and in the log messages; what must not
    # duplicate is the injected retry block itself.
    assert second.count("_mtg_attempts = 6") == 1


def test_missing_target_is_skipped(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "nope.py"
    _run_patch(missing, monkeypatch)
    assert not missing.exists()
    assert "skipping fork-retry patch" in capsys.readouterr().out


def test_unrecognised_source_warns_and_leaves_file_alone(tmp_path, monkeypatch, capsys):
    """Upstream drift must fail loudly, not silently produce a broken file."""
    target = tmp_path / "filesystem_async.py"
    other = "def unrelated():\n    return 1\n"
    target.write_text(other)
    out = _run_patch(target, monkeypatch)
    assert out == other
    assert "could not patch" in capsys.readouterr().out


def test_transient_eagain_is_retried_and_succeeds(patched):
    """The actual bug: fork raises EAGAIN, then works on a later attempt."""
    procs = [_FlakyProcess(fail_times=2), _FlakyProcess(fail_times=1)]
    sleeps: list[float] = []
    results = _exec_patched(patched, procs, sleeps)

    assert all(p.started for p in procs)
    assert [p.attempts for p in procs] == [3, 2]
    assert results == ["collected"], "the loop must carry on to the rest of the body"
    # Exponential backoff: 0.5, 1.0 for the first proc; 0.5 for the second.
    assert sleeps == [0.5, 1.0, 0.5]


def test_persistent_eagain_still_raises(patched):
    """A genuinely exhausted resource must not be retried forever."""
    procs = [_FlakyProcess(fail_times=99)]
    sleeps: list[float] = []
    with pytest.raises(BlockingIOError):
        _exec_patched(patched, procs, sleeps)
    assert procs[0].attempts == 6, "bounded at 6 attempts"


def test_non_eagain_oserror_is_not_retried(patched):
    """An unrelated OSError must surface immediately, undisguised."""

    class _Broken:
        attempts = 0

        def start(self):
            type(self).attempts += 1
            raise OSError(errno.EPERM, "nope")

    sleeps: list[float] = []
    with pytest.raises(OSError) as exc:
        _exec_patched(patched, [_Broken()], sleeps)
    assert exc.value.errno == errno.EPERM
    assert _Broken.attempts == 1
    assert sleeps == []
