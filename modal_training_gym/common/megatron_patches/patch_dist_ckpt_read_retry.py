"""Retry torch_dist checkpoint reads that fail with EINVAL/EIO on Modal Volumes.

Resuming a TB-scale torch_dist checkpoint off a Modal Volume intermittently
fails with ``OSError: [Errno 22] Invalid argument`` inside
``FileSystemReader.read_data`` on a subset of ranks, while the same files read
back byte-perfect from separate containers (verified with whole-file and
concurrent full-checkpoint re-reads). The data is intact; the read path inside
a loaded multi-node training container fails intermittently, and a failed read
costs a full multi-node retry cycle.

This patch wraps ``FileSystemReader.read_data`` with a retry: on ``OSError``
with ``EINVAL``/``EIO`` it backs off and re-runs the whole call, which reopens
every stream (the streams are opened inside ``read_data``). Re-running is
idempotent — ``load_bytes`` replaces the entry and ``copy_`` rewrites the same
target tensor with the same bytes. Any other errno, and exhaustion after 6
attempts, re-raise unchanged. Every retry logs loudly so a clean load and a
patched-over flaky load are distinguishable.

Patching the class method (rather than rewriting the body) keeps the patch
robust to upstream drift and covers subclasses that don't override
``read_data``.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

filesystem_py = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/torch/distributed/checkpoint/filesystem.py"
)

if not filesystem_py.exists():
    print(f"WARNING: {filesystem_py} not found — skipping read-retry patch")
    raise SystemExit(0)

src = filesystem_py.read_text()
marker = "PATCHED_READ_RETRY"

if marker in src:
    print("filesystem.py already patched for checkpoint read retry")
    raise SystemExit(0)

if "class FileSystemReader" not in src or "def read_data" not in src:
    print(
        "WARNING: could not patch filesystem.py for read retry — "
        "FileSystemReader.read_data not found"
    )
    raise SystemExit(0)

wrapper = f'''

# {marker}: torch_dist checkpoint reads through a Modal Volume mount
# intermittently fail with EINVAL/EIO on a subset of ranks while the underlying
# files are intact (verified by re-reading the full checkpoint from separate
# containers). A failed read costs a whole multi-node retry cycle, so reopen
# and retry instead. Re-running read_data is idempotent: it re-copies the same
# bytes into the same targets. See patch_dist_ckpt_read_retry.py.
def _mtg_wrap_read_data(_orig):
    import errno as _errno
    import functools as _functools
    import logging as _logging
    import time as _time

    _log = _logging.getLogger(__name__)
    _attempts = 6

    @_functools.wraps(_orig)
    def read_data(self, plan, planner):
        for _try in range(_attempts):
            try:
                result = _orig(self, plan, planner)
                if _try:
                    _log.warning(
                        "{marker}: checkpoint read succeeded on attempt %d/%d",
                        _try + 1,
                        _attempts,
                    )
                return result
            except OSError as exc:
                if exc.errno not in (_errno.EINVAL, _errno.EIO):
                    raise
                if _try == _attempts - 1:
                    _log.error(
                        "{marker}: checkpoint read still failing with errno %s "
                        "after %d attempts; the flaky-read assumption does not "
                        "hold here",
                        exc.errno,
                        _attempts,
                    )
                    raise
                _log.warning(
                    "{marker}: checkpoint read hit errno %s (%s); "
                    "reopening and retrying (attempt %d/%d)",
                    exc.errno,
                    exc,
                    _try + 1,
                    _attempts,
                )
                _time.sleep(5.0 * (_try + 1))

    return read_data


FileSystemReader.read_data = _mtg_wrap_read_data(FileSystemReader.read_data)
'''

filesystem_py.write_text(src + wrapper)
print("Patched filesystem.py: FileSystemReader.read_data retries EINVAL/EIO")
