"""Optionally rate-limit torch_dist checkpoint shard writes.

A TB-scale save writes its shards to the Modal Volume mount at local-disk
speed (~64 GiB per node in under a minute), far faster than the mount uploads,
and the resulting burst degrades the cluster network hard enough to reset
established TCP connections between containers — which kills the driver's Ray
RPCs and fails the run at its checkpoint. Reads of the same magnitude are
naturally paced (~1 GiB/s per node during engine weight load) and never
trigger this, so pacing the writes into the proven-safe envelope removes the
burst and keeps the mount's uncommitted buffer shallow.

``FileSystemWriterAsync.write_preloaded_data`` opens one stream per forked
writer process and pushes every shard item through it. This patch wraps that
stream in a limiter that slices large writes and sleeps to enforce
``MILES_CKPT_WRITE_BWLIMIT_MBPS`` (megabytes/second, **per writer process** —
Megatron runs 2 writers per rank, so a node's rate is
``16 x MILES_CKPT_WRITE_BWLIMIT_MBPS`` at 8 ranks/node). Unset, empty, or 0
leaves the stream untouched, so models that do not opt in are unaffected.
Checkpoint bytes are unchanged; only the write pacing is.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

filesystem_async_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)

if not filesystem_async_py.exists():
    print(f"WARNING: {filesystem_async_py} not found — skipping write-throttle patch")
    raise SystemExit(0)

src = filesystem_async_py.read_text()
marker = "PATCHED_WRITE_THROTTLE"

if marker in src:
    print("filesystem_async.py already patched for write throttling")
    raise SystemExit(0)

pattern = re.compile(
    r'^( +)with open_file\(file_name, "wb"\) as stream:\n',
    re.MULTILINE,
)
m = pattern.search(src)

if not m:
    print(
        "WARNING: could not patch filesystem_async.py for write throttling — "
        '`with open_file(file_name, "wb") as stream:` was not found'
    )
    raise SystemExit(0)

indent = m.group(1)
inner = indent + "    "
replacement = (
    f'{indent}with open_file(file_name, "wb") as stream:\n'
    f"{inner}# {marker}: pace shard writes so the Volume mount's upload keeps\n"
    f"{inner}# up; the unpaced burst resets cluster TCP connections. No-op\n"
    f"{inner}# unless MILES_CKPT_WRITE_BWLIMIT_MBPS is set. See\n"
    f"{inner}# patch_dist_ckpt_write_throttle.py.\n"
    f"{inner}stream = _mtg_throttled_stream(stream)\n"
)
src = pattern.sub(lambda _: replacement, src, count=1)

helper = f"""

# {marker}: helpers for the shard-write rate limit above.
class _MtgThrottledWriter:
    _CHUNK = 8 * 1024 * 1024

    def __init__(self, raw, rate_bytes_per_s):
        self._raw = raw
        self._rate = float(rate_bytes_per_s)
        self._start = None
        self._written = 0

    def write(self, data):
        import time as _time

        mv = memoryview(data)
        if mv.itemsize != 1:
            mv = mv.cast("B")
        if len(mv) == 0:
            return self._raw.write(mv)
        total = 0
        for offset in range(0, len(mv), self._CHUNK):
            chunk = mv[offset : offset + self._CHUNK]
            self._raw.write(chunk)
            now = _time.monotonic()
            if self._start is None:
                self._start = now
            total += len(chunk)
            self._written += len(chunk)
            ahead = self._written / self._rate - (now - self._start)
            if ahead > 0:
                _time.sleep(ahead)
        return total

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _mtg_throttled_stream(stream):
    import os as _os

    raw = _os.environ.get("MILES_CKPT_WRITE_BWLIMIT_MBPS", "")
    try:
        mbps = float(raw or 0)
    except ValueError:
        mbps = 0.0
    if mbps <= 0:
        return stream
    return _MtgThrottledWriter(stream, mbps * 1024 * 1024)
"""

filesystem_async_py.write_text(src + helper)
print("Patched filesystem_async.py: shard writes honor MILES_CKPT_WRITE_BWLIMIT_MBPS")
