"""Bounded retries for transient EINVAL reads in torch checkpoint views."""

from __future__ import annotations

import importlib.util
import pathlib

try:
    spec = importlib.util.find_spec("torch.distributed.checkpoint.utils")
except ModuleNotFoundError:
    spec = None
utils_py = pathlib.Path(spec.origin) if spec and spec.origin else None
if utils_py is None or not utils_py.exists():
    print("WARNING: torch checkpoint utils not found — skipping reader retry patch")
    raise SystemExit(0)

src = utils_py.read_text()
marker = "TRAINING_GYM_CHECKPOINT_READER_RETRY"
if marker in src:
    print("Checkpoint reader retry patch already applied")
    raise SystemExit(0)

readinto_old = (
    "\n".join(
        [
            "    def readinto(self, b):",
            "        max_size = self.len - self.tell()",
            "        if max_size == 0:",
            "            return 0",
            "        if len(b) > max_size:",
            "            b = memoryview(b)[:max_size]",
            "        return self.base_stream.readinto(b)  # type: ignore[attr-defined]",
        ]
    )
    + "\n"
)
read_old = (
    "\n".join(
        [
            "    def read(self, size=-1):",
            "        max_size = self.len - self.tell()",
            "        if size == -1 or size > max_size:",
            "            size = max_size",
            "        return self.base_stream.read(size)",
        ]
    )
    + "\n"
)

readinto_new = (
    "\n".join(
        [
            "    def readinto(self, b):",
            "        max_size = self.len - self.tell()",
            "        if max_size == 0:",
            "            return 0",
            "        if len(b) > max_size:",
            "            b = memoryview(b)[:max_size]",
            "        start = self.base_stream.tell()",
            "        for attempt in range(5):",
            "            try:",
            "                return self.base_stream.readinto(b)  # type: ignore[attr-defined]",
            "            except OSError as exc:",
            '                if getattr(exc, "errno", None) != errno.EINVAL or attempt == 4:',
            "                    raise",
            "                _tg_checkpoint_read_backoff(self.base_stream, start, attempt)",
            '        raise AssertionError("unreachable")',
        ]
    )
    + "\n"
)
read_new = (
    "\n".join(
        [
            "    def read(self, size=-1):",
            "        max_size = self.len - self.tell()",
            "        if size == -1 or size > max_size:",
            "            size = max_size",
            "        start = self.base_stream.tell()",
            "        for attempt in range(5):",
            "            try:",
            "                return self.base_stream.read(size)",
            "            except OSError as exc:",
            '                if getattr(exc, "errno", None) != errno.EINVAL or attempt == 4:',
            "                    raise",
            "                _tg_checkpoint_read_backoff(self.base_stream, start, attempt)",
            '        raise AssertionError("unreachable")',
        ]
    )
    + "\n"
)

if readinto_old not in src or read_old not in src:
    print("WARNING: checkpoint reader signatures changed — skipping reader retry patch")
    raise SystemExit(0)

anchor = "import cProfile\n"
if anchor not in src:
    print(
        "WARNING: checkpoint reader import anchor not found — skipping reader retry patch"
    )
    raise SystemExit(0)
src = src.replace(anchor, anchor + "import errno\nimport time\n", 1)

helper = "\n".join(
    [
        "",
        "",
        "def _tg_checkpoint_read_backoff(stream, start, attempt):",
        "    try:",
        "        stream.seek(start)",
        "    except OSError:",
        "        pass",
        "    try:",
        "        rank = dist.get_rank()",
        "    except Exception:",
        "        rank = 0",
        "    time.sleep(0.05 * (2**attempt) + 0.002 * ((rank % 32) + 1))",
        "",
        "",
    ]
)
reader_anchor = "class _ReaderView(io.IOBase):"
if reader_anchor not in src:
    print("WARNING: _ReaderView anchor not found — skipping reader retry patch")
    raise SystemExit(0)
src = src.replace(reader_anchor, helper + reader_anchor, 1)
src = src.replace(readinto_old, readinto_new, 1).replace(read_old, read_new, 1)
src = src.replace(reader_anchor, reader_anchor + f"  # {marker}", 1)
utils_py.write_text(src)
print(f"Patched {utils_py} with bounded EINVAL checkpoint-read retries")
