"""Retry ``os.fork()`` on EAGAIN in Megatron's torch_dist checkpoint writer.

``FileSystemWriterAsync.write_preloaded_data_multiproc`` (``filesystem_async.py``)
forks ``thread_count`` helper processes **per rank** to write the ``.distcp``
shards — 2 per rank by default, so 16 on an 8-rank node. On Modal that fork
intermittently fails::

    File ".../strategies/filesystem_async.py", line 315, in write_preloaded_data_multiproc
        p.start()
    File ".../multiprocessing/popen_fork.py", line 66, in _launch
        self.pid = os.fork()
    BlockingIOError: [Errno 11] Resource temporarily unavailable

Observed on Nemotron-3-Ultra's 4-layer slice: **2 of 3 checkpoint saves died this
way**, each taking the whole training run with them, while the run that survived
saved in 34 s. Measured on the training container, every limit that would
normally explain EAGAIN is ruled out: ``RLIMIT_NPROC`` unlimited, cgroup
``pids.max`` unlimited, ~400 processes / ~8 200 threads against
``kernel.pid_max`` 65 536, and cgroup memory 88 GiB of a 2 048 GiB limit. Forking
from a process holding 50 GB of RSS was also reproduced as *harmless* in
isolation, so it is not the CPU-offloaded optimizer's footprint either.

What is left is a transient, environment-level inability to create a task — and
EAGAIN is by definition "temporarily unavailable". POSIX's prescribed response to
EAGAIN on ``fork()`` is to retry, which is exactly what this patch does: up to 6
attempts with exponential backoff, re-raising anything that is not EAGAIN and
re-raising EAGAIN once the attempts are exhausted. Checkpoint contents,
sharding, and the write plan are untouched, so a save that would have succeeded
behaves identically.

Why not a flag: Megatron builds the save strategy as
``get_default_save_sharded_strategy(args.ckpt_format)`` (``checkpointing.py``)
with no ``thread_count`` argument, and exposes no CLI option for it, so the fork
count is not reachable from a recipe. Why not ``--async-save
--use-persistent-ckpt-worker`` (whose ``PersistentAsyncCaller`` uses *spawn*):
miles calls ``save_checkpoint`` and expects it to have completed on return, and
never finalizes the returned async request — enabling it risks silently
incomplete checkpoints, which is worse than a loud retry.

Retrying ``Process.start()`` is safe: CPython assigns ``self._popen`` only after
``self._Popen(self)`` returns, and ``_target``/``_args`` are deleted only after
that, so a ``start()`` that raised leaves the object reusable. ``start()`` also
calls ``_cleanup()``, which reaps exited children and may itself free whatever
the fork was short of.

Applied to every miles image (see ``frameworks/miles/launcher.py``), not just one
recipe: any miles model saving a torch_dist checkpoint forks the same writer.
Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

filesystem_async_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)

if not filesystem_async_py.exists():
    print(f"WARNING: {filesystem_async_py} not found — skipping fork-retry patch")
    raise SystemExit(0)

src = filesystem_async_py.read_text()
marker = "PATCHED_FORK_RETRY"

if marker in src:
    print("filesystem_async.py already patched for fork retry")
    raise SystemExit(0)

# Target the writer's start loop:
#     for p in p_list:
#         p.start()
# Indentation is read from the source so the replacement lines up regardless of
# how deeply nested upstream has it.
pattern = re.compile(r"^( +)for p in p_list:\n( +)p\.start\(\)\n", re.MULTILINE)
m = pattern.search(src)

if not m:
    print(
        "WARNING: could not patch filesystem_async.py for fork retry — "
        "`for p in p_list: p.start()` not found"
    )
    raise SystemExit(0)

outer, inner = m.group(1), m.group(2)
body = inner + "    "
replacement = (
    f"{outer}for p in p_list:\n"
    f"{inner}# {marker}: fork() intermittently returns EAGAIN on Modal even with\n"
    f"{inner}# NPROC/pids/memory limits all far from exhaustion. EAGAIN means\n"
    f"{inner}# 'try again', and a failed start() leaves the Process reusable, so\n"
    f"{inner}# retry with backoff instead of losing the whole training run.\n"
    f"{inner}import errno as _mtg_errno\n"
    f"{inner}import time as _mtg_time\n"
    "\n"
    f"{inner}_mtg_attempts = 6\n"
    f"{inner}for _mtg_try in range(_mtg_attempts):\n"
    f"{body}try:\n"
    f"{body}    p.start()\n"
    f"{body}    break\n"
    f"{body}except OSError as _mtg_exc:\n"
    f"{body}    if _mtg_exc.errno != _mtg_errno.EAGAIN:\n"
    f"{body}        raise\n"
    f"{body}    if _mtg_try == _mtg_attempts - 1:\n"
    f"{body}        raise\n"
    f"{body}    _mtg_time.sleep(0.5 * (2**_mtg_try))\n"
)

src = pattern.sub(lambda _: replacement, src, count=1)
filesystem_async_py.write_text(src)
print("Patched filesystem_async.py to retry fork() on EAGAIN")
