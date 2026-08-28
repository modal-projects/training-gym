"""Stage checkpoint shards to pageable host memory, not pinned, before the fork.

``FileSystemWriterAsync`` saves a torch_dist checkpoint in two steps
(``filesystem_async.py``)::

    partial(self.preload_tensors, self.write_buckets, True)   # non_blocking=True
    ...
    ctx = mp.get_context("fork")                              # then fork writers

``preload_tensors(..., non_blocking=True)`` copies every shard D2H with
``tensor.to("cpu", non_blocking=True)``, which stages them in **pinned** host
memory — upstream's own docstring says so: *"The tensors will be stored in
pinned memory if non_blocking=True."* The writer then forks one helper process
per bucket while the parent still holds all of it.

Upstream already treats that ordering as unsafe, but guards only one platform::

    if non_blocking and getattr(torch.version, "hip", None):
        # Currently on the ROCm platform, forking a subprocess afterward
        # with pinned_memory=True will trigger segmentation fault
        non_blocking = False

The hazard is fork-after-pinned, not the GPU vendor. At TB scale the staging
step pins ~320 GiB of host memory per node — unswappable, unreclaimable, and
invisible to RSS-based monitors — while the writers fork.

This patch extends upstream's own mitigation to every platform: force
``non_blocking=False`` in ``preload_tensors``, so shards land in ordinary
pageable memory that is swappable, is not driver-pinned, and is safe to hold
across a fork.

Cost: the D2H copy is synchronous rather than overlapped, so the staging step is
somewhat slower. Checkpoint contents are byte-identical — this changes where the
staged copy lives, not what is written. A save that already succeeded produces
the same checkpoint.

Not reachable from a recipe: the ``True`` is a positional literal inside
``FileSystemWriterAsync``, with no argument, flag or environment variable
exposed.

Applied to every miles image (see ``frameworks/miles/launcher.py``): any miles
model writing a torch_dist checkpoint stages through this path, and the risk
scales with checkpoint size.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

filesystem_async_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)

if not filesystem_async_py.exists():
    print(f"WARNING: {filesystem_async_py} not found — skipping blocking-D2H patch")
    raise SystemExit(0)

src = filesystem_async_py.read_text()
marker = "PATCHED_BLOCKING_D2H"

if marker in src:
    print("filesystem_async.py already patched for blocking D2H staging")
    raise SystemExit(0)

# Target upstream's ROCm-only guard and widen it to every platform. Matching the
# condition (rather than the `True` at the call site) keeps the change in one
# place and leaves the call site's contract untouched.
pattern = re.compile(
    r"^( +)if non_blocking and getattr\(torch\.version, \"hip\", None\):\n",
    re.MULTILINE,
)
m = pattern.search(src)

if not m:
    print(
        "WARNING: could not patch filesystem_async.py for blocking D2H — "
        'the ROCm guard `if non_blocking and getattr(torch.version, "hip", None):` '
        "was not found"
    )
    raise SystemExit(0)

indent = m.group(1)
replacement = (
    f"{indent}# {marker}: upstream disables pinned D2H staging on ROCm because\n"
    f"{indent}# forking the writer subprocesses afterwards segfaults. The hazard is\n"
    f"{indent}# fork-after-pinned, not the vendor: a TB-scale save pins ~320 GiB of\n"
    f"{indent}# host memory per node and the fork happens while it is all held.\n"
    f"{indent}# See patch_dist_ckpt_blocking_d2h.py.\n"
    f"{indent}if non_blocking:\n"
)
src = pattern.sub(lambda _: replacement, src, count=1)
filesystem_async_py.write_text(src)
print("Patched filesystem_async.py: D2H staging is now blocking on every platform")
