"""Raise SGLang's post-load barrier timeout for very large volume-backed models.

After each engine rank loads its weights, SGLang runs a ``monitored_barrier`` so
a rank that finished cannot silently wait forever on one that died
(``dist_barrier_after_load`` in
``sglang/srt/model_executor/model_runner_components/load_model_utils.py``). Its
budget is a module-level constant::

    UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # 8 minutes

Eight minutes is not enough for a terabyte-scale model whose weights live on a
Modal Volume, and the failure is badly disguised. Measured on
Nemotron-3-Ultra-550B (`plain-strategy-b1fbd96f9800`, 16 x 8 H200, 32-GPU
engines):

- the engine's **head** node finished loading in 88 s, because it had just
  downloaded the checkpoint itself and read it back out of warm page cache;
- every **other** node had to pull it cold from the Volume. Measured cold read
  throughput is **1.06 GiB/s for a single uncontended reader**, so the 1.02 TiB
  checkpoint takes **~16.4 min** — already 2x the budget before 15 nodes x 8
  ranks start contending for the same Volume;
- at 88 s + 480 s = 21:43:52 the barrier fired and every head rank raised
  ``ValueError: TP rank N could finish the model loading, but there are other
  ranks that didn't finish loading. It is likely due to unexpected failures
  (e.g., OOM) or a slow node.``

The engine then tore down its TCPStore, and the 24 ranks still loading spent the
next 20 minutes emitting ``Broken pipe`` / ``should dump`` NCCL warnings — which
is all that is visible unless you go looking for the original ValueError. There
was no OOM and no dead rank: the nodes were simply still reading.

This does **not** hardcode a bigger number. Raising the timeout has a real cost —
a genuinely dead rank stops being detected in 8 minutes and instead hangs the
engine for however long the new budget allows — and most miles models never need
it: a single-node engine's ranks all share the downloader's page cache, and the
barrier group is per-TP-group, so they never wait on a cold reader at all.
Baking 3600 s into the shared image would hand every small model a 60-minute hang
on a real failure in exchange for nothing.

So the patch makes the constant **configurable, defaulting to upstream's 480 s**:

    UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
        os.environ.get("MILES_LOAD_BARRIER_TIMEOUT_S", "480")
    )

A recipe that needs longer sets ``MILES_LOAD_BARRIER_TIMEOUT_S`` in its
``environment`` (Nemotron-3-Ultra sets 3600). Every other model keeps upstream's
fast failure detection unchanged, and the knob is reachable from a recipe without
another patch. The env var is read at import time, which is after the launcher has
put the recipe's ``environment`` into the container.

Executed at image-build time via ``python3 <this file>``.
"""

import re
from pathlib import Path

path = Path(
    "/sgl-workspace/sglang/python/sglang/srt/model_executor/"
    "model_runner_components/load_model_utils.py"
)

# Upstream's value, kept as the default so unaffected models are untouched.
DEFAULT_TIMEOUT_S = 480
ENV_VAR = "MILES_LOAD_BARRIER_TIMEOUT_S"

if not path.exists():
    print(f"{path} not found; skipping SGLang load-barrier patch")
    raise SystemExit(0)

src = path.read_text()
marker = "PATCHED_LOAD_BARRIER_TIMEOUT"

if marker in src:
    print("load_model_utils.py already patched for the load-barrier timeout")
    raise SystemExit(0)

pattern = re.compile(
    r"^UNBALANCED_MODEL_LOADING_TIMEOUT_S\s*=\s*(\d+).*$", re.MULTILINE
)
m = pattern.search(src)

if not m:
    print(
        "WARNING: could not patch load_model_utils.py — "
        "`UNBALANCED_MODEL_LOADING_TIMEOUT_S = <int>` not found"
    )
    raise SystemExit(0)

old_value = m.group(1)
replacement = (
    f"import os as _mtg_os  # {marker}\n"
    f"# {marker}: was a hardcoded {old_value}. A TB-scale checkpoint on a\n"
    "# Modal Volume reads at ~1 GiB/s cold, so an engine's non-downloading\n"
    "# nodes can need ~30 min; only the node that downloaded it has the file\n"
    "# in page cache. Raising this for everyone would cost small models their\n"
    "# fast dead-rank detection, so it is a per-recipe opt-in via\n"
    f"# ${ENV_VAR}. See patch_sglang_load_barrier.py.\n"
    "UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(\n"
    f'    _mtg_os.environ.get("{ENV_VAR}", "{DEFAULT_TIMEOUT_S}")\n'
    ")"
)
src = pattern.sub(lambda _: replacement, src, count=1)
path.write_text(src)
print(
    f"Patched load_model_utils.py: UNBALANCED_MODEL_LOADING_TIMEOUT_S "
    f"{old_value} -> ${ENV_VAR} (default {DEFAULT_TIMEOUT_S})"
)
