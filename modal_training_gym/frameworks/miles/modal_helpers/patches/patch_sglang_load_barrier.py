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

This raises the constant to 3600 s. It only extends how long a *healthy* rank
waits for a slow peer; a genuinely dead rank still fails, just later. Nothing
about loading, sharding or serving changes, so an engine that came up inside 8
minutes behaves identically.

The constant is module-level with no environment variable and no CLI flag, so a
recipe cannot reach it — hence a patch.

Only single-node engines avoid this entirely (the barrier group is per-TP-group,
and a one-node engine's ranks all share the downloader's page cache). Any miles
model large enough to need a multi-node rollout engine is exposed, which is why
this is applied to every miles image rather than one recipe.

Executed at image-build time via ``python3 <this file>``.
"""

import re
from pathlib import Path

path = Path(
    "/sgl-workspace/sglang/python/sglang/srt/model_executor/"
    "model_runner_components/load_model_utils.py"
)

NEW_TIMEOUT_S = 3600

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
    f"UNBALANCED_MODEL_LOADING_TIMEOUT_S = {NEW_TIMEOUT_S}  # {marker}: was {old_value}."
    "\n# A TB-scale checkpoint on a Modal Volume reads at ~1 GiB/s cold, so nodes that"
    "\n# did not download it need far longer than 8 min. Only the downloader's own node"
    "\n# has it in page cache. See patch_sglang_load_barrier.py."
)
src = pattern.sub(lambda _: replacement, src, count=1)
path.write_text(src)
print(
    f"Patched load_model_utils.py: UNBALANCED_MODEL_LOADING_TIMEOUT_S "
    f"{old_value} -> {NEW_TIMEOUT_S}"
)
