"""Thin wrapper around slime's convert_torch_dist_to_hf.py for Modal volumes.

Transparent pass-through to the upstream script.
"""

import os

_UPSTREAM = "/root/slime/tools/convert_torch_dist_to_hf.py"

if os.path.exists(_UPSTREAM):
    exec(compile(open(_UPSTREAM).read(), _UPSTREAM, "exec"))
else:
    raise FileNotFoundError(f"Upstream conversion script not found: {_UPSTREAM}")
