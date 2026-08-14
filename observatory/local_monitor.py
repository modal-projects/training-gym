"""Local CPU/mem (+ GPU when nvidia-smi is around) sampler for the live
watcher. Returns schema.MonitorSample dicts with source="local-watcher"."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

_VM_PAGESIZE_RE = re.compile(r"page size of (\d+) bytes")
_VM_ROW_RE = re.compile(r"^([A-Za-z -]+):\s+(\d+)\.")


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout


def _mem_darwin() -> tuple[float, float]:
    total = int(_run(["sysctl", "-n", "hw.memsize"]).strip())
    out = _run(["vm_stat"])
    m = _VM_PAGESIZE_RE.search(out)
    page = int(m.group(1)) if m else 4096
    pages: dict[str, int] = {}
    for line in out.splitlines():
        if row := _VM_ROW_RE.match(line.strip()):
            pages[row.group(1).strip().lower()] = int(row.group(2))
    used = page * (pages.get("pages active", 0) + pages.get("pages wired down", 0)
                   + pages.get("pages occupied by compressor", 0))
    return used / 2**30, total / 2**30


def _mem_linux() -> tuple[float, float]:
    info: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            info[k.strip()] = int(v.split()[0])  # kB
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", info.get("MemFree", 0))
    return (total - avail) / 2**20, total / 2**20


def _num(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except ValueError:
        return None  # "[N/A]" etc.


def _gpus() -> Optional[list[dict]]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = _run(["nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        vals = [_num(c) for c in cols]
        gpus.append({"id": int(vals[0]) if vals[0] is not None else len(gpus),
                     "util_pct": vals[1], "mem_used_mib": vals[2],
                     "mem_total_mib": vals[3], "temp_c": vals[4],
                     "power_w": vals[5]})
    return gpus or None


def sample() -> dict:
    try:
        load1, load5, _ = os.getloadavg()
    except OSError:
        load1 = load5 = None
    used = total = None
    try:
        if sys.platform == "darwin":
            used, total = _mem_darwin()
        elif os.path.exists("/proc/meminfo"):
            used, total = _mem_linux()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    gpus = _gpus()
    out = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu": gpus[0] if gpus else None,
        "cpu_load_1m": round(load1, 2) if load1 is not None else None,
        "cpu_load_5m": round(load5, 2) if load5 is not None else None,
        "mem_used_gib": round(used, 2) if used is not None else None,
        "mem_total_gib": round(total, 2) if total is not None else None,
        "source": "local-watcher",
    }
    if gpus and len(gpus) > 1:
        out["gpus"] = gpus
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(sample(), indent=1))
