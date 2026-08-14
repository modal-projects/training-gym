#!/usr/bin/env python3
"""GPU/system sampler for Modal jobs — stdlib only, safe to import anywhere.

Appends one schema.MonitorSample JSON line per tick to a .jsonl the
observatory ingests as `system_monitor`. Opt-in wiring inside any Modal GPU
function is three lines:

    from observatory.sampler import start_sampler
    stop = start_sampler("/out/obs/<tag>/system_monitor.jsonl")  # daemon thread
    stop.set()  # optional at job end; the daemon dies with the container anyway

Volume note: when out_path lives on a modal.Volume, appended samples only
become visible to readers after volume.commit(). The sampler never commits —
for long jobs, commit periodically from the training loop (or accept that
samples land at Modal's function-exit commit).

Standalone: python3 -m observatory.sampler <out_path> [--interval 10]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
    "--format=csv,noheader,nounits",
]


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None  # "[N/A]" etc.


def _gpu_samples():
    """Per-GPU dicts from nvidia-smi, or None on CPU-only hosts / any failure."""
    try:
        out = subprocess.run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, util, mem_used, mem_total, temp, power = (_num(p) for p in parts)
        gpus.append({
            "id": int(idx) if idx is not None else None,
            "util_pct": util,
            "mem_used_mib": mem_used,
            "mem_total_mib": mem_total,
            "temp_c": temp,
            "power_w": power,
        })
    return gpus or None


def _mem_gib():
    """(used_gib, total_gib) from /proc/meminfo; (None, None) on darwin etc."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = float(rest.split()[0])  # kB
        total = info["MemTotal"] / 2**20
        avail = info.get("MemAvailable", info.get("MemFree", 0.0)) / 2**20
        return round(total - avail, 3), round(total, 3)
    except (OSError, KeyError, ValueError, IndexError):
        return None, None


def sample() -> dict:
    """One MonitorSample (schema.py) for right now."""
    gpus = _gpu_samples()
    try:
        load1, load5, _ = os.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = None
    mem_used, mem_total = _mem_gib()
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gpu": gpus[0] if gpus else None,
        "gpus": gpus,
        "cpu_load_1m": load1,
        "cpu_load_5m": load5,
        "mem_used_gib": mem_used,
        "mem_total_gib": mem_total,
        "source": "modal-sampler",
    }


def start_sampler(out_path: str, interval_s: float = 10) -> threading.Event:
    """Append one sample line to out_path every interval_s. Returns the stop Event."""
    stop = threading.Event()
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)

    def _loop():
        while not stop.is_set():
            line = json.dumps(sample())
            try:
                with open(out_path, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass  # volume hiccup; try again next tick
            stop.wait(interval_s)

    threading.Thread(target=_loop, name="obs-sampler", daemon=True).start()
    return stop


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample GPU/CPU/mem into a MonitorSample jsonl.")
    ap.add_argument("out_path")
    ap.add_argument("--interval", type=float, default=10)
    args = ap.parse_args()
    stop = start_sampler(args.out_path, args.interval)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
