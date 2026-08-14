"""Meter GPU usage from the Modal control plane and overlay it per run.

`runs/GPU_LOG.jsonl` is the agent's SELF-REPORT: it misses sandboxes the
agent launches through its own Modal code (observed live: an H200 vLLM
server burning for an hour while the dashboard read 0.0) and loses lines on
hard exits. This module computes the authoritative number from the compute
environment's sandbox history — the raw `SandboxList` gRPC carries
created_at, task started/finished, gpu_type, and gpu count for finished and
live sandboxes alike — and attributes GPU-seconds to runs by time-window
overlap with each run's [launched_at, finished_at] from the viewer index.

    python3 -m observatory.gpu_metering                       # meter + upload
    python3 -m observatory.gpu_metering --no-upload --verbose # inspect only

Output: `gpu_metered.json` at the observatory volume root (the same overlay
pattern as leaderboard.jsonl; app.py merges it into index rows as
`gpu_hours_metered`). Attribution clips each sandbox's GPU interval to the
run window, so a sandbox spanning two run windows contributes only its
overlap to each — and such rows are flagged `shared_window` on both runs.
GPU time inside the environment but outside every run window is reported as
`unattributed_gpu_hours` (operator debugging, deleted runs, clock drift).

Sandbox tags: rows carrying a `learning_agent_run_id` tag (gpu_launcher sets
it when
$LEARNING_AGENT_RUN_ID is in the workspace env) are attributed to that run directly,
window or not. Untagged rows fall back to the time window.

Stdlib-only at import time; `modal` loads lazily inside fetch_sandboxes().
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_API_BASE = "https://modal-labs-lab-dev--lab-observatory-web.modal.run"
DEFAULT_COMPUTE_ENV = "lab-dev"
OVERLAY_FILE = "gpu_metered.json"
_PAGE_LIMIT = 200  # safety valve on SandboxList pagination


def _iso_to_ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def fetch_sandboxes(compute_env: str, since: float = 0.0) -> list[dict]:
    """All sandboxes (finished + live) in the environment, newest first.
    Returns plain dicts so attribution stays testable without modal."""
    import asyncio

    from modal.client import _Client
    from modal_proto import api_pb2

    async def _fetch() -> list[dict]:
        client = await _Client.from_env()
        seen: dict[str, object] = {}
        before = 0.0
        for _ in range(_PAGE_LIMIT):
            resp = await client.stub.SandboxList(api_pb2.SandboxListRequest(
                environment_name=compute_env, include_finished=True,
                before_timestamp=before))
            if not resp.sandboxes:
                break
            for s in resp.sandboxes:
                seen[s.id] = s
            page_oldest = min(s.created_at for s in resp.sandboxes)
            if page_oldest <= since or page_oldest >= before and before:
                break
            before = page_oldest
        rows = []
        for s in seen.values():
            if since and s.created_at < since:
                continue
            ti = s.task_info
            gpu_type = (ti.gpu_type or "").upper()
            n_gpus = 0
            if gpu_type and gpu_type != "CPU":
                n_gpus = max(1, ti.gpu_config.count if ti.HasField("gpu_config") else 1)
            rows.append({
                "sandbox_id": s.id,
                "app_id": s.app_id,
                "tags": {t.tag_name: t.tag_value for t in s.tags},
                "created_at": s.created_at,
                "started_at": ti.started_at or None,
                "finished_at": ti.finished_at or None,
                "gpu_type": gpu_type or None,
                "n_gpus": n_gpus,
            })
        rows.sort(key=lambda r: -r["created_at"])
        return rows

    return asyncio.run(_fetch())


STALE_AFTER_S = 15 * 60  # a "running" run with no heartbeat this long is dead


def run_windows_from_api(api_base: str,
                         now: float | None = None) -> dict[str, tuple[float, float | None]]:
    """{run_id: (start_ts, end_ts | None-if-live)} from the viewer index.

    Only a run in state "running" with a FRESH heartbeat gets an open-ended
    window. Everything else (finished, "stale" — watcher died without a
    finished_at) closes at finished_at, else the last heartbeat (updated_at),
    else the budget cap. All windows are additionally capped at
    launched_at + time_budget_h + 30min — run.sh hard-kills at budget+grace,
    so nothing a run owns can outlive that; without the cap one dead run
    with a frozen record would swallow every later sandbox in the env."""
    now = now or time.time()
    with urllib.request.urlopen(f"{api_base.rstrip('/')}/api/runs", timeout=30) as r:
        rows = json.load(r)
    windows: dict[str, tuple[float, float | None]] = {}
    for row in rows:
        rid, start = row.get("run_id"), _iso_to_ts(row.get("launched_at"))
        if not rid or start is None:
            continue
        end = _iso_to_ts(row.get("finished_at"))
        if end is None:
            beat = _iso_to_ts(row.get("updated_at"))
            live = (row.get("state") == "running"
                    and beat is not None and now - beat < STALE_AFTER_S)
            end = None if live else beat  # dead: close at the last heartbeat
        budget_h = row.get("time_budget_h")
        if isinstance(budget_h, (int, float)) and budget_h > 0:
            cap = start + (budget_h + 0.5) * 3600
            if end is not None:
                end = min(end, cap)
            elif now > cap:
                end = cap
        if end is None and row.get("state") != "running":
            end = start  # no heartbeat, no budget: attribute nothing
        windows[rid] = (start, end)
    return windows


def attribute(sandboxes: list[dict],
              windows: dict[str, tuple[float, float | None]],
              now: float | None = None) -> dict:
    """Clip each GPU sandbox's active interval to every run window it
    overlaps. A `learning_agent_run_id` tag (or the legacy `lab_run_id`) wins outright; multi-window rows are
    flagged shared_window; GPU time outside all windows is reported."""
    now = now or time.time()
    runs: dict[str, dict] = {
        rid: {"gpu_hours": 0.0, "by_gpu": {}, "n_gpu_sandboxes": 0,
              "shared_window": False, "sandboxes": []}
        for rid in windows
    }
    unattributed = 0.0
    for sb in sandboxes:
        if not sb["n_gpus"]:
            continue  # CPU-only: free, out of scope for GPU-hours
        start = sb["started_at"] or sb["created_at"]
        end = sb["finished_at"] or now
        total = max(0.0, end - start)
        tagged = (sb["tags"].get("learning_agent_run_id")
                  or sb["tags"].get("lab_run_id"))   # legacy tag on old sandboxes
        hits: list[tuple[str, float]] = []
        if tagged and tagged in runs:
            hits = [(tagged, total)]
        else:
            for rid, (ws, we) in windows.items():
                ov = max(0.0, min(end, we or now) - max(start, ws))
                if ov > 0:
                    hits.append((rid, ov))
        covered = 0.0
        for rid, seconds in hits:
            entry = runs[rid]
            hours = seconds * sb["n_gpus"] / 3600.0
            entry["gpu_hours"] += hours
            entry["by_gpu"][sb["gpu_type"]] = (
                entry["by_gpu"].get(sb["gpu_type"], 0.0) + hours)
            entry["n_gpu_sandboxes"] += 1
            if len(hits) > 1:
                entry["shared_window"] = True
            entry["sandboxes"].append({
                "sandbox_id": sb["sandbox_id"], "app_id": sb["app_id"],
                "gpu": f"{sb['gpu_type']}:{sb['n_gpus']}",
                "seconds": round(seconds, 1),
                "tagged": bool(tagged),
                "shared": len(hits) > 1,
            })
            covered = max(covered, seconds)
        unattributed += max(0.0, total - covered) * sb["n_gpus"] / 3600.0
    for entry in runs.values():
        entry["gpu_hours"] = round(entry["gpu_hours"], 3)
        entry["by_gpu"] = {k: round(v, 3) for k, v in entry["by_gpu"].items()}
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "unattributed_gpu_hours": round(unattributed, 3),
        "runs": {rid: e for rid, e in runs.items() if e["n_gpu_sandboxes"]},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Meter GPU-hours from Modal sandbox history; upload the overlay.")
    ap.add_argument("--api-base", default=DEFAULT_API_BASE,
                    help="viewer URL for run windows")
    ap.add_argument("--compute-env", default=DEFAULT_COMPUTE_ENV,
                    help="Modal environment the runs compute in")
    ap.add_argument("--since-days", type=float, default=30.0,
                    help="ignore sandboxes created earlier than this many days ago")
    ap.add_argument("--stage", default="",
                    help="also write gpu_metered.json into this local dir")
    ap.add_argument("--no-upload", action="store_true",
                    help="compute and print only; skip the volume upload")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-sandbox attribution rows")
    args = ap.parse_args(argv)

    windows = run_windows_from_api(args.api_base)
    sandboxes = fetch_sandboxes(args.compute_env,
                                since=time.time() - args.since_days * 86400)
    overlay = attribute(sandboxes, windows)
    overlay["compute_env"] = args.compute_env

    for rid, e in sorted(overlay["runs"].items()):
        flag = " SHARED-WINDOW" if e["shared_window"] else ""
        print(f"[gpu-metered] {rid}: {e['gpu_hours']}h "
              f"({e['n_gpu_sandboxes']} gpu sandboxes){flag}")
        if args.verbose:
            for row in e["sandboxes"]:
                print(f"    {row['sandbox_id']} {row['gpu']} {row['seconds']}s"
                      f"{' tagged' if row['tagged'] else ''}"
                      f"{' shared' if row['shared'] else ''}")
    print(f"[gpu-metered] unattributed: {overlay['unattributed_gpu_hours']}h "
          f"across env {args.compute_env}")

    payload = json.dumps(overlay, indent=1) + "\n"
    if args.stage:
        dest = Path(args.stage) / OVERLAY_FILE
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload)
        print(f"[gpu-metered] staged -> {dest}")
    if not args.no_upload:
        from . import volume_io
        tmp = Path(__file__).resolve().parent / f".{OVERLAY_FILE}.tmp"
        tmp.write_text(payload)
        try:
            dest = volume_io.push_file(tmp, OVERLAY_FILE)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"[gpu-metered] uploaded -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
