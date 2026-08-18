"""Observatory ingestion CLI.

  python3 -m observatory.cli ingest <dir> [--no-upload] [--data-dir D] [--archive-workspace]
  python3 -m observatory.cli watch  <dir> [--interval 20] [--no-upload] [--data-dir D] [--archive-workspace]

ingest: normalize one run dir into a staging dir shaped exactly like the
volume run dir (record.json / workspace.json / status.json / raw/) and upload.
watch: live mode — stamp arrival times for new complete trace lines into
.obs/line_ts.jsonl, sample local CPU/mem into .obs/system_monitor.jsonl,
re-ingest every --interval seconds (workspace snapshot skipped while running),
finalize + exit when solve_status.txt appears.

--data-dir D stages into D/runs/<run_id>/ — the exact layout the viewer's
`app.py --data-dir D` reads — so local (no-Modal) viewing needs no volume at
all. Upload to the shared volume still happens unless --no-upload is given;
the two targets are independent.
--archive-workspace adds raw/workspace.tar.gz — the full submission folder
(minus corpus/.git/venvs/caches) — so teammates can pull the exact workspace
from the volume, not just its inlined snapshot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from . import local_monitor, schema, volume_io
from .normalize import atif, collect

RAW_FILES = ("trace.jsonl", "audit.json", "prompt.txt", "solve.err", "solve_status.txt")

# Kept out of the workspace archive: bulk that is reproducible from elsewhere
# (corpus is distributed separately; venvs/caches rebuild; agents/_runs nests
# the run dir itself and would recurse).
ARCHIVE_EXCLUDE_COMPONENTS = {".git", ".obs", "node_modules", "__pycache__"}
ARCHIVE_EXCLUDE_COMPONENT_PREFIXES = (".venv",)


def _archive_filter(ws_root: Path):
    def keep(info: tarfile.TarInfo):
        parts = Path(info.name).parts  # Path() drops any leading "./"
        for i, part in enumerate(parts):
            if part in ARCHIVE_EXCLUDE_COMPONENTS or \
                    part.startswith(ARCHIVE_EXCLUDE_COMPONENT_PREFIXES):
                return None
            # the run dirs nest under agents/_runs and would recurse the archive
            if part == "_runs" and i > 0 and parts[i - 1] == "agents":
                return None
        # workspace corpus lives at task/corpus (pre-2026-08-12 layout:
        # tasks/<task>/corpus — keep pruning old archives too)
        if len(parts) >= 2 and parts[0] == "task" and parts[1] == "corpus":
            return None
        if len(parts) >= 3 and parts[0] == "tasks" and parts[2] == "corpus":
            return None
        # the pinned task-model weights live in the workspace since 2026-08-12
        # (workspace/model/) — 15+ GB of safetensors that ballooned the archive
        if parts and parts[0] == "model":
            return None
        return info

    return keep


def archive_workspace(ws_root: Path, out_path: Path) -> int:
    """tar.gz the workspace (submission folder) into out_path; returns bytes."""
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(str(ws_root), arcname=".", filter=_archive_filter(ws_root))
    return out_path.stat().st_size


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n")


def stage(record: dict, ws_snapshot, run_dir: Path, stage_dir: Path) -> Path:
    stage_dir = Path(stage_dir)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    raw = stage_dir / schema.RAW_DIR
    raw.mkdir(parents=True)
    _write_json(stage_dir / schema.RECORD_FILE, record)
    if ws_snapshot is not None:
        _write_json(stage_dir / schema.WORKSPACE_FILE, ws_snapshot)
    _write_json(stage_dir / schema.STATUS_FILE, collect.build_status(record))
    _write_json(stage_dir / schema.TRAJECTORY_FILE,
                atif.events_to_atif(record, record.get("events") or []))
    for name in RAW_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, raw / name)
    sidecar = run_dir / ".obs" / "line_ts.jsonl"
    if sidecar.exists():
        shutil.copy2(sidecar, raw / "line_ts.jsonl")
    return stage_dir


def _counts(record: dict, ws_snapshot) -> str:
    bits = [f"{len(record['events'])} events",
            f"{len(record['sessions'])} sessions",
            f"{len(record['scores']['results'])} results",
            f"{len(record['system_monitor'])} monitor samples"]
    if ws_snapshot is not None:
        bits.append(f"{ws_snapshot['total_files']} ws files "
                    f"({ws_snapshot['inlined_files']} inlined)")
    return ", ".join(bits)


def _stage_root(args, run_id: str) -> Path:
    """--data-dir D → D/runs/<run_id> (the viewer's local layout); else temp."""
    if getattr(args, "data_dir", None):
        return Path(args.data_dir) / schema.RUNS_PREFIX / run_id
    if getattr(args, "stage_dir", None):
        return Path(args.stage_dir)
    return Path(tempfile.mkdtemp(prefix=f"obs_{run_id}_"))


def _maybe_archive(args, ws_root, stage_dir: Path) -> None:
    if not getattr(args, "archive_workspace", False):
        return
    if ws_root is None:
        print("archive-workspace: no workspace root resolved — skipped")
        return
    out = stage_dir / schema.RAW_DIR / "workspace.tar.gz"
    size = archive_workspace(Path(ws_root), out)
    print(f"archived workspace -> raw/workspace.tar.gz ({size / 1e6:.1f} MB)")


def cmd_ingest(args) -> int:
    record, ws_snapshot = collect.build_record(args.dir)
    run_id = record["meta"]["run_id"]
    run_dir = Path(record["meta"]["run_dir"])
    stage_dir = _stage_root(args, run_id)
    stage(record, ws_snapshot, run_dir, stage_dir)
    _, ws_root = collect.resolve_dirs(args.dir)
    _maybe_archive(args, ws_root, stage_dir)
    print(f"{run_id}: {_counts(record, ws_snapshot)}")
    print(f"staged: {stage_dir}")
    if args.no_upload:
        print(f"volume path (not uploaded): {schema.run_paths(run_id)['base']}")
    else:
        dest = volume_io.push_run(stage_dir, run_id)
        print(f"uploaded: {dest}")
    return 0


# ---- watch ----

def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _resume(trace: Path, sidecar: Path) -> tuple[int, int]:
    """(byte_offset, last_stamped_line) from a prior watcher's sidecar."""
    line_no = 0
    if sidecar.exists():
        for row in sidecar.read_text(errors="replace").splitlines():
            try:
                line_no = max(line_no, int(json.loads(row).get("line", 0)))
            except (ValueError, json.JSONDecodeError):
                pass
    offset = 0
    if line_no and trace.exists():
        with open(trace, "rb") as f:
            for _ in range(line_no):
                if not f.readline():
                    break
            offset = f.tell()
    return offset, line_no


def _tail(trace: Path, sidecar: Path, offset: int, line_no: int,
          ts: str) -> tuple[int, int, int]:
    """Stamp arrival ts for newline-terminated lines past offset; a trailing
    partial line stays unstamped until a later pass sees its newline."""
    if not trace.exists():
        return offset, line_no, 0
    with open(trace, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    rows = []
    pos = 0
    while (nl := chunk.find(b"\n", pos)) >= 0:
        line_no += 1
        rows.append({"line": line_no, "ts": ts})
        pos = nl + 1
    offset += pos
    if rows:
        _append_jsonl(sidecar, rows)
    return offset, line_no, len(rows)


def cmd_watch(args) -> int:
    run_dir, ws_root = collect.resolve_dirs(args.dir)
    run_id = run_dir.name
    obs = run_dir / ".obs"
    obs.mkdir(exist_ok=True)
    trace = run_dir / "trace.jsonl"
    sidecar = obs / "line_ts.jsonl"
    offset, line_no = _resume(trace, sidecar)
    stage_dir = _stage_root(args, run_id)
    print(f"watching {run_dir} (interval {args.interval}s, "
          f"{'no upload' if args.no_upload else 'uploading'})")

    # Live workspace: a full snapshot every pass is too heavy (walks every
    # cloned-package file, multi-MB JSON re-uploaded each cycle), but only-at-
    # the-end left the Workspace tab empty for a whole 24h run. Rebuild it
    # every ~10 minutes and reuse the cached one in between.
    snap_every = max(1, round(600 / max(1, args.interval)))
    ws_cache = None
    pass_no = 0
    while True:
        ts = collect.now_iso()
        offset, line_no, new = _tail(trace, sidecar, offset, line_no, ts)
        _append_jsonl(obs / "system_monitor.jsonl", [local_monitor.sample()])
        done = (run_dir / "solve_status.txt").exists()
        snap_now = done or pass_no % snap_every == 0
        pass_no += 1
        record, ws_snapshot = collect.build_record(run_dir, include_workspace=snap_now)
        if ws_snapshot is not None:
            ws_cache = ws_snapshot
        stage(record, ws_cache, run_dir, stage_dir)
        if done:
            # the workspace is final only now — archive on the last pass
            _maybe_archive(args, ws_root, stage_dir)
        if not args.no_upload:
            volume_io.push_run(stage_dir, run_id)
        state = record["index_row"]["state"]
        print(f"[{ts}] {run_id}: +{new} lines ({line_no} total), state={state}")
        if done:
            print(f"final ingest done: {_counts(record, ws_snapshot)}")
            return 0
        time.sleep(args.interval)


def cmd_sync_scores(args) -> int:
    """Upload the operator's LEADERBOARD.jsonl to the volume root so the viewer
    can overlay official test scores (by run_id) next to self-reported dev
    scores. Run after recording a score; safe to re-run (overwrites)."""
    src = Path(args.leaderboard)
    if not src.is_file():
        print(f"no leaderboard at {src}", file=sys.stderr)
        return 2
    n = sum(1 for line in src.read_text(errors="replace").splitlines() if line.strip())
    if args.data_dir:
        dest_local = Path(args.data_dir) / "leaderboard.jsonl"
        dest_local.parent.mkdir(parents=True, exist_ok=True)
        dest_local.write_bytes(src.read_bytes())
        print(f"staged {n} rows -> {dest_local}")
    if not args.no_upload:
        dest = volume_io.push_file(src, "leaderboard.jsonl")
        print(f"synced {n} leaderboard rows -> {dest}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="observatory.cli",
                                 description="Learning Agent Observatory ingestion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(
        no_upload="skip the shared-volume upload",
        data_dir="also stage into <D>/runs/<run_id>/ — the layout `app.py --data-dir D` serves",
        archive="add raw/workspace.tar.gz (full submission folder, minus corpus/.git/venvs)",
    )

    p = sub.add_parser("ingest", help="normalize one run dir and upload")
    p.add_argument("dir", help="ws_* dir, its workspace/, or an agents/_runs/<name> dir")
    p.add_argument("--no-upload", action="store_true", help=common["no_upload"])
    p.add_argument("--data-dir", default=None, help=common["data_dir"])
    p.add_argument("--stage-dir", default=None, help="explicit staging dir (overridden by --data-dir)")
    p.add_argument("--archive-workspace", action="store_true", help=common["archive"])
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("watch", help="live re-ingest until the run finishes")
    p.add_argument("dir")
    p.add_argument("--interval", type=float, default=20)
    p.add_argument("--no-upload", action="store_true", help=common["no_upload"])
    p.add_argument("--data-dir", default=None, help=common["data_dir"])
    p.add_argument("--archive-workspace", action="store_true", help=common["archive"])
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("sync-scores",
                       help="upload runs/LEADERBOARD.jsonl so the viewer shows official test scores")
    p.add_argument("--leaderboard", default=str(volume_io.REPO_ROOT / "runs" / "LEADERBOARD.jsonl"),
                   help="path to the operator leaderboard (default: repo runs/LEADERBOARD.jsonl)")
    p.add_argument("--no-upload", action="store_true", help=common["no_upload"])
    p.add_argument("--data-dir", default=None,
                   help="also copy to <D>/leaderboard.jsonl for a local app.py --data-dir viewer")
    p.set_defaults(fn=cmd_sync_scores)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
