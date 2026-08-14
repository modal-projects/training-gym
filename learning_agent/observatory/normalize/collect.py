"""Run-dir collector: resolve a run dir, detect the trace format, and build
the RunRecord (+ WorkspaceSnapshot) defined in observatory/schema.py."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import schema
from . import claude_stream, codex, gemini, learning, opencode
from .workspace import snapshot

TASKS = ("dspy", "openclaw", "fav2")
PARSERS = {
    schema.TRACE_CLAUDE: claude_stream,
    schema.TRACE_CODEX: codex,
    schema.TRACE_GEMINI: gemini,
    schema.TRACE_OPENCODE: opencode,
    schema.TRACE_UNKNOWN: claude_stream,  # unknowns become system/unknown events
}

BASE_MODEL_RE = re.compile(r"student model\s+([A-Za-z0-9_.\-/]+)")
BUDGET_H_RE = re.compile(r"You have `?([0-9.]+)`? hours of wall-clock")

# tool_use block names that carry a shell command (in input["command"]), per
# trace format. Claude's Bash tool and codex's hoisted exec ("command", see
# codex.py) are the two the harness exercises; gemini/opencode entries are
# best-effort (their shell tools, when present) and any other tool name is
# skipped, so a format we don't recognize simply yields no learning actions
# rather than crashing. TRACE_UNKNOWN is parsed by claude_stream, so it shares
# claude's "Bash".
_COMMAND_TOOLS = {
    schema.TRACE_CLAUDE: {"Bash"},
    schema.TRACE_CODEX: {"command"},
    schema.TRACE_GEMINI: {"run_shell_command", "shell", "Shell"},
    schema.TRACE_OPENCODE: {"bash"},
    schema.TRACE_UNKNOWN: {"Bash"},
}
_LEARNING_CMD_MAX = 500  # LearningAction.command is trimmed to this many chars
# A folder-form tool entrypoint anywhere in a command (used to locate the tool
# dir for card lookups and manifest checks; mirrors learning.py's patterns).
_FOLDER_RUNPY_RE = re.compile(
    r"toolbox/[A-Za-z0-9_\-]+/(?:[A-Za-z0-9_\-]+/)*[A-Za-z0-9_\-]+/run\.py")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_json(path: Path) -> Optional[dict]:
    try:
        obj = json.loads(path.read_text(errors="replace"))
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _load_track(run_dir: Path, ws_root: Optional[Path]) -> Optional[str]:
    """run_meta.json = {"track","scaffold","task","hours","archetype","prepared_at"}, written by
    the seeding routine into $RUN_PARENT (== ws_root.parent for sandboxed runs). Checked at
    run_dir first, then ws_root.parent; missing/unparseable at both -> None. Runs that
    predate tracks (no run_meta.json anywhere, e.g. in-place runs) get track=None."""
    for candidate in (run_dir, ws_root.parent if ws_root is not None else None):
        if candidate is None:
            continue
        meta = _load_json(candidate / "run_meta.json")
        if meta is None:
            continue
        track = meta.get("track")
        if isinstance(track, str):
            return track
    return None


# ---- learning timeline (see observatory/normalize/learning.py) ----

def _load_seed_manifest(run_dir: Path, ws_root: Optional[Path]) -> Optional[set[str]]:
    """seed_manifest.txt = `git ls-tree -r HEAD` of the seed repo (lines like
    "100644 blob <sha>\\t<path>"), written by the seeding routine into $RUN_PARENT.
    Same two candidate locations as run_meta.json (run_dir first, then
    ws_root.parent). Returns the set of workspace-relative seed paths, or None
    when no manifest file exists at either location. The None-vs-empty-set
    distinction matters: None (no file) means "old run, can't tell seed from
    invented" and disables invented detection; an empty set (file present but
    empty) still enables it."""
    for candidate in (run_dir, ws_root.parent if ws_root is not None else None):
        if candidate is None:
            continue
        try:
            text = (candidate / "seed_manifest.txt").read_text(errors="replace")
        except OSError:
            continue
        paths: set[str] = set()
        for line in text.splitlines():
            tab = line.find("\t")  # "<mode> <type> <sha>\t<path>"
            if tab != -1:
                path = line[tab + 1:].strip()
                if path:
                    paths.add(path)
        return paths
    return None


def _command_text(block: dict, fmt: str) -> Optional[str]:
    """The shell command string in a tool_use block, or None when the block
    isn't a command-carrying tool_use for this format. Guards a missing/non-dict
    input and a non-string command so a malformed trace never crashes ingest."""
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return None
    if block.get("name") not in _COMMAND_TOOLS.get(fmt, frozenset()):
        return None
    inp = block.get("input")
    if not isinstance(inp, dict):
        return None
    cmd = inp.get("command")
    return cmd if isinstance(cmd, str) else None


def _ws_relative(token: str, ws_root: Optional[Path]) -> str:
    """Normalize a script token to a workspace-relative path for manifest /
    snapshot comparison: strip an absolute ws_root prefix (commands may invoke a
    script by absolute path under the workspace) and any leading "./"."""
    tok = token
    if ws_root is not None:
        prefix = str(ws_root) + "/"
        if tok.startswith(prefix):
            tok = tok[len(prefix):]
    while tok.startswith("./"):
        tok = tok[2:]
    return tok


def _collect_learning(events: list[dict], fmt: str, run_dir: Path,
                      ws_root: Optional[Path], ws_snapshot: Optional[dict]
                      ) -> tuple[list[dict], Optional[dict]]:
    """Walk normalized events and classify shell commands in their tool_use
    blocks into the learning timeline (schema.LearningAction).

    Provenance / invented policy (plan Task 6):
      - Registry matches get provenance "seed" when a seed manifest is available,
        else "unknown" (old runs predate seed_manifest.txt — show them as
        unknown rather than guessing).
      - Invented-tool detection runs only when BOTH the manifest AND a workspace
        snapshot are present. A script token absent from the manifest but present
        in the snapshot's file list is an invented tool (kind "tool",
        provenance "invented"). Absent from the snapshot -> skipped as noise (a
        typo or a temp script deleted before the snapshot was taken is therefore
        missed; accepted per the plan). No manifest or no snapshot -> no invented
        entries at all.

    nth_use is a 1-based per-tool-identity counter in event order. Identity is
    (kind, tool), which is unique across registry kinds (data/train/eval/evolve)
    and invented tools (kind "tool", tool = the script path).

    Returns (learning_actions, learning_counts). learning_counts is None when no
    actions were found, so the index row stays clean for non-agent/empty runs."""
    manifest = _load_seed_manifest(run_dir, ws_root)
    provenance = "seed" if manifest is not None else "unknown"
    snapshot_paths: Optional[set[str]] = None
    if manifest is not None and ws_snapshot is not None:
        snapshot_paths = {f["path"] for f in (ws_snapshot.get("files") or [])
                          if isinstance(f, dict) and isinstance(f.get("path"), str)}

    actions: list[dict] = []
    nth: dict[tuple, int] = {}

    def emit(event_i, ts, kind, tool, prov, command, args) -> None:
        key = (kind, tool)
        nth[key] = nth.get(key, 0) + 1
        actions.append({
            "event_i": event_i, "ts": ts, "kind": kind, "tool": tool,
            "provenance": prov, "command": command[:_LEARNING_CMD_MAX],
            "args": args, "nth_use": nth[key],
        })

    # Inlined snapshot contents, for reading invented tool cards (tool.yaml)
    # without touching the filesystem twice.
    snapshot_text: dict[str, str] = {}
    if ws_snapshot is not None:
        snapshot_text = {f["path"]: f.get("content") or ""
                         for f in (ws_snapshot.get("files") or [])
                         if isinstance(f, dict) and isinstance(f.get("path"), str)}

    def refine_folder_hit(hit: dict, cmd: str) -> tuple[str, str]:
        """(kind, provenance) for a registry hit, folder-form aware:
        - a folder tool whose run.py is NOT in the seed manifest is invented,
          registry match or not (the structural patterns match inventions too);
        - a catch-all hit (kind "tool" — unknown category) takes its kind from
          the tool's own card in the snapshot when one exists (kind: <value>,
          parsed by line to keep this module dependency-free)."""
        kind, prov = hit["kind"], provenance
        m = _FOLDER_RUNPY_RE.search(cmd)
        if m is None:
            return kind, prov
        run_rel = _ws_relative(m.group(0), ws_root)
        if manifest is not None and run_rel not in manifest:
            prov = "invented"
        if kind == "tool":
            card = snapshot_text.get(run_rel.rsplit("/", 1)[0] + "/tool.yaml", "")
            for line in card.splitlines():
                if line.startswith("kind:"):
                    val = line.split(":", 1)[1].strip()
                    if val in ("data", "train", "eval", "evolve", "harness", "infra"):
                        kind = val
                    break
        return kind, prov

    for event in events:
        event_i = event.get("i")
        ts = event.get("ts")
        for block in event.get("blocks") or []:
            cmd = _command_text(block, fmt)
            if cmd is None:
                continue
            for hit in learning.classify_command(cmd):
                kind, prov = refine_folder_hit(hit, cmd)
                emit(event_i, ts, kind, hit["tool"], prov, cmd,
                     hit.get("args") or {})
            if snapshot_paths is None:
                continue  # no manifest or no snapshot -> skip invented detection
            for token in learning.extract_script_paths(cmd):
                rel = _ws_relative(token, ws_root)
                if rel in manifest or rel not in snapshot_paths:
                    continue  # seed script (in manifest) or noise (not in snapshot)
                emit(event_i, ts, "tool", rel, "invented", cmd, {})

    # Invented-tool CARDS: any tool doc in the snapshot whose sibling run.py
    # is absent from the seed manifest is a tool this run created — surface it
    # so the dashboard can render it with attribution, whether or not the
    # trace shows it being executed. Current form is tool.md; tool.yaml (+
    # README) is the pre-2026-08-05 form and old workspaces keep parsing.
    cards: list[dict] = []
    if manifest is not None:
        seen_dirs: set[str] = set()
        for path, text in sorted(snapshot_text.items()):
            if not path.startswith("toolbox/"):
                continue
            if path.endswith("/tool.md"):
                doc_key = "tool_md"
            elif path.endswith("/tool.yaml"):
                doc_key = "tool_yaml"
            else:
                continue
            tool_dir = path.rsplit("/", 1)[0]
            if f"{tool_dir}/run.py" in manifest or tool_dir in seen_dirs:
                continue
            seen_dirs.add(tool_dir)
            cards.append({"path": tool_dir, doc_key: text,
                          "readme": snapshot_text.get(f"{tool_dir}/README.md", "")})

    if not actions and not cards:
        return [], None, []
    counts = {"data": 0, "train": 0, "eval": 0, "evolve": 0, "harness": 0, "infra": 0, "invented_tools": 0}
    invented: set[str] = set()
    for a in actions:
        if a["kind"] in ("data", "train", "eval", "evolve", "harness", "infra"):
            counts[a["kind"]] += 1
        if a["kind"] == "tool" or a["provenance"] == "invented":
            invented.add(a["tool"])
    counts["invented_tools"] = max(len(invented), len(cards))
    return actions, counts, cards


# ---- dir resolution + run-id parsing ----

def resolve_dirs(path) -> tuple[Path, Optional[Path]]:
    """Accept a ws_* dir, its workspace/ dir, or an inner agents/_runs/<name>
    run dir; return (run_dir, ws_root). ws_root is None when unresolvable."""
    p = Path(path).resolve()
    ws = None
    if (p / "workspace").is_dir():
        ws = p / "workspace"
    elif (p / "agents" / "_runs").is_dir():
        ws = p
    if ws is not None:
        runs = sorted(d for d in (ws / "agents" / "_runs").iterdir() if d.is_dir())
        if not runs:
            raise FileNotFoundError(f"no run dirs under {ws}/agents/_runs")
        return runs[-1], ws
    # inner run dir
    if any((p / n).exists() for n in ("trace.jsonl", "solve_status.txt", "prompt.txt")):
        if p.parent.name == "_runs" and p.parent.parent.name == "agents":
            ws = p.parent.parent.parent
        return p, ws
    raise FileNotFoundError(f"not a run dir or workspace dir: {p}")


def split_run_id(run_id: str) -> tuple[str, Optional[str]]:
    """<scaffold>_<task>_<stamp>; scaffold may contain underscores, so the task
    token is matched from the right (a stamp must follow it)."""
    parts = run_id.split("_")
    for j in range(len(parts) - 2, 0, -1):
        if parts[j] in TASKS:
            return "_".join(parts[:j]), parts[j]
    return run_id, None


# ---- trace format detection ----

def detect_format(trace_path: Path) -> str:
    if not trace_path.exists():
        return schema.TRACE_UNKNOWN
    try:
        with open(trace_path, encoding="utf-8", errors="replace") as f:
            head = [f.readline() for _ in range(20)]
    except OSError:
        return schema.TRACE_UNKNOWN
    for _, obj, _, _ in claude_stream.iter_lines(head):
        if obj is None:
            continue
        t = obj.get("type")
        if t == "thread.started":
            return schema.TRACE_CODEX
        if t in claude_stream.CLAUDE_TYPES:
            return schema.TRACE_CLAUDE
        if isinstance(obj.get("msg"), dict) or t in ("session_configured", "task_started"):
            return schema.TRACE_CODEX
        if "part" in obj or t in ("step_start", "step_finish"):
            return schema.TRACE_OPENCODE
        if "method" in obj or t in ("init", "message", "tool_use", "tool_result"):
            return schema.TRACE_GEMINI
    return schema.TRACE_UNKNOWN


def _gpu_hours(gpu_log: list) -> "int | None":
    """GPU-hours from runs/GPU_LOG.jsonl, billing-style integers: sum
    seconds x n_gpus over the jobs, then round UP — 1 GPU for any part of
    an hour counts 1, 4 GPUs count 4 (CPU sandboxes count zero). None when
    the run never logged a GPU job."""
    total = 0.0
    seen = False
    for row in gpu_log:
        if not isinstance(row, dict):
            continue
        seen = True
        try:
            total += float(row.get("seconds", 0)) * int(row.get("n_gpus", 0))
        except (TypeError, ValueError):
            continue
    if not seen:
        return None
    import math
    return math.ceil(total / 3600)


# ---- scores ----

def _collect_scores(ws_root: Optional[Path]) -> dict:
    scores = {"checkpoints": [], "learning_log": [], "leaderboard": [],
              "results": [], "gpu_log": []}
    if ws_root is None:
        return scores
    runs = ws_root / "runs"
    if not runs.is_dir():
        return scores
    scores["checkpoints"] = _load_jsonl(runs / "CHECKPOINTS.jsonl")
    scores["learning_log"] = _load_jsonl(runs / "LEARNING_LOG.jsonl")
    # The learning log (2026-08-11+) subsumes the checkpoint ledger: its
    # checkpoint-kind entries feed the same dev-score/tag machinery, so both
    # eras of runs flow through one path.
    scores["checkpoints"] += [
        e for e in scores["learning_log"]
        if e.get("kind") == "checkpoint" and e.get("tag")]
    scores["gpu_log"] = _load_jsonl(runs / "GPU_LOG.jsonl")

    tag_dirs = sorted(d for d in runs.iterdir() if d.is_dir())
    tags = {d.name for d in tag_dirs} | {c.get("tag") for c in scores["checkpoints"]}
    scores["leaderboard"] = [row for row in _load_jsonl(runs / "LEADERBOARD.jsonl")
                             if row.get("tag") in tags]

    for tag_dir in tag_dirs:
        for budget_dir in sorted(tag_dir.glob("budget_*")):
            try:
                budget = int(budget_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            eval_meta = _load_json(budget_dir / "eval_meta.json") or {}
            for rf in sorted(budget_dir.glob("results_*.json")):
                data = _load_json(rf)
                if data is None:
                    continue
                entry = dict(data)  # keep file verbatim: mean/claim_score untouched
                entry["tag"] = tag_dir.name
                entry["split"] = rf.stem.split("_", 1)[1]
                entry["budget"] = budget
                per_q = entry.get("per_question")
                if isinstance(per_q, dict):
                    for qid, m in eval_meta.items():
                        if qid in per_q and isinstance(m, dict):
                            per_q[qid].update(m)
                tcs = [m["tool_calls"] for m in eval_meta.values()
                       if isinstance(m, dict) and isinstance(m.get("tool_calls"), (int, float))]
                entry["tool_calls_avg"] = round(sum(tcs) / len(tcs), 2) if tcs else None
                scores["results"].append(entry)
    return scores


# ---- index-row derivations ----

def _derive_flags(results: list[dict]) -> tuple[Optional[bool], Optional[str]]:
    canon_vals = [r.get("canonical") for r in results]
    if any(v is False for v in canon_vals):
        canonical = False
    elif any(v is True for v in canon_vals):
        canonical = True
    else:
        canonical = None
    integ_vals = [r.get("integrity") for r in results if r.get("integrity")]
    if any(v == "DIRTY" for v in integ_vals):
        integrity = "DIRTY"
    elif integ_vals:
        integrity = "OK"
    else:
        integrity = None
    return canonical, integrity


def _best_dev(results: list[dict], checkpoints: list[dict]) -> tuple[Optional[float], Optional[list], Optional[str]]:
    dev = [r for r in results
           if r.get("split") == "dev" and isinstance(r.get("mean"), (int, float))]
    if dev:
        best = max(dev, key=lambda r: r["mean"])
        return best["mean"], best.get("bootstrap_ci95"), best.get("tag")
    # No harness results files — agents that measure themselves through their own
    # instruments (e.g. the opencode scaffolds) still register dev_score in the
    # checkpoint ledger. Self-reported by definition; no CI available.
    cps = [c for c in checkpoints if isinstance(c.get("dev_score"), (int, float))]
    if not cps:
        return None, None, None
    best = max(cps, key=lambda c: c["dev_score"])
    return best["dev_score"], None, best.get("tag")


# ---- the collector ----

def build_record(path, include_workspace: bool = True) -> tuple[dict, Optional[dict]]:
    run_dir, ws_root = resolve_dirs(path)
    run_id = run_dir.name
    scaffold, task = split_run_id(run_id)

    trace_path = run_dir / "trace.jsonl"
    line_ts = {r["line"]: r["ts"] for r in _load_jsonl(run_dir / ".obs" / "line_ts.jsonl")
               if isinstance(r.get("line"), int) and r.get("ts")}
    fmt = detect_format(trace_path)
    lines = (trace_path.read_text(errors="replace").splitlines()
             if trace_path.exists() else [])
    parsed = PARSERS[fmt].parse_trace(lines, line_ts or None)
    events, sessions, summary = parsed["events"], parsed["sessions"], parsed["summary"]

    exit_code = duration_s = None
    status_path = run_dir / "solve_status.txt"
    if status_path.exists():
        for line in status_path.read_text(errors="replace").splitlines():
            k, _, v = line.partition("=")
            try:
                if k.strip() == "exit":
                    exit_code = int(v)
                elif k.strip() == "seconds":
                    duration_s = int(v)
            except ValueError:
                pass
        state = schema.STATE_FINISHED if exit_code == 0 else schema.STATE_ERROR
    else:
        state = schema.STATE_RUNNING

    base_model = time_budget_h = None
    prompt_path = run_dir / "prompt.txt"
    if prompt_path.exists():
        prompt = prompt_path.read_text(errors="replace")
        if m := BASE_MODEL_RE.search(prompt):
            base_model = m.group(1)
        if m := BUDGET_H_RE.search(prompt):
            time_budget_h = float(m.group(1))

    scores = _collect_scores(ws_root)
    track = _load_track(run_dir, ws_root)
    audit = _load_json(run_dir / "audit.json")
    monitor = _load_jsonl(run_dir / ".obs" / "system_monitor.jsonl")
    ws_snapshot = snapshot(ws_root) if include_workspace and ws_root else None
    learning_actions, learning_counts, invented_cards = _collect_learning(
        events, fmt, run_dir, ws_root, ws_snapshot)

    launched_at = parsed["meta_bits"].get("launched_at")
    finished_at = parsed["meta_bits"].get("finished_at") if state != schema.STATE_RUNNING else None
    best_score, best_ci, best_tag = _best_dev(scores["results"], scores["checkpoints"])
    gpu_hours = _gpu_hours(scores["gpu_log"])
    canonical, integrity = _derive_flags(scores["results"])
    build_ts = now_iso()

    meta = {
        "run_id": run_id, "run_dir": str(run_dir), "scaffold": scaffold,
        "task": task, "base_model": base_model, "trace_format": fmt,
        "time_budget_h": time_budget_h, "launched_at": launched_at,
        "finished_at": finished_at, "exit_code": exit_code, "track": track,
        "build_ts": build_ts, "schema_version": schema.SCHEMA_VERSION,
    }
    index_row = {
        "run_id": run_id, "kind": "agent_run", "state": state,
        "task": task, "scaffold": scaffold,
        "agent_model": (summary["agent_models"] or [None])[0],
        "base_model": base_model, "trace_format": fmt,
        "time_budget_h": time_budget_h,
        "launched_at": launched_at, "finished_at": finished_at,
        "duration_s": duration_s,
        "num_turns": summary["num_turns"], "num_events": len(events),
        "session_count": summary["session_count"],
        "total_cost_usd": summary["total_cost_usd"],
        "best_dev_score": best_score, "best_dev_ci": best_ci, "best_tag": best_tag,
        "track": track, "learning_counts": learning_counts,
        "gpu_hours": gpu_hours,
        "canonical": canonical, "integrity": integrity,
        "audit": audit.get("integrity") if audit else None,
        "has_system_monitor": bool(monitor),
        "has_workspace": ws_snapshot is not None,
        "updated_at": build_ts,
    }
    record = {
        "schema_version": schema.SCHEMA_VERSION,
        "index_row": index_row,
        "meta": meta,
        "summary": summary,
        "sessions": sessions,
        "events": events,
        "scores": scores,
        "judgements": {"audit": audit},
        "system_monitor": monitor,
        "learning": learning_actions,
        "invented_tools": invented_cards,
    }
    return record, ws_snapshot


def build_status(record: dict) -> dict:
    events = record.get("events") or []
    return {
        "run_id": record["meta"]["run_id"],
        "state": record["index_row"]["state"],
        "updated_at": record["index_row"]["updated_at"],
        "num_events": len(events),
        "last_event_ts": next((e.get("ts") for e in reversed(events) if e.get("ts")), None),
        "exit_code": record["meta"]["exit_code"],
    }
