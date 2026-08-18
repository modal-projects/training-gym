"""Observatory data contracts.

Field-level truth for everything that crosses a boundary: normalizer -> volume
-> web app -> frontend. Pure stdlib (TypedDicts + helpers); nothing here may
import modal, fastapi, or anything outside the standard library.

Shape lineage: posttrainbench.com viewer records (index_row / meta / summary /
sessions / events / system_monitor), extended with Learning Agent's scores + judgements
+ learning timeline.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

SCHEMA_VERSION = 1

# Run lifecycle states (status.json "state" and index_row "state").
STATE_RUNNING = "running"    # watcher is live-syncing, run not finished
STATE_FINISHED = "finished"  # solve_status.txt present / final ingest done
STATE_ERROR = "error"        # finished with nonzero exit or ingest saw a crash
STATE_STALE = "stale"        # was running, but no update for > STALE_AFTER_S
STATES = (STATE_RUNNING, STATE_FINISHED, STATE_ERROR, STATE_STALE)
STALE_AFTER_S = 300

# Trace formats the normalizer understands (meta.trace_format).
TRACE_CLAUDE = "claude-stream-json"   # claude*, glm5, qwen3max scaffolds
TRACE_CODEX = "codex-events"          # codex* scaffolds
TRACE_GEMINI = "gemini-stream-json"
TRACE_OPENCODE = "opencode-json"
TRACE_UNKNOWN = "unknown"

# Volume layout — single source of truth for both CLI and app.
RUNS_PREFIX = "runs"                  # <volume-root>/runs/<run_id>/
RECORD_FILE = "record.json"
WORKSPACE_FILE = "workspace.json"
STATUS_FILE = "status.json"
RAW_DIR = "raw"

# Workspace snapshot limits.
WS_INLINE_MAX_BYTES = 64 * 1024       # per-file inline cap
WS_TOTAL_INLINE_MAX_BYTES = 24 * 1024 * 1024
WS_EXCLUDE_DIRS = {".git", "corpus", "__pycache__", ".venv", ".venv-rl",
                   "node_modules", "_runs"}


class IndexRow(TypedDict, total=False):
    run_id: str
    kind: str                  # "agent_run" (v1's only kind)
    state: str                 # STATE_*
    task: str                  # dspy | openclaw | fav2 | ...
    scaffold: str              # claude_reprompt, codex_non_api_high, ...
    agent_model: Optional[str] # model actually seen in the trace
    base_model: Optional[str]  # student model being trained
    trace_format: str
    time_budget_h: Optional[float]
    launched_at: Optional[str]   # ISO8601
    finished_at: Optional[str]
    duration_s: Optional[float]
    num_turns: Optional[int]
    num_events: int
    session_count: int
    total_cost_usd: Optional[float]
    best_dev_score: Optional[float]   # max over scores.results dev means; null-safe
    best_dev_ci: Optional[list[float]]
    best_tag: Optional[str]
    track: Optional[str]      # easy | medium | hard | None (run_meta.json, pre-track runs)
    learning_counts: Optional[dict]  # {"data","train","eval","evolve","invented_tools": int}
                                      # tallied from RunRecord.learning; None if not computed
    gpu_hours: Optional[float]  # sum(seconds x n_gpus)/3600 over runs/GPU_LOG.jsonl
    canonical: Optional[bool]  # False if ANY contributing result is non-canonical
    integrity: Optional[str]   # "OK" | "DIRTY" | None
    audit: Optional[str]       # "CLEAN" | "CONTAMINATED" | None
    has_system_monitor: bool
    has_workspace: bool
    updated_at: str            # ISO8601 of last ingest


class Meta(TypedDict, total=False):
    run_id: str
    run_dir: str               # local origin path (informational)
    scaffold: str
    task: str
    base_model: Optional[str]
    trace_format: str
    time_budget_h: Optional[float]
    launched_at: Optional[str]
    finished_at: Optional[str]
    exit_code: Optional[int]   # from solve_status.txt "exit=N"
    track: Optional[str]       # easy | medium | hard | None (run_meta.json, pre-track runs)
    build_ts: str              # when this record was built
    schema_version: int


class Summary(TypedDict, total=False):
    agent_models: list[str]
    tools_offered: list[str]
    permission_mode: Optional[str]
    cwd: Optional[str]
    num_turns: Optional[int]
    duration_ms: Optional[int]
    total_cost_usd: Optional[float]
    usage_total: dict[str, int]   # input_tokens, output_tokens, cache_*_tokens
    stop_reasons: list[str]
    final_result_text: Optional[str]
    session_count: int
    session_ids: list[str]


class Session(TypedDict, total=False):
    session_idx: int
    session_id: str
    ts_start: Optional[str]
    model: Optional[str]
    cwd: Optional[str]
    permission_mode: Optional[str]
    tools: list[str]


class Event(TypedDict, total=False):
    """One normalized trace event.

    Content is hoisted to event-level `blocks` (posttrainbench-compatible);
    every trace format is mapped into these four block shapes so the frontend
    has exactly one rendering path:

      {"type": "thinking", "thinking": str}
      {"type": "text", "text": str}
      {"type": "tool_use", "id": str, "name": str, "input": dict}
      {"type": "tool_result", "tool_use_id": str, "content": str|list,
       "is_error": bool}

    Learning Agent trace.jsonl lines carry no timestamps; `ts` comes from the watcher's
    arrival-time sidecar (raw/line_ts.jsonl: {"line": <1-based>, "ts": ISO})
    when present, else None.
    """
    i: int                     # 0-based position
    ts: Optional[str]          # ISO8601 or None
    type: str                  # assistant | user | system | result
    subtype: Optional[str]     # e.g. "init" for system, "success" for result
    session_id: Optional[str]
    session_idx: Optional[int]
    parent_tool_use_id: Optional[str]
    turn: Optional[int]        # 1-based; set on assistant events, see DESIGN.md
    blocks: list[dict[str, Any]]      # assistant/user events
    usage: Optional[dict[str, Any]]   # per-message usage when present
    model: Optional[str]
    uuid: Optional[str]
    # result events only:
    duration_ms: Optional[int]
    num_turns: Optional[int]
    total_cost_usd: Optional[float]
    stop_reason: Optional[str]
    result: Optional[str]      # final result text
    # system/unknown events keep a trimmed original:
    raw: Optional[dict[str, Any]]


class ResultEntry(TypedDict, total=False):
    """One runs/<tag>/budget_<b>/results_<split>.json, joined with eval_meta."""
    tag: str
    split: str                 # dev | test
    budget: int
    mean: Optional[float]      # None when all questions failed — never 0
    bootstrap_ci95: Optional[list[float]]
    n: Optional[int]
    n_failed: Optional[int]
    all_failed: Optional[bool]
    secondary_mean: Optional[float]
    canonical: Optional[bool]
    integrity: Optional[str]
    provenance: dict[str, Any]  # judge_model, judge_backend, *_sha, seed, ...
    per_question: dict[str, Any]  # {qid: {claim_score|None, failed, verdicts,
                                  #        votes?, secondary?, tool_calls?,
                                  #        completion_tokens?, answer?}}
    tool_calls_avg: Optional[float]  # from eval_meta join; search-collapse signal


class Scores(TypedDict, total=False):
    checkpoints: list[dict[str, Any]]   # CHECKPOINTS.jsonl rows, verbatim (pre-2026-08-11
                                        # runs; newer runs synthesize these from the
                                        # learning log's checkpoint-kind entries)
    learning_log: list[dict[str, Any]]  # LEARNING_LOG.jsonl rows, verbatim — the agent's
                                        # own experiment record: {ts, kind, what, why,
                                        # result/dev_score, artifacts}
    gpu_log: list[dict[str, Any]]       # GPU_LOG.jsonl rows, verbatim
    leaderboard: list[dict[str, Any]]   # LEADERBOARD.jsonl rows, verbatim
    results: list[ResultEntry]


class Judgements(TypedDict, total=False):
    audit: Optional[dict[str, Any]]     # audit.json verbatim


class MonitorSample(TypedDict, total=False):
    ts: str
    gpu: Optional[dict[str, Any]]  # {id, util_pct, mem_used_mib, mem_total_mib,
                                   #  temp_c, power_w} or None on CPU-only hosts
    gpus: Optional[list[dict[str, Any]]]  # multi-GPU hosts; gpu = gpus[0]
    cpu_load_1m: Optional[float]
    cpu_load_5m: Optional[float]
    cpu_pct: Optional[float]
    mem_used_gib: Optional[float]
    mem_total_gib: Optional[float]
    source: Optional[str]          # "local-watcher" | "modal-sampler"


class LearningAction(TypedDict, total=False):
    """One classified learning-tool invocation on the "learning timeline": Learning Agent's
    thesis is "learning as tools" (the agent generates training data, trains,
    evaluates, and runs evolve recipes as ordinary shell commands), and this is
    the post-hoc classification of one such command found in a trace tool_use
    block. Produced in two passes: observatory/normalize/learning.py's
    `classify_command` does the pure string classification (kind, tool,
    provenance, command, args); the collector (observatory/normalize/collect.py)
    fills in event_i/ts/nth_use once it knows which event the command came from
    and how it orders against the run's other actions.

    kind: "data"   — toolbox/data_toolbox/gen/*.py (a corpus-generation script)
          "train"  — bench.py train / pipeline/train.py (SFT), or
                      bench.py rl / pipeline/rl.py (RL)
          "eval"   — toolbox/eval_tool/rubric_eval.py, or bench.py score
          "evolve" — toolbox/evolve/run_recipe.py
          "tool"   — an invented tool: an executed script that matched none of
                      the four kinds above (see provenance)

    tool: the registry's short name for the matched seed tool (e.g. a gen/
          stem like "paraphrase", or "sft" | "rl" | "rubric_eval" |
          "bench_score" | "run_recipe"); for kind "tool" (invented), the
          script path/token itself.

    provenance: "seed"     — command matched the seed-tool registry
                "invented" — an executed script absent from the run's seed
                             manifest (git ls-tree of the checkout at prepare
                             time) but present in the workspace snapshot
                "unknown"  — no seed manifest available to tell seed scripts
                             from invented ones apart from a registry match
                             (old runs that predate seed_manifest.txt)
    """
    event_i: int                # index into RunRecord.events
    ts: Optional[str]           # ISO8601 or None, from the owning event
    kind: str                   # data | train | eval | evolve | tool
    tool: str
    provenance: str             # seed | invented | unknown
    command: str                # the full shell command string this was parsed from
    args: dict                  # best-effort surfaced flags, e.g. {"tag": "t1"}
    nth_use: int                # 1-based occurrence count of this tool in the run


class InventedToolCard(TypedDict, total=False):
    """A tool this run created (TOOL_SPEC folder form): its card + README,
    verbatim from the workspace snapshot, for dashboard rendering with
    attribution. Detected structurally — a toolbox/**/tool.yaml whose sibling
    run.py is absent from the seed manifest — executed or not."""
    path: str          # workspace-relative tool dir, e.g. toolbox/data_toolbox/gen/x
    tool_yaml: str     # raw card text
    readme: str        # raw README text ("" if missing)


class RunRecord(TypedDict, total=False):
    schema_version: int
    index_row: IndexRow
    meta: Meta
    summary: Summary
    sessions: list[Session]
    events: list[Event]
    scores: Scores
    judgements: Judgements
    system_monitor: list[MonitorSample]
    learning: list[LearningAction]
    invented_tools: list[InventedToolCard]


class Status(TypedDict, total=False):
    run_id: str
    state: str
    updated_at: str
    num_events: int
    last_event_ts: Optional[str]
    exit_code: Optional[int]


class WorkspaceFile(TypedDict, total=False):
    path: str        # workspace-relative, posix
    size: int
    inline: bool
    content: Optional[str]   # present iff inline
    truncated: bool


class WorkspaceSnapshot(TypedDict, total=False):
    built_at: str
    root: str                # local origin path (informational)
    total_files: int
    total_bytes: int
    inlined_files: int
    files: list[WorkspaceFile]


def run_paths(run_id: str) -> dict[str, str]:
    """Volume-relative paths for one run — use everywhere, never hand-build."""
    base = f"{RUNS_PREFIX}/{run_id}"
    return {
        "base": base,
        "record": f"{base}/{RECORD_FILE}",
        "workspace": f"{base}/{WORKSPACE_FILE}",
        "status": f"{base}/{STATUS_FILE}",
        "raw": f"{base}/{RAW_DIR}",
    }
