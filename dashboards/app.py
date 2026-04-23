"""Modal-hosted FastAPI dashboard for training-gym runs.

Shows all training runs (``_modal_job_type=training``), grouped into
collapsible sections by ``_modal_framework``.  Framework pills in the
toolbar let users toggle which sections are visible.

Architecture:
    A cron job (``refresh_training_metadata``) runs every 5 minutes,
    fetches all Modal apps, checks their tags, and writes training run
    metadata into a ``modal.Dict``.  The ASGI dashboard reads from the
    Dict for instant page loads.

Deploy with:

    uv run modal deploy dashboards/app.py

Modal will print the public URL; open it in a browser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import modal

from modal_training_gym.common import COMMON_TRAINING_GYM_TAGS

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.118.0",
        "modal",
    )
    .add_local_python_source("modal_training_gym", copy=True)
)

app = modal.App(
    "training-gym-dashboard",
    image=image,
    tags={**COMMON_TRAINING_GYM_TAGS, "_modal_framework": "dashboard"},
)

training_runs_dict = modal.Dict.from_name("training-gym-runs", create_if_missing=True)

RUNS_KEY = "runs"


@dataclass
class TrainingMetadata:
    framework: str = "(untagged)"


@dataclass
class TrainingRun:
    app_id: str
    name: str
    description: str
    state: str
    created_at: float
    stopped_at: float | None
    metadata: TrainingMetadata
    tags: dict[str, str] = field(default_factory=dict)


_FETCH_SCRIPT = r"""
import json, sys, modal

result = __import__("subprocess").run(
    ["modal", "app", "list", "--json"],
    capture_output=True, text=True,
)
app_list = json.loads(result.stdout)

seen, runs = set(), []
for info in app_list:
    name = info["Description"]
    if name in seen:
        continue
    seen.add(name)
    try:
        tags = modal.App.lookup(name).get_tags()
    except Exception:
        continue
    if tags.get("_modal_job_type") != "training":
        continue
    runs.append({
        "app_id": info["App ID"],
        "name": name,
        "state": info.get("State", "unknown"),
        "created_at": info.get("Created at", ""),
        "stopped_at": info.get("Stopped at"),
        "framework": tags.get("_modal_framework", "(untagged)"),
        "tags": tags,
    })
json.dump(runs, sys.stdout)
"""


@app.function(
    schedule=modal.Period(minutes=5),
    timeout=120,
)
def refresh_training_metadata():
    """Cron job: runs a subprocess to list apps and fetch tags (avoids
    the container's gRPC event-loop mismatch), then writes to the Dict."""
    import json
    import subprocess
    from datetime import datetime, timezone

    result = subprocess.run(
        ["python", "-c", _FETCH_SCRIPT],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        print(f"Fetch script failed: {result.stderr}")
        return

    raw_runs = json.loads(result.stdout)

    def _parse_ts(s):
        if not s:
            return 0.0
        try:
            dt = datetime.fromisoformat(s)
            return dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp()
        except (ValueError, TypeError):
            return 0.0

    runs: list[dict] = []
    for r in raw_runs:
        meta = TrainingMetadata(framework=r["framework"])
        run = TrainingRun(
            app_id=r["app_id"],
            name=r["name"],
            description=r["name"],
            state=r["state"].title(),
            created_at=_parse_ts(r["created_at"]),
            stopped_at=_parse_ts(r["stopped_at"]) if r["stopped_at"] else None,
            metadata=meta,
            tags=r["tags"],
        )
        runs.append(asdict(run))

    runs.sort(key=lambda r: r["created_at"], reverse=True)
    training_runs_dict[RUNS_KEY] = runs
    print(f"Refreshed: {len(runs)} training run(s)")


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>training-gym dashboard</title>
  <style>
    :root {
      --bg: #0b0d10;
      --panel: #141820;
      --border: #262c36;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #7dd3fc;
      --accent-dim: rgba(125, 211, 252, 0.12);
      --green: #4ade80;
      --yellow: #fbbf24;
      --red: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* ── Top bar ─────────────────────────────────────────────────── */
    .topbar {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 1.25rem 2rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;
      flex-wrap: wrap;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar h1 {
      font-size: 1.25rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .topbar .sep {
      width: 1px;
      height: 1.5rem;
      background: var(--border);
    }
    #status {
      color: var(--muted);
      font-size: 0.85em;
      margin-left: auto;
    }

    /* ── Filter bar ──────────────────────────────────────────────── */
    .filters {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 0.75rem 2rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
      position: sticky;
      top: 60px;
      z-index: 9;
    }
    .filter-label {
      color: var(--muted);
      font-size: 0.8em;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-weight: 600;
      white-space: nowrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.3rem 0.7rem;
      border-radius: 9999px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-size: 0.85em;
      cursor: pointer;
      transition: all 0.15s;
      user-select: none;
    }
    .pill:hover { border-color: var(--accent); color: var(--text); }
    .pill.active {
      background: var(--accent-dim);
      border-color: var(--accent);
      color: var(--accent);
    }
    .pill .pill-count {
      font-size: 0.8em;
      opacity: 0.7;
    }
    .filter-sep {
      width: 1px;
      height: 1.2rem;
      background: var(--border);
      margin: 0 0.25rem;
    }

    /* State pills */
    .pill[data-state="Running"].active,
    .pill[data-state="Deployed"].active { border-color: var(--green); color: var(--green); background: rgba(74, 222, 128, 0.1); }
    .pill[data-state="Stopped"].active { border-color: var(--muted); }
    .pill[data-state="Ephemeral"].active { border-color: var(--yellow); color: var(--yellow); background: rgba(251, 191, 36, 0.1); }

    /* ── Search ───────────────────────────────────────────────────── */
    #search {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      padding: 0.35rem 0.7rem;
      font: inherit;
      font-size: 0.85em;
      width: 14rem;
      outline: none;
    }
    #search:focus { border-color: var(--accent); }
    #search::placeholder { color: var(--muted); }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .btn {
      background: #1f2937;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.35rem 0.85rem;
      border-radius: 6px;
      cursor: pointer;
      font: inherit;
      font-size: 0.85em;
      transition: background 0.15s;
    }
    .btn:hover { background: #374151; }

    /* ── Main content ────────────────────────────────────────────── */
    .main { padding: 1.5rem 2rem 3rem; }

    .framework-section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 1rem;
      overflow: hidden;
    }
    .framework-header {
      padding: 0.65rem 1rem;
      background: #1a1f2a;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      user-select: none;
    }
    .framework-header:hover { background: #1e2430; }
    .framework-left {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .collapse-arrow {
      color: var(--muted);
      font-size: 0.7em;
      transition: transform 0.2s;
    }
    .framework-section.collapsed .collapse-arrow { transform: rotate(-90deg); }
    .framework-section.collapsed .fw-body { display: none; }
    .framework-name { font-weight: 600; color: var(--accent); }
    .framework-count { color: var(--muted); font-size: 0.85em; }
    .framework-stats {
      display: flex;
      gap: 0.75rem;
      font-size: 0.8em;
    }
    .stat-running { color: var(--green); }
    .stat-stopped { color: var(--muted); }

    /* ── Table ────────────────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: 0.55rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    th {
      font-weight: 500;
      color: var(--muted);
      font-size: 0.8em;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      position: sticky;
    }
    tr:last-child td { border-bottom: none; }
    .app-id {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em;
      color: var(--accent);
    }
    .run-name { font-weight: 500; }
    .run-desc { color: var(--muted); font-size: 0.9em; margin-top: 0.15rem; }
    .tag {
      display: inline-block;
      padding: 0.1rem 0.45rem;
      margin: 0.1rem 0.2rem 0.1rem 0;
      background: #1f2937;
      border-radius: 4px;
      font-size: 0.8em;
      color: var(--muted);
    }
    .tag strong { color: var(--text); font-weight: 500; }

    /* State badges */
    .state-badge {
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      font-size: 0.8em;
      font-weight: 500;
    }
    .state-Running, .state-Deployed { background: rgba(74,222,128,0.12); color: var(--green); }
    .state-Stopped, .state-Stopping { background: rgba(156,163,175,0.12); color: var(--muted); }
    .state-Ephemeral { background: rgba(251,191,36,0.12); color: var(--yellow); }
    .state-Initializing { background: rgba(125,211,252,0.12); color: var(--accent); }
    .state-Errored { background: rgba(248,113,113,0.12); color: var(--red); }

    .empty {
      color: var(--muted);
      padding: 3rem;
      text-align: center;
      font-size: 0.95em;
    }

    /* ── Duration ─────────────────────────────────────────────────── */
    .duration { color: var(--muted); font-size: 0.85em; }
  </style>
</head>
<body>

  <!-- Top bar -->
  <header class="topbar">
    <h1>training-gym</h1>
    <div class="sep"></div>
    <button class="btn" id="refresh">Refresh</button>
    <span id="status"></span>
  </header>

  <!-- Filters -->
  <nav class="filters" id="filter-bar">
    <span class="filter-label">Framework</span>
    <div id="fw-pills"></div>
    <div class="filter-sep"></div>
    <span class="filter-label">State</span>
    <div id="state-pills"></div>
    <div class="filter-sep"></div>
    <input id="search" type="text" placeholder="Search by name or app id..." />
  </nav>

  <!-- Runs -->
  <div class="main" id="content"></div>

  <script>
    const CORE_TAGS = new Set(["training", "source", "_modal_framework", "_modal_job_type"]);

    let allRuns = [];
    let activeFrameworks = new Set();
    let activeStates = new Set();

    function fmtDate(ts) {
      if (!ts) return "—";
      return new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19) + " UTC";
    }

    function fmtDuration(start, end) {
      if (!start) return "—";
      const endTs = end || Date.now() / 1000;
      let secs = Math.floor(endTs - start);
      if (secs < 0) secs = 0;
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      const s = secs % 60;
      if (h > 0) return `${h}h ${m}m`;
      if (m > 0) return `${m}m ${s}s`;
      return `${s}s`;
    }

    function renderExtraTags(tags) {
      const extras = Object.entries(tags).filter(([k]) => !CORE_TAGS.has(k));
      if (!extras.length) return "—";
      return extras
        .map(([k, v]) => `<span class="tag"><strong>${esc(k)}</strong>=${esc(v)}</span>`)
        .join("");
    }

    function esc(s) {
      const d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }

    function buildPills() {
      const fwCounts = {};
      const stateCounts = {};
      for (const r of allRuns) {
        const fw = (r.metadata && r.metadata.framework) || r.framework || "(untagged)";
        r._fw = fw;
        fwCounts[fw] = (fwCounts[fw] || 0) + 1;
        stateCounts[r.state] = (stateCounts[r.state] || 0) + 1;
      }

      const fwContainer = document.getElementById("fw-pills");
      const frameworks = Object.keys(fwCounts).sort();

      if (activeFrameworks.size === 0) {
        activeFrameworks = new Set(frameworks);
      }

      fwContainer.innerHTML = `<button class="pill active" data-fw="__all__">All <span class="pill-count">${allRuns.length}</span></button>` +
        frameworks.map(fw => {
          const active = activeFrameworks.has(fw) ? " active" : "";
          return `<button class="pill${active}" data-fw="${esc(fw)}">${esc(fw)} <span class="pill-count">${fwCounts[fw]}</span></button>`;
        }).join("");

      fwContainer.querySelectorAll(".pill").forEach(btn => {
        btn.addEventListener("click", () => {
          const fw = btn.dataset.fw;
          if (fw === "__all__") {
            const allActive = activeFrameworks.size === frameworks.length;
            activeFrameworks = allActive ? new Set() : new Set(frameworks);
          } else {
            if (activeFrameworks.has(fw)) activeFrameworks.delete(fw);
            else activeFrameworks.add(fw);
          }
          updatePillStates();
          renderRuns();
        });
      });

      const stateContainer = document.getElementById("state-pills");
      const states = Object.keys(stateCounts).sort();
      if (activeStates.size === 0) {
        activeStates = new Set(states);
      }
      stateContainer.innerHTML = states.map(st => {
        const active = activeStates.has(st) ? " active" : "";
        return `<button class="pill${active}" data-state="${esc(st)}">${esc(st)} <span class="pill-count">${stateCounts[st]}</span></button>`;
      }).join("");

      stateContainer.querySelectorAll(".pill").forEach(btn => {
        btn.addEventListener("click", () => {
          const st = btn.dataset.state;
          if (activeStates.has(st)) activeStates.delete(st);
          else activeStates.add(st);
          updatePillStates();
          renderRuns();
        });
      });
    }

    function updatePillStates() {
      const fwCounts = {};
      for (const r of allRuns) fwCounts[r._fw] = (fwCounts[r._fw] || 0) + 1;
      const allFws = Object.keys(fwCounts);

      document.querySelectorAll("#fw-pills .pill").forEach(btn => {
        const fw = btn.dataset.fw;
        if (fw === "__all__") {
          btn.classList.toggle("active", activeFrameworks.size === allFws.length);
        } else {
          btn.classList.toggle("active", activeFrameworks.has(fw));
        }
      });
      document.querySelectorAll("#state-pills .pill").forEach(btn => {
        btn.classList.toggle("active", activeStates.has(btn.dataset.state));
      });
    }

    function getFilteredRuns() {
      const q = document.getElementById("search").value.toLowerCase().trim();
      return allRuns.filter(r => {
        if (!activeFrameworks.has(r._fw)) return false;
        if (!activeStates.has(r.state)) return false;
        if (q && !r.name.toLowerCase().includes(q) && !r.app_id.toLowerCase().includes(q)) return false;
        return true;
      });
    }

    function renderRuns() {
      const content = document.getElementById("content");
      const runs = getFilteredRuns();
      const status = document.getElementById("status");

      if (!allRuns.length) {
        content.innerHTML = '<div class="empty">No training runs found. The cron job populates data every 5 minutes &mdash; if you just deployed, wait for the first refresh.</div>';
        status.textContent = "0 runs";
        return;
      }
      if (!runs.length) {
        content.innerHTML = '<div class="empty">No runs match the current filters.</div>';
        status.textContent = `0 of ${allRuns.length} runs shown`;
        return;
      }

      const groups = {};
      for (const r of runs) {
        (groups[r._fw] ||= []).push(r);
      }
      const frameworks = Object.keys(groups).sort();

      content.innerHTML = frameworks.map(fw => {
        const fwRuns = groups[fw];
        const running = fwRuns.filter(r => r.state === "Running" || r.state === "Deployed").length;
        const statsHtml = running > 0
          ? `<span class="stat-running">${running} active</span>`
          : `<span class="stat-stopped">none active</span>`;

        const rows = fwRuns.map(r => `
          <tr>
            <td class="app-id">${esc(r.app_id)}</td>
            <td>
              <div class="run-name">${esc(r.name || "—")}</div>
              ${r.description ? `<div class="run-desc">${esc(r.description)}</div>` : ""}
            </td>
            <td><span class="state-badge state-${r.state}">${r.state}</span></td>
            <td>${fmtDate(r.created_at)}</td>
            <td class="duration">${fmtDuration(r.created_at, r.stopped_at)}</td>
            <td>${renderExtraTags(r.tags)}</td>
          </tr>`).join("");

        return `
          <section class="framework-section" id="fw-${esc(fw)}">
            <header class="framework-header" onclick="this.parentElement.classList.toggle('collapsed')">
              <div class="framework-left">
                <span class="collapse-arrow">&#9660;</span>
                <span class="framework-name">${esc(fw)}</span>
                <span class="framework-count">${fwRuns.length} run${fwRuns.length === 1 ? "" : "s"}</span>
              </div>
              <div class="framework-stats">${statsHtml}</div>
            </header>
            <div class="fw-body">
              <table>
                <thead>
                  <tr>
                    <th>App ID</th>
                    <th>Name</th>
                    <th>State</th>
                    <th>Created</th>
                    <th>Duration</th>
                    <th>Tags</th>
                  </tr>
                </thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          </section>`;
      }).join("");

      status.textContent = runs.length === allRuns.length
        ? `${allRuns.length} run${allRuns.length === 1 ? "" : "s"} across ${frameworks.length} framework${frameworks.length === 1 ? "" : "s"}`
        : `${runs.length} of ${allRuns.length} runs across ${frameworks.length} framework${frameworks.length === 1 ? "" : "s"}`;
    }

    async function load() {
      const status = document.getElementById("status");
      status.textContent = "loading…";
      try {
        const res = await fetch("/api/runs");
        const data = await res.json();
        allRuns = data.runs || [];
        activeFrameworks = new Set();
        activeStates = new Set();
        buildPills();
        renderRuns();
      } catch (e) {
        document.getElementById("content").innerHTML = `<div class="empty">Failed to load: ${e}</div>`;
        status.textContent = "error";
      }
    }

    document.getElementById("refresh").addEventListener("click", load);
    document.getElementById("search").addEventListener("input", renderRuns);
    load();
  </script>
</body>
</html>
"""


@app.function()
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    web = FastAPI()

    @web.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @web.get("/api/runs")
    async def api_runs() -> JSONResponse:
        try:
            runs = await training_runs_dict.get.aio(RUNS_KEY)
        except KeyError:
            runs = []
        return JSONResponse({"runs": runs if runs is not None else []})

    return web
