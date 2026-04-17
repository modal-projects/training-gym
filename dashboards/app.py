"""Modal-hosted FastAPI dashboard for training-gym runs.

Lists every Modal app tagged `source=training-gym`, groups them by the
`framework` tag, and shows app id, state, created-at, and any custom tags
per run.

Deploy with:

    uv run modal deploy dashboards/app.py

Modal will print the public URL; open it in a browser.
"""

from __future__ import annotations

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
    tags={**COMMON_TRAINING_GYM_TAGS, "framework": "dashboard"},
)


def _fetch_runs_blocking() -> dict:
    """Synchronous core: uses Modal's public (synchronicity-wrapped) Client.

    Must be called via `asyncio.to_thread` from an async context — the
    `_Client` internal async API would fail with "RPC made outside of task
    context" because FastAPI's event loop isn't synchronicity-managed.
    """
    from modal.client import Client
    from modal_proto import api_pb2

    client = Client.from_env()
    apps_resp = client.stub.AppList(api_pb2.AppListRequest())

    state_enum = api_pb2.AppState if hasattr(api_pb2, "AppState") else None

    runs: list[dict] = []
    for app_item in apps_resp.apps:
        tags_resp = client.stub.AppGetTags(
            api_pb2.AppGetTagsRequest(app_id=app_item.app_id)
        )
        tags = {t.key: t.value for t in tags_resp.tags}
        if tags.get("source") != "training-gym":
            continue
        state_name = (
            state_enum.Name(app_item.state) if state_enum is not None
            else str(app_item.state)
        )
        runs.append(
            {
                "app_id": app_item.app_id,
                "name": app_item.name,
                "description": app_item.description,
                "state": state_name.replace("APP_STATE_", "").title(),
                "created_at": float(app_item.created_at),
                "stopped_at": float(app_item.stopped_at) if app_item.stopped_at else None,
                "framework": tags.get("framework", "(untagged)"),
                "tags": tags,
            }
        )
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return {"runs": runs}


async def _fetch_runs() -> dict:
    import asyncio

    return await asyncio.to_thread(_fetch_runs_blocking)


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
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2rem;
    }
    h1 { margin: 0 0 0.5rem; font-size: 1.5rem; }
    .subtitle { color: var(--muted); margin-bottom: 2rem; }
    .framework-section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 1.5rem;
      overflow: hidden;
    }
    .framework-header {
      padding: 0.75rem 1rem;
      background: #1a1f2a;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .framework-name { font-weight: 600; color: var(--accent); }
    .framework-count { color: var(--muted); font-size: 0.9em; }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: 0.6rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    th {
      font-weight: 500;
      color: var(--muted);
      font-size: 0.85em;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    tr:last-child td { border-bottom: none; }
    .app-id {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em;
      color: var(--accent);
    }
    .tag {
      display: inline-block;
      padding: 0.1rem 0.4rem;
      margin: 0.1rem 0.2rem 0.1rem 0;
      background: #1f2937;
      border-radius: 4px;
      font-size: 0.8em;
      color: var(--muted);
    }
    .tag strong { color: var(--text); font-weight: 500; }
    .state-Deployed, .state-Running { color: #4ade80; }
    .state-Stopped, .state-Stopping { color: var(--muted); }
    .state-Ephemeral { color: #fbbf24; }
    .state-Initializing { color: #7dd3fc; }
    .empty { color: var(--muted); padding: 3rem; text-align: center; }
    button {
      background: #1f2937;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.5rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      font: inherit;
    }
    button:hover { background: #374151; }
    .toolbar { margin-bottom: 1rem; display: flex; gap: 1rem; align-items: center; }
  </style>
</head>
<body>
  <h1>training-gym runs</h1>
  <div class="subtitle">Every Modal app tagged <code>source=training-gym</code>, grouped by framework.</div>
  <div class="toolbar">
    <button id="refresh">↻ refresh</button>
    <span id="status" class="subtitle"></span>
  </div>
  <div id="content"></div>

  <script>
    const CORE_TAGS = new Set(["training", "source", "framework"]);

    function fmtDate(ts) {
      if (!ts) return "—";
      return new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19) + " UTC";
    }

    function renderExtraTags(tags) {
      const extras = Object.entries(tags).filter(([k]) => !CORE_TAGS.has(k));
      if (!extras.length) return "—";
      return extras
        .map(([k, v]) => `<span class="tag"><strong>${k}</strong>=${v}</span>`)
        .join("");
    }

    async function load() {
      const status = document.getElementById("status");
      const content = document.getElementById("content");
      status.textContent = "loading…";
      try {
        const res = await fetch("/api/runs");
        const data = await res.json();
        const runs = data.runs || [];
        if (!runs.length) {
          content.innerHTML = '<div class="empty">No training-gym apps found in this environment.</div>';
          status.textContent = "0 runs";
          return;
        }
        // Group by framework.
        const groups = {};
        for (const r of runs) {
          (groups[r.framework] ||= []).push(r);
        }
        const frameworks = Object.keys(groups).sort();
        content.innerHTML = frameworks
          .map((fw) => {
            const rows = groups[fw]
              .map(
                (r) => `
                  <tr>
                    <td class="app-id">${r.app_id}</td>
                    <td>${r.name || "—"}${r.description ? `<div class="subtitle">${r.description}</div>` : ""}</td>
                    <td class="state-${r.state}">${r.state}</td>
                    <td>${fmtDate(r.created_at)}</td>
                    <td>${fmtDate(r.stopped_at)}</td>
                    <td>${renderExtraTags(r.tags)}</td>
                  </tr>`
              )
              .join("");
            return `
              <section class="framework-section">
                <header class="framework-header">
                  <span class="framework-name">${fw}</span>
                  <span class="framework-count">${groups[fw].length} run${groups[fw].length === 1 ? "" : "s"}</span>
                </header>
                <table>
                  <thead>
                    <tr>
                      <th>App ID</th>
                      <th>Name / Description</th>
                      <th>State</th>
                      <th>Created</th>
                      <th>Stopped</th>
                      <th>Custom tags</th>
                    </tr>
                  </thead>
                  <tbody>${rows}</tbody>
                </table>
              </section>`;
          })
          .join("");
        status.textContent = `${runs.length} run${runs.length === 1 ? "" : "s"} across ${frameworks.length} framework${frameworks.length === 1 ? "" : "s"}`;
      } catch (e) {
        content.innerHTML = `<div class="empty">Failed to load: ${e}</div>`;
        status.textContent = "error";
      }
    }

    document.getElementById("refresh").addEventListener("click", load);
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
        return JSONResponse(await _fetch_runs())

    return web
