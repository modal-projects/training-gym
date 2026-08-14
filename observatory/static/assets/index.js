// Index page: stats strip, task×scaffold heatmap, filterable runs table. Polls /api/runs.

const POLL_MS = 15000;
let runs = [];

const $ = (sel) => document.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) if (c != null) n.append(c.nodeType ? c : String(c));
  return n;
}

// ---- formatting ----

function fmtScore(x) {
  if (x == null || !Number.isFinite(x)) return "—";
  return (Math.round(x * 1000) / 1000).toString();
}

function fmtCost(x) {
  return x == null ? "—" : `$${x.toFixed(2)}`;
}

function fmtDuration(s) {
  if (s == null || !Number.isFinite(s)) return "—";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function relTime(iso) {
  const t = Date.parse(iso ?? "");
  if (!Number.isFinite(t)) return "—";
  const d = (Date.now() - t) / 1000;
  if (d < 0) return "future";
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

function stateChip(state) {
  const s = state || "unknown";
  return el("span", { class: `chip state-${s}` }, el("span", { class: "dot" }), s);
}

function badgeChips(r) {
  const out = [];
  if (r.canonical === false) out.push(el("span", { class: "chip b-amber", title: "at least one contributing result is non-canonical" }, "non-canonical"));
  if (r.integrity === "DIRTY") out.push(el("span", { class: "chip b-red" }, "DIRTY"));
  if (r.audit === "CONTAMINATED") out.push(el("span", { class: "chip b-red" }, "CONTAMINATED"));
  else if (r.audit === "CLEAN") out.push(el("span", { class: "chip b-green" }, "CLEAN"));
  return out;
}

// medium/hard tracks: the agent authored its own dev gold, so dev scores are
// self-measured and not comparable across runs — badge wherever one renders.
function selfReportedChip(track) {
  if (track !== "medium" && track !== "hard") return null;
  return el("span", { class: "chip b-amber", title: "medium/hard track — the agent authored its own dev gold; this score is self-measured and not comparable across runs" }, "self-reported");
}

// Compact learning_counts chips ({data,train,eval,evolve,invented_tools} or
// null, see schema.IndexRow.learning_counts): one muted chip per nonzero
// count, short labels; invented tools get the same amber "★" treatment as
// the run page's Learning tab. null or all-zero -> no chips (nothing to show
// for runs that predate the learning timeline, or that used no learning tools).
const LEARNING_KIND_TITLES = {
  data: "data-generation actions", train: "training launches (SFT/RL/OPSD)",
  eval: "eval launches (rubric_eval / bench.py score)", evolve: "self-evolve runs",
  harness: "task-agent harness actions", infra: "inference/serving actions",
};

function learningChips(counts) {
  if (!counts) return null;
  const chips = [];
  for (const k of ["data", "train", "eval", "evolve", "harness", "infra"]) {
    if (counts[k]) chips.push(el("span", { class: "chip", title: LEARNING_KIND_TITLES[k] }, `${counts[k]} ${k}`));
  }
  if (counts.invented_tools) {
    chips.push(el("span", { class: "chip b-amber",
      title: "invented tools: scripts run by the agent that aren't in its seed-tool registry or manifest" },
      `${counts.invented_tools} invented★`));
  }
  return chips.length ? el("span", { class: "badges" }, chips) : null;
}

// ---- data ----

async function load() {
  try {
    const res = await fetch("/api/runs");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    runs = await res.json();
    if (!Array.isArray(runs)) runs = [];
    $("#error").hidden = true;
  } catch (e) {
    $("#error").hidden = false;
    $("#error").textContent = `failed to load /api/runs: ${e.message}`;
    return;
  }
  runs.sort((a, b) => String(b.updated_at ?? "").localeCompare(String(a.updated_at ?? "")));
  render();
}

// ---- render ----

function render() {
  renderStats();
  renderHeatmap();
  renderTaskOptions();
  renderTable();
}

function renderStats() {
  const tasks = new Set(runs.map((r) => r.task).filter(Boolean));
  const scaffolds = new Set(runs.map((r) => r.scaffold).filter(Boolean));
  const live = runs.filter((r) => r.state === "running").length;
  $("#stats").replaceChildren(
    el("div", { class: "stat" }, el("div", { class: "n num", text: String(runs.length) }), el("div", { class: "lbl", text: "runs" })),
    el("div", { class: "stat" }, el("div", { class: "n num", text: String(tasks.size) }), el("div", { class: "lbl", text: "tasks" })),
    el("div", { class: "stat" }, el("div", { class: "n num", text: String(scaffolds.size) }), el("div", { class: "lbl", text: "scaffolds" })),
    el("div", { class: "stat live" }, el("div", { class: "n num", text: String(live) }), el("div", { class: "lbl", text: "live now" })),
  );
}

function renderHeatmap() {
  const box = $("#heatmap");
  const tasks = [...new Set(runs.map((r) => r.task).filter(Boolean))].sort();
  const scaffolds = [...new Set(runs.map((r) => r.scaffold).filter(Boolean))].sort();
  if (!tasks.length || !scaffolds.length) {
    box.replaceChildren(el("div", { class: "empty", text: "no runs yet" }));
    return;
  }
  // best finished dev score per (task, scaffold); the winning run's track rides
  // along so a self-measured medium/hard score is marked, never silently mixed in.
  const best = new Map(); // "task|scaffold" -> {score, self}
  for (const r of runs) {
    if (r.state !== "finished" || r.best_dev_score == null) continue;
    const k = `${r.task}|${r.scaffold}`;
    if (!best.has(k) || r.best_dev_score > best.get(k).score)
      best.set(k, { score: r.best_dev_score, self: r.track === "medium" || r.track === "hard" });
  }
  const table = el("table", { class: "heatmap" });
  table.append(el("tr", {}, el("th"), scaffolds.map((s) => el("th", { class: "col", title: s }, s))));
  for (const t of tasks) {
    const rowVals = scaffolds.map((s) => best.get(`${t}|${s}`)?.score).filter((v) => v != null);
    const rowMax = rowVals.length ? Math.max(...rowVals, 1e-9) : 1;
    const tr = el("tr", {}, el("th", { text: t }));
    for (const s of scaffolds) {
      const b = best.get(`${t}|${s}`);
      const v = b?.score;
      const td = el("td", { class: v == null ? "cell blank" : "cell" }, v == null ? "·" : fmtScore(v));
      if (v != null) {
        if (b.self) td.append(el("span", { class: "warn-ico", title: "self-reported (agent-authored gold)" }, "*"));
        const a = 0.08 + 0.62 * Math.max(0, Math.min(1, v / rowMax));
        td.style.background = `rgba(var(--accent-rgb), ${a.toFixed(3)})`;
        td.title = `${t} × ${s} — best dev ${fmtScore(v)}${b.self ? " · self-reported (agent-authored gold)" : ""} (click to filter)`;
        td.addEventListener("click", () => {
          $("#filter-task").value = t;
          $("#filter-text").value = s;
          renderTable();
        });
      }
      tr.append(td);
    }
    table.append(tr);
  }
  box.classList.remove("empty");
  box.replaceChildren(table);
}

function renderTaskOptions() {
  const sel = $("#filter-task");
  const keep = sel.value;
  const tasks = [...new Set(runs.map((r) => r.task).filter(Boolean))].sort();
  sel.replaceChildren(el("option", { value: "" }, "task: all"), ...tasks.map((t) => el("option", { value: t }, t)));
  if (tasks.includes(keep)) sel.value = keep;
}

function filtered() {
  const q = $("#filter-text").value.trim().toLowerCase();
  const st = $("#filter-state").value;
  const task = $("#filter-task").value;
  const track = $("#filter-track").value;
  return runs.filter((r) => {
    // default view hides stale (dead heartbeats): they are audit relics, not
    // active work — pick "stale" or "everything" in the filter to see them
    if (!st && r.state === "stale") return false;
    if (st && st !== "everything" && r.state !== st) return false;
    if (task && r.task !== task) return false;
    if (track && r.track !== track) return false;
    if (q) {
      const hay = [r.run_id, r.task, r.scaffold, r.track, r.agent_model, r.base_model, r.best_tag, r.state]
        .filter(Boolean).join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function renderTable() {
  const rows = filtered();
  $("#run-count").textContent = `${rows.length}/${runs.length}`;
  $("#table-empty").hidden = rows.length > 0;
  const tbody = $("#runs-tbody");
  tbody.replaceChildren(...rows.map((r) => {
    const ci = Array.isArray(r.best_dev_ci) && r.best_dev_ci.length === 2
      ? el("span", { class: "faint" }, ` [${fmtScore(r.best_dev_ci[0])}, ${fmtScore(r.best_dev_ci[1])}]`) : null;
    const selfChip = selfReportedChip(r.track);
    // official test score (leaderboard overlay): score + margin vs untrained base
    const margin = typeof r.test_margin === "number"
      ? el("span", { class: "faint", title: "margin vs untrained base, same judge" },
           ` (${r.test_margin >= 0 ? "+" : ""}${fmtScore(r.test_margin)})`) : null;
    const testTitle = r.test_judge ? `judge: ${r.test_judge}${r.test_canonical === false ? " (canonical:false)" : ""}` : "";
    return el("tr", {},
      el("td", {}, el("a", { href: `/run?id=${encodeURIComponent(r.run_id ?? "")}` }, r.run_id ?? "—")),
      el("td", {}, stateChip(r.state)),
      el("td", { text: r.task ?? "—" }),
      el("td", { text: r.scaffold ?? "—" }),
      el("td", { text: r.track ?? "—" }),
      el("td", { text: r.agent_model ?? "—" }),
      el("td", { class: "r num" }, fmtScore(r.best_dev_score), ci, selfChip ? " " : null, selfChip),
      el("td", { class: "r num", title: testTitle }, fmtScore(r.test_score), margin),
      el("td", {}, learningChips(r.learning_counts)),
      el("td", { text: r.best_tag ?? "—" }),
      el("td", {}, el("span", { class: "badges" }, badgeChips(r))),
      el("td", {
        class: "r num",
        title: r.gpu_hours_metered != null ? "metered from Modal sandbox history" : "",
        text: r.gpu_hours_metered != null
          ? String(Math.ceil(r.gpu_hours_metered))
          : (r.gpu_hours == null ? "—" : String(Math.ceil(r.gpu_hours))),
      }),
      el("td", { class: "r num", text: fmtCost(r.total_cost_usd) }),
      el("td", { class: "r num", text: r.num_turns == null ? "—" : String(r.num_turns) }),
      el("td", { class: "r num", text: fmtDuration(r.duration_s) }),
      el("td", { class: "r num", title: r.updated_at ?? "" }, relTime(r.updated_at)),
    );
  }));
}

// ---- wire up ----

$("#filter-text").addEventListener("input", renderTable);
$("#filter-state").addEventListener("change", renderTable);
$("#filter-task").addEventListener("change", renderTable);
$("#filter-track").addEventListener("change", renderTable);

load();
setInterval(load, POLL_MS);
