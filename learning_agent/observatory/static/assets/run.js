// Run view: left overview rail / center tabs (Trace · Judge · Scores · Workspace) /
// right system-metrics rail. Live runs poll /status and append new trace events.

const RUN_ID = new URLSearchParams(location.search).get("id") || "";
const API = `/api/runs/${encodeURIComponent(RUN_ID)}`;
const LIVE_POLL_MS = 5000;
const TABS = ["trace", "judge", "learning", "scores", "workspace"];

const state = {
  record: null,
  status: null,
  workspace: null,
  wsFetched: false,
  renderedEvents: 0,
  lastGutterTurn: null,
  toolSlots: new Map(),      // tool_use_id -> slot element inside its tool card
  toolsExpanded: false,      // trace-wide fold state for tool calls (default collapsed)
  thoughtsExpanded: false,   // trace-wide fold state for thinking cards (default collapsed)
  charts: [],
  modalCharts: [],
  liveTimer: null,
  lastUpdated: null,
  follow: true,
};

const $ = (sel, root = document) => root.querySelector(sel);

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

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ---- formatting ----

function fmtScore(x) {
  if (x == null || !Number.isFinite(x)) return "—";
  return (Math.round(x * 1000) / 1000).toString();
}

function fmtCost(x) { return x == null ? "—" : `$${x.toFixed(2)}`; }

function fmtInt(x) { return x == null ? "—" : String(x); }

function fmtDurationS(s) {
  if (s == null || !Number.isFinite(s)) return "—";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function fmtElapsed(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${mm}:${ss}`;
}

function hhmm(sec) {
  const h = String(Math.floor(sec / 3600)).padStart(2, "0");
  const m = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
  return `${h}:${m}`;
}

function fmtTokens(n) {
  if (n == null) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function fmtBytes(n) {
  if (n == null) return "—";
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MiB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KiB`;
  return `${n} B`;
}

function shortSha(v) { return typeof v === "string" && v.length > 8 ? v.slice(0, 8) : String(v); }

function stateChip(s, id) {
  s = s || "unknown";
  return el("span", { class: `chip state-${s}`, id }, el("span", { class: "dot" }), s);
}

// medium/hard tracks: the agent authored its own dev gold, so dev scores are
// self-measured and not comparable across runs — badge wherever one renders.
function selfReportedChip(track) {
  if (track !== "medium" && track !== "hard") return null;
  return el("span", { class: "chip b-amber", title: "medium/hard track — the agent authored its own dev gold; this score is self-measured and not comparable across runs" }, "self-reported");
}

function kv(k, v, cls) {
  return el("div", { class: "kv" },
    el("span", { class: "k", text: k }),
    el("span", { class: `v num ${cls ?? ""}` }, v ?? "—"));
}

function empty(text) { return el("div", { class: "empty", text }); }

// ---- boot ----

async function boot() {
  if (!RUN_ID) { fatal("missing ?id=<run_id>"); return; }
  document.title = `${RUN_ID} · Learning Agent Observatory`;
  $("#crumb").textContent = RUN_ID;
  try {
    const [rec, st] = await Promise.all([
      getJSON(API),
      getJSON(`${API}/status`).catch(() => null),
    ]);
    state.record = rec ?? {};
    state.status = st;
    state.lastUpdated = st?.updated_at ?? rec?.index_row?.updated_at ?? null;
  } catch (e) {
    fatal(`failed to load run "${RUN_ID}": ${e.message}`);
    return;
  }
  renderLeftRail();
  renderTraceTab();
  renderJudgeTab();
  renderLearningTab();
  renderScoresTab();
  buildWorkspaceShell();
  renderRightRail();
  initTabs();
  maybeStartLive();
}

function fatal(msg) {
  const box = $("#fatal");
  box.hidden = false;
  box.textContent = msg;
}

function curState() { return state.status?.state ?? state.record?.index_row?.state; }

// ---- tabs ----

function tabFromHash() {
  const m = /tab=([a-z]+)/.exec(location.hash);
  return m && TABS.includes(m[1]) ? m[1] : "trace";
}

function activateTab(name, setHash = true) {
  for (const b of document.querySelectorAll("#tabs button"))
    b.classList.toggle("active", b.dataset.tab === name);
  for (const t of TABS)
    $(`#panel-${t}`).classList.toggle("active", t === name);
  if (name === "workspace") ensureWorkspace();
  if (setHash && tabFromHash() !== name) {
    history.replaceState(null, "", `#tab=${name}`);
  }
}

function initTabs() {
  for (const b of document.querySelectorAll("#tabs button"))
    b.addEventListener("click", () => activateTab(b.dataset.tab));
  window.addEventListener("hashchange", () => activateTab(tabFromHash(), false));
  activateTab(tabFromHash(), false);
}

// ---- left rail ----

function renderLeftRail() {
  const rec = state.record;
  const ir = rec.index_row ?? {};
  const meta = rec.meta ?? {};
  const sum = rec.summary ?? {};
  const rail = $("#left-rail");
  rail.replaceChildren();

  // Stamp suffix on run_id, either form the runners emit: agents/run.sh writes
  // `date +%Y%m%d_%H%M%S` (YYYYMMDD_HHMMSS); only the hand-authored demo fixture
  // uses the YYYYMMDDTHHMMSS form. Accept both.
  const track = ir.track ?? meta.track ?? null;
  const stamp = (/(\d{8}[T_]\d{6})$/.exec(ir.run_id ?? "") || [])[1] ?? meta.launched_at ?? "";
  rail.append(el("div", { class: "rail-card" },
    el("div", { class: "task-name", text: ir.task ?? meta.task ?? "?" }),
    el("div", { class: "run-sub" },
      [ir.scaffold ?? meta.scaffold, ir.agent_model, stamp].filter(Boolean).join(" · ")),
    ir.base_model ? el("div", { class: "run-sub faint", text: `student: ${ir.base_model}` }) : null,
    track ? el("div", { class: "run-sub faint", text: `track: ${track}` }) : null,
  ));

  const ci = Array.isArray(ir.best_dev_ci) && ir.best_dev_ci.length === 2
    ? `CI [${fmtScore(ir.best_dev_ci[0])}, ${fmtScore(ir.best_dev_ci[1])}]` : null;
  const selfChip = selfReportedChip(track);
  rail.append(el("div", { class: "rail-card" },
    el("h3", { text: "best dev score" }),
    el("div", { class: `big-score num ${ir.best_dev_score == null ? "none" : ""}`, text: fmtScore(ir.best_dev_score) }),
    ci ? el("div", { class: "score-ci num", text: ci }) : null,
    ir.best_tag || selfChip ? el("div", { class: "chip-row" },
      ir.best_tag ? el("span", { class: "chip b-tag", text: ir.best_tag }) : null, selfChip) : null,
    el("div", { class: "chip-row" }, stateChip(curState(), "run-state-chip")),
  ));

  const duration = ir.duration_s ?? (sum.duration_ms != null ? sum.duration_ms / 1000 : null);
  rail.append(el("div", { class: "rail-card" },
    el("h3", { text: "run" }),
    kv("time budget", meta.time_budget_h != null ? `${meta.time_budget_h}h` : "—"),
    kv("duration", fmtDurationS(duration)),
    kv("turns", fmtInt(sum.num_turns ?? ir.num_turns)),
    kv("sessions", fmtInt(sum.session_count ?? ir.session_count)),
    // billing-style integer: seconds x n_gpus per job, rounded UP — 1 GPU for
    // any part of an hour counts 1, 4 GPUs count 4 (0 = no GPU job yet)
    kv("gpu hours", String(Math.ceil(ir.gpu_hours ?? 0))),
    // control-plane truth (observatory/gpu_metering.py) next to the agent's
    // self-report above; ~shared marks time-window ambiguity between
    // concurrent runs (untagged sandboxes)
    ir.gpu_hours_metered != null
      ? kv("gpu hours (metered)", String(Math.ceil(ir.gpu_hours_metered))
          + (ir.gpu_metered_shared ? " ~shared" : ""))
      : null,
    kv("cost", fmtCost(sum.total_cost_usd ?? ir.total_cost_usd)),
    kv("exit code", meta.exit_code == null ? "—" : String(meta.exit_code),
      meta.exit_code ? "err" : ""),
  ));

  const u = sum.usage_total ?? {};
  rail.append(el("div", { class: "rail-card" },
    el("h3", { text: "tokens" }),
    kv("input", fmtTokens(u.input_tokens)),
    kv("output", fmtTokens(u.output_tokens)),
    kv("cache write", fmtTokens(u.cache_creation_input_tokens)),
    kv("cache read", fmtTokens(u.cache_read_input_tokens)),
  ));

  const badges = [];
  const audit = ir.audit ?? rec.judgements?.audit?.integrity;
  if (audit === "CLEAN") badges.push(el("span", { class: "chip b-green", text: "audit CLEAN" }));
  else if (audit === "CONTAMINATED") badges.push(el("span", { class: "chip b-red", text: "audit CONTAMINATED" }));
  if (ir.canonical === false) badges.push(el("span", { class: "chip b-amber", text: "non-canonical" }));
  else if (ir.canonical === true) badges.push(el("span", { class: "chip", text: "canonical" }));
  if (ir.integrity === "DIRTY") badges.push(el("span", { class: "chip b-red", text: "integrity DIRTY" }));
  else if (ir.integrity) badges.push(el("span", { class: "chip", text: `integrity ${ir.integrity}` }));
  const tags = (rec.scores?.checkpoints ?? []).map((c) => c?.tag).filter(Boolean);
  rail.append(el("div", { class: "rail-card" },
    el("h3", { text: "provenance" }),
    badges.length ? el("div", { class: "chip-row" }, badges) : el("div", { class: "faint", text: "no badges" }),
    tags.length ? el("h3", { text: "checkpoint tags", style: "margin-top:10px" }) : null,
    tags.length ? el("div", { class: "chip-row" }, tags.map((t) => el("span", { class: "chip", text: t }))) : null,
  ));

  const copyBtn = el("button", { class: "copy-btn", type: "button", text: "copy" });
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(RUN_ID);
      copyBtn.textContent = "copied";
      setTimeout(() => { copyBtn.textContent = "copy"; }, 1200);
    } catch (e) { copyBtn.textContent = "failed"; }
  });
  rail.append(el("div", { class: "rail-card" },
    el("h3", { text: "run id" }),
    el("div", { class: "copy-row" }, el("span", { class: "rid", text: RUN_ID }), copyBtn),
    el("div", { style: "margin-top:8px" },
      el("a", { href: `${API}/raw/trace.jsonl`, target: "_blank", rel: "noopener" }, "raw trace.jsonl ↗")),
  ));
}

function updateStateChips(s) {
  const chip = $("#run-state-chip");
  if (chip) chip.replaceWith(stateChip(s, "run-state-chip"));
}

// ---- trace tab ----

function firstTs(events) {
  for (const e of events) {
    const t = Date.parse(e?.ts ?? "");
    if (Number.isFinite(t)) return t;
  }
  return null;
}

function renderTraceTab() {
  const panel = $("#panel-trace");
  panel.replaceChildren();
  const events = state.record.events ?? [];

  const jump = el("input", { type: "number", min: "1", placeholder: "turn", "aria-label": "jump to turn" });
  jump.addEventListener("change", () => jumpToTurn(parseInt(jump.value, 10)));
  jump.addEventListener("keydown", (e) => { if (e.key === "Enter") jumpToTurn(parseInt(jump.value, 10)); });

  const expand = el("input", { type: "checkbox" });
  expand.addEventListener("change", () => $("#trace-list")?.classList.toggle("outputs-all", expand.checked));

  const toolsBox = el("input", { type: "checkbox" });
  toolsBox.checked = state.toolsExpanded;
  toolsBox.addEventListener("change", () => {
    state.toolsExpanded = toolsBox.checked;
    for (const card of document.querySelectorAll("#trace-list .tool-card:not(.thinking-card)")) {
      card.classList.toggle("folded", !state.toolsExpanded);
    }
  });

  const thoughtsBox = el("input", { type: "checkbox" });
  thoughtsBox.checked = state.thoughtsExpanded;
  thoughtsBox.addEventListener("change", () => {
    state.thoughtsExpanded = thoughtsBox.checked;
    for (const card of document.querySelectorAll("#trace-list .thinking-card")) {
      card.classList.toggle("folded", !state.thoughtsExpanded);
    }
  });

  const controls = el("div", { class: "panel" },
    el("div", { class: "panel-head trace-controls" },
      el("span", { id: "trace-counts", class: "muted" }),
      el("span", { class: "grow" }),
      el("label", {}, "jump to turn ", jump),
      el("label", {}, thoughtsBox, " expand thoughts"),
      el("label", {}, toolsBox, " expand tool calls"),
      el("label", {}, expand, " expand outputs"),
    ));
  const followBox = el("span", { id: "follow-box" });
  controls.firstChild.append(followBox);
  panel.append(controls);

  const list = el("div", { class: "trace-list", id: "trace-list" });
  panel.append(list);
  if (!events.length) list.append(empty("no trace events"));

  state.renderedEvents = 0;
  state.lastGutterTurn = null;
  state.toolSlots.clear();
  appendEvents();
  updateTraceCounts();
  renderFollowToggle();
}

function updateTraceCounts() {
  const events = state.record.events ?? [];
  const ns = state.record.summary?.session_count ?? state.record.sessions?.length ?? 0;
  const c = $("#trace-counts");
  if (c) c.textContent = `${events.length} events · ${ns} session${ns === 1 ? "" : "s"}`;
}

function renderFollowToggle() {
  const box = $("#follow-box");
  if (!box) return;
  box.replaceChildren();
  if (curState() !== "running") return;
  const cb = el("input", { type: "checkbox" });
  cb.checked = state.follow;
  cb.addEventListener("change", () => { state.follow = cb.checked; if (cb.checked) scrollTraceBottom(); });
  box.append(el("label", { title: "keep scrolled to latest event" }, cb, " follow"));
}

function appendEvents() {
  const events = state.record.events ?? [];
  const list = $("#trace-list");
  if (!list) return;
  const t0 = firstTs(events);
  const frag = document.createDocumentFragment();
  for (let i = state.renderedEvents; i < events.length; i++) {
    const row = renderEvent(events[i] ?? {}, t0);
    if (row) frag.append(row);
  }
  state.renderedEvents = events.length;
  list.append(frag);
}

function renderEvent(ev, t0) {
  const cards = [];
  if (ev.type === "system" && ev.subtype === "init") {
    cards.push(sessionStartCard(ev));
  } else if (ev.type === "result") {
    cards.push(sessionEndCard(ev));
  } else if (ev.type === "assistant" || ev.type === "user") {
    for (const b of ev.blocks ?? []) {
      const c = renderBlock(ev, b ?? {});
      if (c) cards.push(c);
    }
  } else { // other system / unknown
    const raw = ev.raw ? JSON.stringify(ev.raw) : "";
    // codex prints this once per CLI session for any model id not in its
    // built-in catalog (our endpoints aren't OpenAI models); the fallback
    // works fine and the raw trace keeps the line — the page drops it.
    if (raw.includes("Model metadata for") && raw.includes("fallback metadata")) {
      // no card
    } else {
      cards.push(el("div", { class: "card sys-line" },
        `system · ${ev.subtype ?? "?"}`,
        raw ? el("span", { class: "faint" }, `  ${raw.length > 160 ? raw.slice(0, 160) + "…" : raw}`) : null));
    }
  }
  if (!cards.length) return null;

  const ts = Date.parse(ev.ts ?? "");
  const elapsed = Number.isFinite(ts) && t0 != null ? fmtElapsed(ts - t0) : `#${ev.i ?? "?"}`;
  const gutter = el("div", { class: "ev-gutter" });
  if (ev.turn != null && ev.turn !== state.lastGutterTurn) {
    gutter.append(el("span", { class: "turn", text: `T${ev.turn}` }));
    state.lastGutterTurn = ev.turn;
  }
  gutter.append(el("span", { class: "num", text: elapsed }));

  const row = el("div", { class: "ev-row" }, gutter, el("div", { class: "ev-body" }, cards));
  row.dataset.i = String(ev.i ?? "");
  if (ev.turn != null) row.dataset.turn = String(ev.turn);
  return row;
}

function sessionStartCard(ev) {
  const s = (state.record.sessions ?? []).find((x) => x?.session_idx === ev.session_idx)
    ?? (state.record.sessions ?? [])[0] ?? {};
  return el("div", { class: "card session-edge" },
    el("div", { class: "edge-title", text: `agent CLI session ${ (ev.session_idx ?? 0) + 1 } start`,
                title: "one launch of the agent CLI — the runner re-prompts it (\"you still have Xh, continue\") when it stops early, and each re-prompt resumes the same conversation as a new CLI session" }),
    kv("model", s.model ?? ev.model ?? "—"),
    kv("cwd", s.cwd ?? "—"),
    kv("permission mode", s.permission_mode ?? "—"),
    kv("tools", s.tools ? String(s.tools.length) : "—"),
  );
}

function sessionEndCard(ev) {
  return el("div", { class: "card session-edge" },
    el("div", { class: "edge-title", text: `session end · ${ev.subtype ?? "result"}` }),
    kv("duration", ev.duration_ms != null ? fmtDurationS(ev.duration_ms / 1000) : "—"),
    kv("turns", fmtInt(ev.num_turns)),
    kv("cost", fmtCost(ev.total_cost_usd)),
    kv("stop reason", ev.stop_reason ?? "—"),
    ev.result ? el("p", { style: "margin-top:6px", text: ev.result }) : null,
  );
}

function renderBlock(ev, b) {
  switch (b.type) {
    case "thinking": {
      const t = b.thinking ?? "";
      const preview = t.replace(/\s+/g, " ").trim();
      const body = el("div", { class: "tool-body" },
        el("div", { class: "card thinking" }, el("p", { text: t })));
      const head = el("div", { class: "tool-head" },
        el("span", { class: "tname", text: "thinking" }),
        el("span", { class: "tsummary", text: preview.length > 110 ? preview.slice(0, 110) + "…" : preview }));
      const card = el("div", { class: `tool-card thinking-card${state.thoughtsExpanded ? "" : " folded"}` }, head, body);
      head.addEventListener("click", () => card.classList.toggle("folded"));
      return card;
    }
    case "text":
      return el("div", { class: "card" },
        ev.type === "user" ? el("div", { class: "role", text: "user" }) : null,
        el("p", { text: b.text ?? "" }));
    case "tool_use":
      return toolUseCard(b);
    case "tool_result": {
      const slot = b.tool_use_id != null ? state.toolSlots.get(b.tool_use_id) : null;
      if (slot) {
        fillResultSlot(slot, b);
        return null; // rendered inside its tool_use card
      }
      // orphan result: standalone terminal block
      const orphanSlot = el("div", { class: "tool-result" });
      fillResultSlot(orphanSlot, b);
      return foldableToolCard(
        [el("span", { class: "tname", text: "tool result" }),
         el("span", { class: "tid", text: b.tool_use_id ?? "" })],
        [orphanSlot]);
    }
    default:
      return el("div", { class: "card sys-line", text: `block · ${b.type ?? "?"}` });
  }
}

function toolInputPreview(b) {
  const inp = b.input ?? {};
  let t = typeof inp.command === "string" ? inp.command
    : typeof inp.pattern === "string" ? inp.pattern
    : typeof inp.path === "string" ? inp.path
    : (() => { try { return JSON.stringify(inp); } catch (e) { return String(inp); } })();
  t = String(t).replace(/\s+/g, " ").trim();
  return t.length > 110 ? t.slice(0, 110) + "…" : t;
}

function foldableToolCard(headChildren, bodyChildren) {
  const body = el("div", { class: "tool-body" }, ...bodyChildren);
  const head = el("div", { class: "tool-head" }, ...headChildren);
  const card = el("div", { class: `tool-card${state.toolsExpanded ? "" : " folded"}` }, head, body);
  head.addEventListener("click", () => card.classList.toggle("folded"));
  return card;
}

function toolUseCard(b) {
  let input = "";
  try { input = JSON.stringify(b.input ?? {}, null, 1); } catch (e) { input = String(b.input); }
  const slot = el("div", { class: "tool-result" },
    el("div", { class: "pending", text: "no result yet" }));
  const card = foldableToolCard(
    // codex normalizes its shell executions to name "command"; the card
    // label reads "tool" — the summary already shows the command itself
    [el("span", { class: "tname", text: b.name === "command" ? "tool" : (b.name ?? "tool") }),
     el("span", { class: "tsummary", text: toolInputPreview(b) }),
     el("span", { class: "tid", text: b.id ?? "" })],
    [clampable(input, "tool-input"), slot]);
  if (b.id != null) state.toolSlots.set(b.id, slot);
  return card;
}

function resultText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((p) => {
      if (p == null) return "";
      if (typeof p === "string") return p;
      if (typeof p.text === "string") return p.text;
      try { return JSON.stringify(p); } catch (e) { return String(p); }
    }).join("\n");
  }
  try { return JSON.stringify(content, null, 1); } catch (e) { return String(content); }
}

function fillResultSlot(slot, b) {
  slot.classList.toggle("is-error", b.is_error === true);
  slot.replaceChildren(
    el("div", { class: "result-tag", text: b.is_error ? "result · error" : "result" }),
    clampable(resultText(b.content)),
  );
}

function clampable(text, cls) {
  const wrap = el("div", { class: "clamp-wrap" });
  const pre = el("pre", { class: `clamp ${cls ?? ""}`, text });
  wrap.append(pre);
  const lines = text.split("\n").length;
  if (lines > 6 || text.length > 700) {
    wrap.classList.add("collapsible");
    const btn = el("button", { class: "clamp-btn", type: "button", text: `▸ expand (${lines} lines)` });
    btn.addEventListener("click", () => {
      const open = wrap.classList.toggle("expanded");
      btn.textContent = open ? "▾ collapse" : `▸ expand (${lines} lines)`;
    });
    wrap.append(btn);
  }
  return wrap;
}

function jumpToTurn(n) {
  if (!Number.isFinite(n)) return;
  const row = $(`#trace-list [data-turn="${n}"]`);
  if (!row) return;
  activateTab("trace");
  row.scrollIntoView({ behavior: "smooth", block: "start" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

// same jump-and-flash mechanism as jumpToTurn, but by event index (every
// trace row carries data-i; not every event carries a turn) — used by the
// Learning tab to jump to the trace event a learning action was found in.
function jumpToEvent(i) {
  if (i == null || !Number.isFinite(+i)) return;
  const row = $(`#trace-list [data-i="${i}"]`);
  if (!row) return;
  activateTab("trace");
  row.scrollIntoView({ behavior: "smooth", block: "start" });
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 1600);
}

function scrollTraceBottom() {
  const list = $("#trace-list");
  if (list) window.scrollTo({ top: list.getBoundingClientRect().bottom + window.scrollY, behavior: "smooth" });
}

// ---- live mode ----

function maybeStartLive() {
  if (curState() !== "running") return;
  state.liveTimer = setInterval(pollStatus, LIVE_POLL_MS);
}

async function pollStatus() {
  let st;
  try { st = await getJSON(`${API}/status`); } catch (e) { return; }
  state.status = st;
  updateStateChips(st.state);
  if (st.updated_at && st.updated_at !== state.lastUpdated) {
    state.lastUpdated = st.updated_at;
    try { state.record = await getJSON(API); } catch (e) { return; }
    appendEvents();
    updateTraceCounts();
    renderLeftRail();
    renderJudgeTab();
    renderLearningTab();
    renderScoresTab();
    renderRightRail();
    if (state.follow) scrollTraceBottom();
  }
  if (st.state !== "running" && state.liveTimer) {
    clearInterval(state.liveTimer);
    state.liveTimer = null;
    renderFollowToggle();
  }
}

// ---- judge tab ----

function renderJudgeTab() {
  const panel = $("#panel-judge");
  panel.replaceChildren();
  panel.append(auditSection(state.record.judgements?.audit));

  const results = state.record.scores?.results ?? [];
  if (!results.length) { panel.append(empty("no judged results")); return; }

  const sel = el("select", { "aria-label": "result" },
    results.map((r, i) => el("option", { value: String(i) },
      `${r?.tag ?? "?"} · ${r?.split ?? "?"} · budget ${r?.budget ?? "?"}`)));
  const holder = el("div", { style: "display:flex;flex-direction:column;gap:14px" });
  sel.addEventListener("change", () => renderJudgeResult(results[+sel.value], holder));
  panel.append(el("div", { class: "judge-controls" }, el("h2", { text: "per-question drill-down" }), sel));
  panel.append(holder);
  renderJudgeResult(results[0], holder);
}

function auditSection(audit) {
  const box = el("div", { style: "display:flex;flex-direction:column;gap:8px" });
  if (!audit) { box.append(empty("no audit")); return box; }
  const verdict = audit.integrity ?? "?";
  const clean = verdict === "CLEAN";
  const cards = el("div", { class: "audit-cards" });
  cards.append(el("div", { class: "audit-card" },
    el("h3", { text: "audit verdict" }),
    el("div", { class: `verdict ${clean ? "clean" : "contaminated"}` },
      clean ? "✓" : "✕", verdict),
    audit.task ? el("div", { class: "faint", text: `task: ${audit.task}` }) : null,
  ));
  const findings = audit.findings ?? [];
  cards.append(el("div", { class: "audit-card" },
    el("h3", { text: "findings" }),
    findings.length
      ? el("ul", {}, findings.map((f) => el("li", { text: typeof f === "string" ? f : JSON.stringify(f) })))
      : el("div", { class: "faint", text: "none" }),
  ));
  const grid = (obj) => el("div", { class: "kv-grid" },
    Object.entries(obj ?? {}).map(([k, v]) => el("div", { class: "cell" },
      el("span", { class: "k", text: k }), el("span", { class: "v num", text: String(v) }))));
  cards.append(el("div", { class: "audit-card" }, el("h3", { text: "access counts" }), grid(audit.access_counts)));
  cards.append(el("div", { class: "audit-card" }, el("h3", { text: "behavior" }), grid(audit.behavior)));
  box.append(cards);
  if (audit.caveat) box.append(el("div", { class: "caveat", text: `caveat: ${audit.caveat}` }));
  return box;
}

function renderJudgeResult(r, holder) {
  holder.replaceChildren();
  if (!r) { holder.append(empty("no result")); return; }
  const pq = r.per_question ?? {};
  const qids = Object.keys(pq).sort();

  const tbody = el("tbody");
  for (const qid of qids) {
    const q = pq[qid] ?? {};
    const failed = q.failed === true || q.claim_score == null;
    const scoreCell = el("td", { class: "r num" });
    if (q.claim_score == null) {
      scoreCell.append("— ", el("span", { class: "chip badge-failed", text: "failed" }));
    } else {
      scoreCell.append(fmtScore(q.claim_score));
    }
    const row = el("tr", { class: "q-row" },
      el("td", { text: qid }),
      scoreCell,
      el("td", { class: "r num", text: fmtInt(q.tool_calls) }),
      el("td", { class: "r num", text: fmtInt(q.completion_tokens) }),
    );
    const detail = el("tr", { class: "q-detail", hidden: "" },
      el("td", { colspan: "4" }, claimDetail(q, failed)));
    row.addEventListener("click", () => { detail.hidden = !detail.hidden; });
    tbody.append(row, detail);
  }

  holder.append(el("div", { class: "panel" },
    el("div", { class: "panel-head" },
      el("h2", { text: `questions · ${qids.length}` }),
      el("span", { class: "faint", text: "click a row for per-claim verdicts" })),
    el("div", { class: "scroll-x" },
      qids.length
        ? el("table", { class: "data" },
            el("thead", {}, el("tr", {},
              el("th", { text: "qid" }), el("th", { class: "r", text: "claim score" }),
              el("th", { class: "r", text: "tool calls" }), el("th", { class: "r", text: "completion tokens" }))),
            tbody)
        : empty("no per-question data")),
  ));
  holder.append(provenanceCard(r));
}

function claimDetail(q, failed) {
  const verdicts = q.verdicts ?? {};
  const claims = Object.keys(verdicts).sort();
  const box = el("div", {});
  if (!claims.length) {
    box.append(el("div", { class: "faint", text: failed ? "no verdicts — question failed" : "no verdicts" }));
  } else {
    box.append(el("table", { class: "data claims-table" },
      el("thead", {}, el("tr", {},
        el("th", { text: "claim" }), el("th", { class: "r", text: "final" }), el("th", { text: "votes" }))),
      el("tbody", {}, claims.map((c) => {
        const votes = q.votes?.[c];
        return el("tr", {},
          el("td", { text: c }),
          el("td", { class: "r num", text: fmtScore(verdicts[c]) }),
          el("td", { class: "num", text: Array.isArray(votes) ? votes.map(fmtScore).join(", ") : "—" }));
      }))));
  }
  if (q.secondary?.kind != null) {
    box.append(el("div", { class: "faint", style: "margin-top:6px",
      text: `secondary · ${q.secondary.kind}: ${fmtScore(q.secondary.score)}` }));
  }
  if (typeof q.answer === "string" && q.answer) {
    box.append(el("div", { style: "margin-top:6px" },
      el("div", { class: "role", text: "answer" }), el("p", { style: "white-space:pre-wrap;margin:0", text: q.answer })));
  }
  return box;
}

function provenanceCard(r) {
  const p = r.provenance ?? {};
  const cell = (k, v, title) => el("div", { class: "cell" },
    el("span", { class: "k", text: k }),
    el("span", { class: "v num", title: title ?? "" }, v ?? "—"));
  const shaCell = (k) => p[k] == null ? null
    : cell(k.replace(/_sha$/, " sha"), el("span", { class: "sha", text: `${shortSha(p[k])}…` }), String(p[k]));
  return el("div", { class: "prov-card" },
    el("h3", { text: "judge provenance", style: "font-size:11px;color:var(--fg-muted);margin-bottom:8px" }),
    el("div", { class: "kv-grid" },
      cell("judge model", p.judge_model),
      cell("backend", p.judge_backend),
      cell("canonical", p.canonical == null ? "—" : String(p.canonical)),
      cell("n_votes", p.n_votes == null ? "—" : String(p.n_votes)),
      shaCell("judge_prompt_sha"), shaCell("harness_sha"), shaCell("sys_sha"),
      shaCell("gold_sha"), shaCell("config_sha"),
      p.corpus_pin != null ? cell("corpus pin", String(p.corpus_pin)) : null,
      cell("seed", p.seed == null ? "—" : String(p.seed)),
      cell("timestamp", p.timestamp),
    ));
}

// ---- learning tab ----
//
// Renders RunRecord.learning (see observatory/schema.py LearningAction):
// {event_i, ts, kind, tool, provenance, command, args, nth_use}. kind is one
// of data|train|eval|evolve (a seed-tool registry match) or "tool" (an
// invented, non-registry script — provenance "invented"). Old records may
// have no `learning` key at all (predates the learning timeline); `?? []`
// below keeps that a quiet empty state, never a crash.

const LEARNING_KIND_CLASS = { data: "b-tag", train: "b-green", eval: "b-blue", evolve: "b-violet", harness: "b-amber", infra: "b-tag" };

function learningKindChip(a) {
  // "invented" is a provenance fact, not a kind: seed tools whose category
  // the registry doesn't know (old records) also carry kind "tool" and must
  // NOT wear the star.
  if (a.provenance === "invented") {
    return el("span", { class: "chip b-amber",
      title: "invented tool — written by the agent, not in the seed manifest" },
      "★ invented");
  }
  return el("span", { class: `chip ${LEARNING_KIND_CLASS[a.kind] ?? ""}`, text: a.kind ?? "?" });
}

function learningArgsText(args) {
  if (!args || !Object.keys(args).length) return "—";
  return Object.entries(args).map(([k, v]) => `${k}=${v}`).join(" ");
}

function renderLearningTab() {
  const panel = $("#panel-learning");
  panel.replaceChildren();
  // The research log IS the agent's own lab notebook (runs/LEARNING_LOG.jsonl):
  // every checkpoint, dataset, and submission with its dev score and the
  // agent's own notes. Entries fold to one line; click to expand.
  const llog = state.record.scores?.learning_log ?? [];
  if (!llog.length) {
    panel.append(empty("no research log yet — the agent hasn't written runs/LEARNING_LOG.jsonl"));
    return;
  }
  const kindChip = (k) => el("span", {
    class: `chip ${k === "checkpoint" ? "b-green" : k === "submission" ? "b-tag" : ""}`,
    text: k || "note" });
  const list = el("div", { class: "rlog" });
  for (const e of llog) {
    const preview = [e.what, e.why].filter(Boolean).join(" — ").replace(/\s+/g, " ");
    const head = el("div", { class: "rlog-head" },
      el("span", { class: "num rlog-ts", text: (e.ts || "").replace("T", " ").replace("Z", "").slice(5, 16) }),
      kindChip(e.kind),
      el("span", { class: "rlog-tag", text: e.tag || "" }),
      e.dev_score == null ? null : el("span", { class: "num rlog-score", text: fmtScore(e.dev_score) }),
      el("span", { class: "rlog-preview", text: preview }));
    const body = el("div", { class: "rlog-body" });
    const add = (label, v) => {
      if (v == null || v === "" || (Array.isArray(v) && !v.length)) return;
      const text = Array.isArray(v) ? v.join("  ")
        : typeof v === "object" ? JSON.stringify(v, null, 1) : String(v);
      body.append(el("div", { class: "rlog-field" },
        el("span", { class: "k", text: label }), el("p", { text })));
    };
    add("research notes", e.what);
    add("why", e.why);
    add("result", e.result);
    add("model path", e.model_path);
    add("artifacts", e.artifacts);
    const card = el("div", { class: "rlog-entry folded" }, head, body);
    head.addEventListener("click", () => card.classList.toggle("folded"));
    list.append(card);
  }
  panel.append(el("div", { class: "panel" },
    el("div", { class: "panel-head" },
      el("h2", { text: `research log · ${llog.length}` }),
      el("span", { class: "faint", text: "the agent's own notes — click an entry to expand" })),
    list));
}

// Cards of tools this run created (RunRecord.invented_tools: TOOL_SPEC folders
// found in the workspace snapshot but not the seed manifest). The card fields
// are read from the raw tool.yaml by line — same yaml-lite the spec guarantees.
function cardField(yamlText, key) {
  for (const line of (yamlText ?? "").split("\n")) {
    if (line.startsWith(`${key}:`)) return line.slice(key.length + 1).trim();
  }
  return "";
}

function renderInventedCards(panel) {
  const cards = state.record.invented_tools ?? [];
  if (!cards.length) return;
  const body = el("div", { style: "padding: 10px 14px;" });
  for (const c of cards) {
    const name = cardField(c.tool_yaml, "name") || c.path.split("/").pop();
    const kind = cardField(c.tool_yaml, "kind");
    const cost = cardField(c.tool_yaml, "cost");
    const by = cardField(c.tool_yaml, "created_by");
    const summary = cardField(c.tool_yaml, "summary");
    body.append(el("div", { style: "margin: 8px 0;" },
      el("span", { class: "chip b-amber", text: "★ invented" }), " ",
      el("strong", { text: name }), " ",
      kind ? el("span", { class: "faint", text: `kind=${kind} ` }) : null,
      cost ? el("span", { class: "faint", text: `cost=${cost} ` }) : null,
      by ? el("span", { class: "faint", text: `by ${by}` }) : null,
      el("div", { class: "faint", text: summary || "(no summary in card)" }),
      el("div", { class: "faint mono", text: c.path })));
  }
  panel.append(el("div", { class: "panel" },
    el("div", { class: "panel-head" },
      el("h2", { text: `invented tools · ${cards.length}` }),
      el("span", { class: "faint", text: "TOOL_SPEC folders created by this run — full files in the workspace tab" })),
    body));
}

// ---- scores tab ----

function renderScoresTab() {
  const panel = $("#panel-scores");
  panel.replaceChildren();
  const scores = state.record.scores ?? {};
  const results = scores.results ?? [];
  const track = state.record.index_row?.track ?? state.record.meta?.track ?? null;

  const grid = el("div", { class: "results-grid" });
  for (const r of results) grid.append(resultCard(r ?? {}, track));
  panel.append(el("div", {},
    el("h2", { text: "results", style: "margin-bottom:8px" }),
    results.length ? grid : empty("no results")));

  const devScoreCell = (c) => {
    const chip = selfReportedChip(track);
    return chip ? el("span", {}, fmtScore(c.dev_score), " ", chip) : fmtScore(c.dev_score);
  };
  // the research log lives on the Learning tab now
  const llog = scores.learning_log ?? [];
  // Legacy checkpoint ledger (pre-learning-log runs; for newer runs the
  // collector synthesizes these rows from the log's checkpoint entries).
  if (!llog.length || (scores.checkpoints ?? []).some((c) => c.method)) {
    panel.append(simpleTablePanel("checkpoints", scores.checkpoints ?? [],
      [["tag", (c) => c.tag], ["dev score", devScoreCell, "r num"],
       ["model path", (c) => c.model_path, "wrap"], ["method", (c) => c.method, "wrap"]]));
  }

  const lb = scores.leaderboard ?? [];
  const preferred = ["task", "tag", "split", "score", "ci", "n", "failed", "n_failed",
    "secondary_mean", "judge_model", "backend", "canonical", "integrity"];
  const cols = preferred.filter((k) => lb.some((row) => row && k in row));
  panel.append(simpleTablePanel("leaderboard", lb,
    cols.map((k) => [k.replace(/_/g, " "), (row) => lbCell(row[k]),
      typeof lb[0]?.[k] === "number" ? "r num" : ""])));
}

function lbCell(v) {
  if (v == null) return "—";
  if (Array.isArray(v)) return `[${v.map(fmtScore).join(", ")}]`;
  if (typeof v === "number") return fmtScore(v);
  return String(v);
}

function simpleTablePanel(title, rows, cols) {
  return el("div", { class: "panel" },
    el("div", { class: "panel-head" }, el("h2", { text: `${title} · ${rows.length}` })),
    el("div", { class: "scroll-x" },
      rows.length
        ? el("table", { class: "data" },
            el("thead", {}, el("tr", {}, cols.map(([h, , cls]) => el("th", { class: cls ?? "", text: h })))),
            el("tbody", {}, rows.map((row) => el("tr", {},
              cols.map(([, get, cls]) => el("td", { class: cls ?? "" }, get(row ?? {}) ?? "—"))))))
        : empty(`no ${title}`)));
}

function resultCard(r, track) {
  const ci = Array.isArray(r.bootstrap_ci95) && r.bootstrap_ci95.length === 2
    ? `CI [${fmtScore(r.bootstrap_ci95[0])}, ${fmtScore(r.bootstrap_ci95[1])}]` : null;
  const stats = el("div", { class: "rc-stats" });
  stats.append(el("span", { class: "num" }, `n ${fmtInt(r.n)}`));
  stats.append(el("span", { class: "num" }, `failed ${fmtInt(r.n_failed)}`));
  if (r.secondary_mean != null) stats.append(el("span", { class: "num" }, `secondary ${fmtScore(r.secondary_mean)}`));
  if (r.tool_calls_avg != null) {
    const warn = r.tool_calls_avg < 0.5;
    stats.append(el("span", { class: "num" }, `tools/q ${fmtScore(r.tool_calls_avg)} `,
      warn ? el("span", { class: "warn-ico", title: "avg tool calls < 0.5 — search collapse?" }, "⚠") : null));
  }
  const chips = el("div", { class: "chip-row" });
  if (r.canonical === false) chips.append(el("span", { class: "chip b-amber", text: "non-canonical" }));
  else if (r.canonical === true) chips.append(el("span", { class: "chip", text: "canonical" }));
  if (r.integrity === "DIRTY") chips.append(el("span", { class: "chip b-red", text: "integrity DIRTY" }));
  else if (r.integrity) chips.append(el("span", { class: "chip", text: `integrity ${r.integrity}` }));
  if (r.all_failed) chips.append(el("span", { class: "chip badge-failed", text: "all failed" }));
  const selfChip = selfReportedChip(track);
  if (selfChip) chips.append(selfChip);
  return el("div", { class: "result-card" },
    el("div", { class: "rc-head", text: `${r.tag ?? "?"} · ${r.split ?? "?"} · budget ${r.budget ?? "?"}` }),
    el("div", { class: `rc-mean num ${r.mean == null ? "none" : ""}`, text: fmtScore(r.mean) }),
    ci ? el("div", { class: "rc-ci num", text: ci }) : null,
    stats, chips);
}

// ---- workspace tab ----

function buildWorkspaceShell() {
  $("#panel-workspace").replaceChildren(empty("open the tab to load the workspace snapshot"));
}

async function ensureWorkspace() {
  if (state.wsFetched) return;
  state.wsFetched = true;
  const panel = $("#panel-workspace");
  panel.replaceChildren(empty("loading workspace…"));
  try {
    state.workspace = await getJSON(`${API}/workspace`);
  } catch (e) {
    // the watcher snapshots the workspace only on the final ingest —
    // during a live run there is nothing to fetch yet, and that's normal
    const live = curState() === "running";
    panel.replaceChildren(empty(live
      ? "workspace snapshot is captured when the run finishes — check back after the run ends"
      : `workspace unavailable: ${e.message}`));
    state.wsFetched = false;
    return;
  }
  renderWorkspace();
}

function renderWorkspace() {
  const ws = state.workspace ?? {};
  const files = ws.files ?? [];
  const panel = $("#panel-workspace");
  panel.replaceChildren();

  panel.append(el("div", { class: "muted" },
    `${ws.total_files ?? files.length} files · ${fmtBytes(ws.total_bytes)} · ${ws.inlined_files ?? "?"} inlined`,
    ws.built_at ? el("span", { class: "faint" }, `  ·  snapshot ${ws.built_at}`) : null));

  if (!files.length) { panel.append(empty("empty workspace snapshot")); return; }

  const viewer = el("div", { class: "ws-view" }, empty("select a file"));
  const tree = el("div", { class: "ws-tree" });
  tree.append(...buildTree(files, viewer));
  panel.append(el("div", { class: "ws-layout" }, tree, viewer));
}

function buildTree(files, viewer) {
  // nested {dirs: Map, files: []}
  const root = { dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = (f.path ?? "").split("/").filter(Boolean);
    let node = root;
    for (const part of parts.slice(0, -1)) {
      if (!node.dirs.has(part)) node.dirs.set(part, { dirs: new Map(), files: [] });
      node = node.dirs.get(part);
    }
    node.files.push({ ...f, name: parts[parts.length - 1] ?? f.path });
  }
  const dirSize = (node) => node.files.reduce((a, f) => a + (f.size ?? 0), 0)
    + [...node.dirs.values()].reduce((a, d) => a + dirSize(d), 0);

  function renderNode(node) {
    const out = [];
    for (const [name, child] of [...node.dirs.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
      const det = el("details", { open: "" },
        el("summary", {}, `${name}/`, el("span", { class: "sz num", text: fmtBytes(dirSize(child)) })),
        renderNode(child));
      out.push(det);
    }
    for (const f of [...node.files].sort((a, b) => a.name.localeCompare(b.name))) {
      const btn = el("button", { class: "ws-file", type: "button" },
        f.name, el("span", { class: "sz num", text: fmtBytes(f.size) }));
      btn.addEventListener("click", () => {
        for (const b of document.querySelectorAll(".ws-file.active")) b.classList.remove("active");
        btn.classList.add("active");
        showFile(f, viewer);
      });
      out.push(btn);
    }
    return out;
  }
  return renderNode(root);
}

function showFile(f, viewer) {
  viewer.replaceChildren();
  viewer.append(el("div", { class: "view-head" },
    el("span", { class: "path", text: f.path ?? "?" }),
    el("span", { class: "num", text: fmtBytes(f.size) }),
    f.truncated ? el("span", { class: "chip b-amber", text: "truncated" }) : null));
  if (!f.inline || f.content == null) {
    viewer.append(empty(`not inlined · ${fmtBytes(f.size)} — too large or binary`));
    return;
  }
  const code = el("div", { class: "code-view" });
  const lines = f.content.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  lines.forEach((ln, i) => code.append(el("div", { class: "cl" },
    el("span", { class: "ln num", text: String(i + 1) }),
    el("span", { class: "lc", text: ln === "" ? " " : ln }))));
  viewer.append(code);
}

// ---- right rail: system charts ----

function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const get = (v) => cs.getPropertyValue(v).trim();
  return {
    accent: get("--accent"),
    accentSoft: `rgba(${get("--accent-rgb")}, 0.12)`,
    grid: get("--border"),
    tick: get("--fg-muted"),
  };
}

function chartSpecs(samples) {
  let t0 = null;
  for (const s of samples) {
    const t = Date.parse(s?.ts ?? "");
    if (Number.isFinite(t)) { t0 = t; break; }
  }
  const pts = (get) => {
    const out = [];
    for (const s of samples) {
      const t = Date.parse(s?.ts ?? "");
      const y = get(s ?? {});
      if (Number.isFinite(t) && t0 != null && y != null && Number.isFinite(y))
        out.push({ x: (t - t0) / 1000, y });
    }
    return out;
  };
  const specs = [];
  if (samples.some((s) => s?.gpu)) {
    specs.push({ title: "gpu util %", data: pts((s) => s.gpu?.util_pct), max: 100 });
    specs.push({ title: "gpu mem MiB", data: pts((s) => s.gpu?.mem_used_mib) });
    specs.push({ title: "gpu temp °C", data: pts((s) => s.gpu?.temp_c) });
    specs.push({ title: "gpu power W", data: pts((s) => s.gpu?.power_w) });
  }
  specs.push({ title: "cpu load 1m", data: pts((s) => s.cpu_load_1m) });
  specs.push({ title: "mem used GiB", data: pts((s) => s.mem_used_gib) });
  const gpu = gpuHoursSpec(samples, t0);
  if (gpu) specs.push(gpu);
  return specs.filter((s) => s.data.length);
}

// Cumulative GPU-hours over the run, from runs/GPU_LOG.jsonl rows
// (seconds x n_gpus per job, stepped at each job's end ts). Always drawn,
// flat at 0 until the agent's first GPU job lands in the ledger.
function gpuHoursSpec(samples, t0) {
  const rows = state.record?.scores?.gpu_log ?? [];
  const jobs = rows
    .map((r) => ({ t: Date.parse(r?.ts ?? ""),
                   h: (Number(r?.seconds) || 0) * (Number(r?.n_gpus) || 0) / 3600 }))
    .filter((p) => Number.isFinite(p.t))
    .sort((a, b) => a.t - b.t);
  if (t0 == null) t0 = jobs.length ? jobs[0].t : null;
  const lastSample = samples.length ? Date.parse(samples[samples.length - 1]?.ts ?? "") : NaN;
  const end = Number.isFinite(lastSample) && t0 != null ? (lastSample - t0) / 1000
    : jobs.length && t0 != null ? Math.max(1, (jobs[jobs.length - 1].t - t0) / 1000) : 1;
  const data = [{ x: 0, y: 0 }];
  let cum = 0;
  for (const j of jobs) {
    const x = Math.max(0, (j.t - t0) / 1000);
    data.push({ x, y: cum });      // step: flat until the job lands...
    cum += j.h;
    data.push({ x, y: cum });      // ...then jump by its gpu-hours
  }
  data.push({ x: Math.max(end, data[data.length - 1].x), y: cum });
  // suggestedMax keeps the axis 0..1 while jobs are seconds long — a 3 s
  // sanity check (0.0008 gpu-h) reads as ~0, not a zoomed-in wiggle
  return { title: "gpu hours (cumulative)", data, max: 1 };
}

function makeChart(canvas, spec, big) {
  const c = themeColors();
  return new Chart(canvas, {
    type: "line",
    data: { datasets: [{
      data: spec.data,
      borderColor: c.accent,
      backgroundColor: c.accentSoft,
      fill: true,
      borderWidth: 1.2,
      pointRadius: 0,
      tension: 0.15,
    }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      normalized: true,
      scales: {
        x: {
          type: "linear",
          grid: { color: c.grid, display: big },
          border: { color: c.grid },
          ticks: { color: c.tick, maxTicksLimit: big ? 9 : 4, font: { size: big ? 11 : 9 },
                   callback: (v) => hhmm(v) },
        },
        y: {
          beginAtZero: true,
          suggestedMax: spec.max,
          grid: { color: c.grid, display: big },
          border: { color: c.grid },
          ticks: { color: c.tick, maxTicksLimit: big ? 6 : 3, font: { size: big ? 11 : 9 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: (items) => items.length ? hhmm(items[0].parsed.x) : "" } },
      },
    },
  });
}

function chartBlock(spec) {
  const canvas = el("canvas");
  const box = el("div", { class: "mini-chart" },
    el("div", { class: "chart-title", text: spec.title }),
    el("div", { class: "cwrap" }, canvas));
  return { box, canvas };
}

function renderRightRail() {
  for (const c of state.charts) c.destroy();
  state.charts = [];
  const rail = $("#right-rail");
  rail.replaceChildren();
  const expandBtn = el("button", { type: "button", text: "expand" });
  expandBtn.addEventListener("click", openChartModal);
  rail.append(el("div", { class: "rail-head" },
    el("h3", { text: "system", title: "the agent's CPU container — GPU jobs run as separate Modal jobs; their hours are in the gpu-h column and runs/GPU_LOG.jsonl" }),
    expandBtn));

  const samples = state.record.system_monitor ?? [];
  if (!samples.length) {
    expandBtn.disabled = true;
    rail.append(el("div", { class: "rail-card empty", text: "no telemetry samples for this run" }));
    return;
  }
  if (typeof Chart === "undefined") {
    expandBtn.disabled = true;
    rail.append(el("div", { class: "rail-card empty", text: "charts unavailable — Chart.js CDN not loaded" }));
    return;
  }
  for (const spec of chartSpecs(samples)) {
    const { box, canvas } = chartBlock(spec);
    rail.append(box);
    state.charts.push(makeChart(canvas, spec, false));
  }
}

function openChartModal() {
  const modal = $("#chart-modal");
  const grid = $("#modal-grid");
  for (const c of state.modalCharts) c.destroy();
  state.modalCharts = [];
  grid.replaceChildren();
  const samples = state.record.system_monitor ?? [];
  if (typeof Chart === "undefined" || !samples.length) return;
  for (const spec of chartSpecs(samples)) {
    const { box, canvas } = chartBlock(spec);
    grid.append(box);
    state.modalCharts.push(makeChart(canvas, spec, true));
  }
  modal.classList.remove("hidden");
}

function closeChartModal() {
  $("#chart-modal").classList.add("hidden");
  for (const c of state.modalCharts) c.destroy();
  state.modalCharts = [];
}

$("#modal-close").addEventListener("click", closeChartModal);
$("#chart-modal").addEventListener("click", (e) => { if (e.target === $("#chart-modal")) closeChartModal(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#chart-modal").classList.contains("hidden")) closeChartModal();
});

document.addEventListener("obs-themechange", () => {
  if (!state.record) return;
  renderRightRail();
  if (!$("#chart-modal").classList.contains("hidden")) openChartModal();
});

boot();
