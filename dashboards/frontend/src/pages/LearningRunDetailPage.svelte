<script>
  import { onMount } from "svelte";
  import { ArrowLeft, ChevronDown, ExternalLink } from "lucide-svelte";
  import LineChart from "../components/LineChart.svelte";
  import Loading from "../components/Loading.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import {
    fetchLearningRun,
    fetchLearningRunEvents,
    fetchLearningRunMonitor,
    fetchLearningRunWorkspace,
    fetchLearningRunWorkspaceFile,
  } from "../lib/api.js";
  import { fmtDate, toEpochSeconds } from "../lib/format.js";
  import {
    fmtGpuHours,
    fmtScore,
    fmtSeconds,
    labPillStatus,
  } from "../lib/learning.js";

  let { runId, gymRuns = [], onBack = () => {} } = $props();

  let detail = $state(null);
  let error = $state(null);
  let loadedRunId = $state(null);

  const POLL_MS = 10000;
  const EVENTS_PAGE = 100;

  // Agent trace state: `events` holds a contiguous window
  // [eventsOffset, eventsOffset + events.length) of the full trace.
  let traceOpen = $state(true);
  let events = $state([]);
  let eventsOffset = $state(0);
  let eventsTotal = $state(0);
  let eventsLoading = $state(false);
  let eventsError = $state(null);
  let eventsLoadedOnce = $state(false);
  // True while the window ends at the live tail; jumping to the start turns
  // off the automatic tail-append so the view stays where the user put it.
  let followTail = $state(true);

  async function load() {
    try {
      const data = await fetchLearningRun(runId);
      detail = data;
      error = data === null ? `Run ${runId} not found.` : null;
    } catch (e) {
      if (!detail) error = e instanceof Error ? e.message : String(e);
    }
    loadedRunId = runId;
    void loadMonitor();
  }

  // ── System telemetry (right rail) ──────────────────────────────────────
  let monitor = $state([]);

  async function loadMonitor() {
    try {
      monitor = await fetchLearningRunMonitor(runId);
    } catch {
      // Rail stays empty; next poll retries.
    }
  }

  let monitorT0 = $derived.by(() => {
    for (const s of monitor) {
      const t = toEpochSeconds(s?.ts);
      if (t) return t;
    }
    return null;
  });

  function monitorSeries(get) {
    if (monitorT0 == null) return [];
    const out = [];
    for (const s of monitor) {
      const t = toEpochSeconds(s?.ts);
      const y = get(s || {});
      if (t && typeof y === "number" && Number.isFinite(y)) {
        out.push({ x: (t - monitorT0) / 3600, y });
      }
    }
    return out;
  }

  let cpuSeries = $derived(monitorSeries((s) => s.cpu_load_1m));
  let memSeries = $derived(monitorSeries((s) => s.mem_used_gib));
  let gpuUtilSeries = $derived(monitorSeries((s) => s.gpu?.util_pct));
  let gpuMemSeries = $derived(monitorSeries((s) => s.gpu?.mem_used_mib));

  // Cumulative GPU-hours from the agent's GPU_LOG ledger (GPU jobs run as
  // separate Modal jobs, so they never appear in the CPU-container samples).
  let gpuHoursSeries = $derived.by(() => {
    const rows = detail?.scores?.gpu_log;
    if (!Array.isArray(rows)) return [];
    const jobs = rows
      .map((r) => ({
        t: toEpochSeconds(r?.ts),
        h: ((Number(r?.seconds) || 0) * (Number(r?.n_gpus) || 0)) / 3600,
      }))
      .filter((p) => p.t)
      .sort((a, b) => a.t - b.t);
    if (!jobs.length) return [];
    const t0 = monitorT0 ?? jobs[0].t;
    let total = 0;
    const points = [{ x: 0, y: 0 }];
    for (const job of jobs) {
      total += job.h;
      points.push({ x: Math.max(0, (job.t - t0) / 3600), y: total });
    }
    return points;
  });

  function fmtElapsedH(point) {
    return `${(point.x ?? 0).toFixed(1)}h`;
  }

  // ── Workspace snapshot browser ─────────────────────────────────────────
  let wsOpen = $state(false);
  let wsTree = $state(null); // null = not loaded; {files: []} once fetched
  let wsMissing = $state(false);
  let wsLoading = $state(false);
  let wsError = $state(null);
  let wsFilter = $state("");
  let wsSelected = $state(null); // {path, size, content, truncated}
  let wsFileLoading = $state(false);

  function toggleWorkspace() {
    wsOpen = !wsOpen;
    if (wsOpen && !wsTree && !wsMissing) void loadWorkspace();
  }

  async function loadWorkspace() {
    wsLoading = true;
    wsError = null;
    try {
      const tree = await fetchLearningRunWorkspace(runId);
      if (tree === null) wsMissing = true;
      else wsTree = tree;
    } catch (e) {
      wsError = e instanceof Error ? e.message : String(e);
    }
    wsLoading = false;
  }

  async function openWorkspaceFile(file) {
    if (!file.inline) return;
    wsFileLoading = true;
    wsSelected = { path: file.path, size: file.size, content: null };
    const entry = await fetchLearningRunWorkspaceFile(runId, file.path);
    if (wsSelected?.path === file.path) {
      wsSelected = entry
        ? {
            path: file.path,
            size: file.size,
            content: entry.content ?? "",
            truncated: !!entry.truncated,
          }
        : null;
    }
    wsFileLoading = false;
  }

  // Nested directory structure from the flat path list; when a filter is
  // set, a flat match list replaces the tree.
  let wsRoot = $derived.by(() => {
    if (!wsTree) return null;
    const root = { dirs: new Map(), files: [] };
    for (const file of wsTree.files || []) {
      const parts = String(file.path).split("/");
      let node = root;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!node.dirs.has(parts[i])) {
          node.dirs.set(parts[i], { dirs: new Map(), files: [] });
        }
        node = node.dirs.get(parts[i]);
      }
      node.files.push({ ...file, name: parts[parts.length - 1] });
    }
    return root;
  });

  let wsMatches = $derived.by(() => {
    if (!wsTree || !wsFilter.trim()) return null;
    const q = wsFilter.trim().toLowerCase();
    return (wsTree.files || [])
      .filter((file) => String(file.path).toLowerCase().includes(q))
      .slice(0, 200);
  });

  function sortedDirs(node) {
    return [...node.dirs.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }

  function sortedFiles(node) {
    return [...node.files].sort((a, b) => a.name.localeCompare(b.name));
  }

  function fmtBytes(n) {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—";
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
    return `${n} B`;
  }

  async function loadEventsTail() {
    eventsLoading = true;
    eventsError = null;
    try {
      const page = await fetchLearningRunEvents(runId, { limit: EVENTS_PAGE });
      events = page.events;
      eventsOffset = page.offset;
      eventsTotal = page.total;
      eventsLoadedOnce = true;
      followTail = true;
    } catch (e) {
      eventsError = e instanceof Error ? e.message : String(e);
    }
    eventsLoading = false;
  }

  async function loadEarlierEvents() {
    if (eventsLoading || eventsOffset <= 0) return;
    eventsLoading = true;
    try {
      const from = Math.max(0, eventsOffset - EVENTS_PAGE);
      const page = await fetchLearningRunEvents(runId, {
        offset: from,
        limit: eventsOffset - from,
      });
      events = [...page.events, ...events];
      eventsOffset = page.offset;
      eventsTotal = page.total;
    } catch (e) {
      eventsError = e instanceof Error ? e.message : String(e);
    }
    eventsLoading = false;
  }

  // Newer events appended while the run is live (the window already ends at
  // the previous tail, so this just extends it).
  async function loadNewerEvents() {
    const end = eventsOffset + events.length;
    if (eventsLoading || end >= eventsTotal) return;
    try {
      const page = await fetchLearningRunEvents(runId, {
        offset: end,
        limit: EVENTS_PAGE,
      });
      if (page.offset === end && page.events.length) {
        events = [...events, ...page.events];
      }
      eventsTotal = page.total;
    } catch {
      // Next poll retries; keep the current window.
    }
  }

  // Replace the window with the first page — where session-start reasoning
  // (CoT) lives, which a tail-first view never reaches.
  async function jumpToStart() {
    if (eventsLoading) return;
    eventsLoading = true;
    eventsError = null;
    try {
      const page = await fetchLearningRunEvents(runId, {
        offset: 0,
        limit: EVENTS_PAGE,
      });
      events = page.events;
      eventsOffset = 0;
      eventsTotal = page.total;
      eventsLoadedOnce = true;
      followTail = false;
    } catch (e) {
      eventsError = e instanceof Error ? e.message : String(e);
    }
    eventsLoading = false;
  }

  // The trace opens at the START of the run (that's where the reasoning
  // lives); "Jump to latest" switches to the live tail.
  function openTrace() {
    traceOpen = !traceOpen;
    if (traceOpen && !eventsLoadedOnce) void jumpToStart();
  }

  onMount(() => {
    const interval = window.setInterval(() => {
      const state = detail?.status?.state || detail?.index_row?.state;
      const live = !detail || labPillStatus(state) === "running";
      if (live) void load();
      if (live && traceOpen && eventsLoadedOnce && followTail) {
        eventsTotal = detail?.status?.num_events ?? eventsTotal;
        void loadNewerEvents();
      }
    }, POLL_MS);
    return () => window.clearInterval(interval);
  });

  $effect(() => {
    if (runId !== loadedRunId) {
      detail = null;
      error = null;
      traceOpen = true;
      logOpen = true;
      expandedLog = new Set();
      logKindFilter = new Set();
      events = [];
      eventsOffset = 0;
      eventsTotal = 0;
      eventsLoadedOnce = false;
      eventsError = null;
      monitor = [];
      wsOpen = false;
      wsTree = null;
      wsMissing = false;
      wsError = null;
      wsFilter = "";
      wsSelected = null;
      void load();
      void jumpToStart();
    }
  });

  let row = $derived(detail?.index_row || {});
  let runState = $derived(detail?.status?.state || row.state || "");
  let learningLog = $derived.by(() => {
    const entries = detail?.scores?.learning_log;
    if (!Array.isArray(entries)) return [];
    return entries
      .map((entry, i) => ({ ...entry, _i: i, _ts: toEpochSeconds(entry?.ts) }))
      .sort((a, b) => (a._ts ?? 0) - (b._ts ?? 0) || a._i - b._i);
  });
  let checkpoints = $derived(
    learningLog.filter(
      (entry) => String(entry.kind || "").toLowerCase() === "checkpoint",
    ),
  );
  let scoreSeries = $derived(
    checkpoints
      .filter(
        (entry) =>
          typeof entry.dev_score === "number" && Number.isFinite(entry.dev_score),
      )
      .map((entry) => ({ x: entry._ts ?? 0, y: entry.dev_score, tag: entry.tag })),
  );
  let results = $derived(
    Array.isArray(detail?.scores?.results) ? detail.scores.results : [],
  );

  const KIND_STYLES = {
    checkpoint: "log-kind-checkpoint",
    submission: "log-kind-submission",
    note: "log-kind-note",
  };

  function kindClass(entry) {
    return KIND_STYLES[String(entry.kind || "").toLowerCase()] || "log-kind-note";
  }

  function kindLabel(entry) {
    return String(entry.kind || "note").toLowerCase();
  }

  // ── Scroll windows ────────────────────────────────────────────────────
  // Both histories render inside a fixed-height scrollable window so a long
  // run doesn't take over the page. The size is a per-section preference
  // persisted in localStorage; "auto" removes the cap.
  function readWindowPref(key, fallback) {
    if (typeof window === "undefined") return fallback;
    const value = window.localStorage.getItem(key);
    if (value === "auto" || (value && Number.isFinite(Number(value)))) return value;
    return fallback;
  }

  let logWindow = $state(readWindowPref("lab-log-window", "480"));
  let traceWindow = $state(readWindowPref("lab-trace-window", "560"));

  $effect(() => {
    try {
      window.localStorage.setItem("lab-log-window", logWindow);
      window.localStorage.setItem("lab-trace-window", traceWindow);
    } catch {
      // Private mode etc. — preference just doesn't persist.
    }
  });

  function windowStyle(value) {
    return value === "auto" ? "" : `max-height:${Number(value)}px;`;
  }

  // ── Research log view state ────────────────────────────────────────────
  // Entries render as collapsed one-line rows; clicking expands one, and the
  // section, kind filters, and expand/collapse-all keep a 40+ entry
  // notebook scannable.
  // Both histories start expanded; their scroll windows keep the page
  // manageable, and each section can still be collapsed from its header.
  let logOpen = $state(true);
  let expandedLog = $state(new Set());
  let logKindFilter = $state(new Set()); // empty = all kinds

  function toggleLogEntry(index) {
    const next = new Set(expandedLog);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    expandedLog = next;
  }

  function toggleLogKind(kind) {
    const next = new Set(logKindFilter);
    if (next.has(kind)) next.delete(kind);
    else next.add(kind);
    logKindFilter = next;
  }

  let logKindCounts = $derived(
    learningLog.reduce((acc, entry) => {
      const kind = kindLabel(entry);
      acc[kind] = (acc[kind] || 0) + 1;
      return acc;
    }, {}),
  );

  let filteredLog = $derived(
    logKindFilter.size
      ? learningLog.filter((entry) => logKindFilter.has(kindLabel(entry)))
      : learningLog,
  );

  let allLogExpanded = $derived(
    filteredLog.length > 0 && filteredLog.every((entry) => expandedLog.has(entry._i)),
  );

  function toggleAllLog() {
    expandedLog = allLogExpanded
      ? new Set()
      : new Set(filteredLog.map((entry) => entry._i));
  }

  // Score movement across the scored entries (checkpoints and submissions
  // both carry dev_score), plus the best-scoring entry for the ★ marker.
  let logScoreDeltas = $derived.by(() => {
    const deltas = new Map();
    let prev = null;
    for (const entry of learningLog) {
      const score = entry.dev_score;
      if (typeof score !== "number" || !Number.isFinite(score)) continue;
      if (prev != null) deltas.set(entry._i, score - prev);
      prev = score;
    }
    return deltas;
  });

  let bestLogEntryIndex = $derived.by(() => {
    let best = null;
    let bestScore = -Infinity;
    for (const entry of learningLog) {
      const score = entry.dev_score;
      if (typeof score === "number" && Number.isFinite(score) && score > bestScore) {
        bestScore = score;
        best = entry._i;
      }
    }
    return best;
  });

  function fmtDelta(delta) {
    const sign = delta > 0 ? "+" : "";
    return `${sign}${delta.toFixed(3)}`;
  }

  // Best-effort link from a checkpoint log entry to a gym training run: the
  // agent trains through the gym SDK, so its tag or checkpoint path often
  // shows up in the gym run's id or TrainResult paths.
  function matchGymRun(entry) {
    const tag = String(entry.tag || "").trim();
    const modelPath = String(entry.model_path || "").trim();
    if (!tag && !modelPath) return null;
    return (
      gymRuns.find((run) => {
        const result = run.train_result || {};
        const haystacks = [
          run.run_id,
          result.training_run_id,
          result.checkpoint_dir,
          result.model_path,
        ].map((v) => String(v || ""));
        if (tag && haystacks.some((h) => h.includes(tag))) return true;
        return !!modelPath && haystacks.some((h) => h && modelPath.includes(h));
      }) || null
    );
  }

  function fmtChartDate(row) {
    return fmtDate(row.x);
  }

  function entryResult(entry) {
    if (typeof entry.dev_score === "number" && Number.isFinite(entry.dev_score)) {
      return `dev ${fmtScore(entry.dev_score)}`;
    }
    return entry.result ? String(entry.result) : "";
  }

  // ── Trace block rendering ──────────────────────────────────────────────
  // Events carry posttrainbench-style blocks: thinking | text | tool_use |
  // tool_result (see observatory/schema.py).

  function toolCommand(block) {
    const input = block?.input;
    if (input && typeof input.command === "string") return input.command;
    try {
      return JSON.stringify(input ?? {});
    } catch {
      return "";
    }
  }

  function toolResultText(block) {
    const content = block?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((part) =>
          typeof part === "string" ? part : String(part?.text ?? ""),
        )
        .join("\n");
    }
    if (content == null) return "";
    try {
      return JSON.stringify(content);
    } catch {
      return "";
    }
  }

  function eventTime(event) {
    const seconds = toEpochSeconds(event?.ts);
    return seconds ? fmtDate(seconds) : "";
  }

  // Message text carried by system events (errors, init, codex item noise) —
  // without this a run that is dying in a reconnect loop renders as bare
  // "system · error" labels and looks empty instead of broken.
  function systemMessage(event) {
    const raw = event?.raw;
    if (!raw || typeof raw !== "object") return "";
    if (typeof raw.message === "string") return raw.message;
    if (raw.item && typeof raw.item.message === "string") return raw.item.message;
    return "";
  }

  // Display label for an event. The data keeps chat-API role names
  // (assistant/user — tool results ride user-role messages on the wire),
  // but these runs are fully autonomous, so the UI says what actually
  // happened: the agent acted, or the harness returned tool output. Same
  // convention as the observatory run view.
  function eventLabel(event) {
    if (event.type === "assistant") return "agent";
    if (event.type === "user") {
      const blocks = event.blocks || [];
      return blocks.some((block) => block.type === "tool_result")
        ? "tool output"
        : "input";
    }
    if (event.type === "result") return "session end";
    return event.type || "event";
  }
</script>

<section class="flex flex-col gap-[20px] p-[0_24px_24px] max-[900px]:p-[0_16px_24px] min-w-0">
  <div class="flex items-center justify-between gap-[12px] flex-wrap">
    <button
      class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[6px_10px] [font:inherit] text-[13px] font-medium text-(--muted) bg-transparent cursor-pointer ghost-hover"
      onclick={onBack}
    >
      <ArrowLeft size={14} strokeWidth={2.1} />
      <span>All learning runs</span>
    </button>
    {#if detail?.obs_url}
      <a
        class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[6px_10px] no-underline text-(--muted) text-[13px] font-medium ghost-hover"
        href={detail.obs_url}
        target="_blank"
        rel="noopener noreferrer"
      >
        <span>Trace &amp; telemetry in Observatory</span>
        <ExternalLink size={13} strokeWidth={2.1} />
      </a>
    {/if}
  </div>

  {#if !detail && !error}
    <div class="page-empty"><Loading /> Loading learning run…</div>
  {:else if error && !detail}
    <div class="page-empty">Failed to load: {error}</div>
  {:else}
    <header class="flex items-center gap-[12px] flex-wrap">
      <h2 class="m-0 text-(--text-bright) text-[18px] font-medium [font-family:var(--font-mono)] break-all">
        {row.run_id || runId}
      </h2>
      <StatusPill status={labPillStatus(runState)} label={runState || null} />
    </header>

    <div class="info-band">
      <div class="rail-card">
        <h4 class="rail-title">overview</h4>
        <dl class="m-0 flex flex-col gap-[5px]">
          <div class="rail-row"><dt>task</dt><dd>{row.task || "\u2014"}</dd></div>
          <div class="rail-row"><dt>scaffold</dt><dd class="[font-family:var(--font-mono)] text-[11.5px]">{row.scaffold || "\u2014"}</dd></div>
          <div class="rail-row"><dt>track</dt><dd>{row.track || "\u2014"}</dd></div>
          <div class="rail-row"><dt>budget</dt><dd>{row.time_budget_h ? `${row.time_budget_h}h` : "\u2014"}</dd></div>
          <div class="rail-row"><dt>started</dt><dd><TimeAgo timestamp={row.launched_at} showJustNow falsyRepresentation="\u2014" /></dd></div>
          <div class="rail-row"><dt>duration</dt><dd>{fmtSeconds(row.duration_s)}</dd></div>
        </dl>
      </div>
      <div class="rail-card">
        <h4 class="rail-title">results</h4>
        <dl class="m-0 flex flex-col gap-[5px]">
          <div class="rail-row"><dt>best dev</dt><dd class="text-(--green,#4ade80)">{fmtScore(row.best_dev_score)}{#if row.best_tag}<span class="text-(--muted)"> ({row.best_tag})</span>{/if}</dd></div>
          <div class="rail-row"><dt>checkpoints</dt><dd>{checkpoints.length}</dd></div>
          <div class="rail-row"><dt>log entries</dt><dd>{learningLog.length}</dd></div>
          <div class="rail-row"><dt>trace events</dt><dd>{row.num_events ?? "\u2014"}</dd></div>
          <div class="rail-row"><dt>gpu hours</dt><dd>{fmtGpuHours(row.gpu_hours)}</dd></div>
          {#if wsTree}
            <div class="rail-row"><dt>workspace</dt><dd>{wsTree.total_files} files</dd></div>
          {/if}
        </dl>
      </div>
    </div>

    <section class="flex flex-col gap-[10px]">
      <h3
        class="m-0 text-(--text-bright) text-[15px] font-medium"
        title="The agent's CPU container \u2014 GPU jobs run as separate Modal jobs; their hours are in the cumulative gpu-hours chart and runs/GPU_LOG.jsonl"
      >System</h3>
      {#if !monitor.length && gpuHoursSeries.length < 2}
        <div class="page-empty">No telemetry samples for this run.</div>
      {:else}
        <div class="charts-row">
          {#if cpuSeries.length >= 2}
            <div class="rail-card"><LineChart title="cpu load 1m" data={cpuSeries} height={92} formatX={fmtElapsedH} formatY={(v) => v.toFixed(2)} /></div>
          {/if}
          {#if memSeries.length >= 2}
            <div class="rail-card"><LineChart title="mem used GiB" data={memSeries} height={92} formatX={fmtElapsedH} formatY={(v) => v.toFixed(2)} /></div>
          {/if}
          {#if gpuUtilSeries.length >= 2}
            <div class="rail-card"><LineChart title="gpu util %" data={gpuUtilSeries} height={92} formatX={fmtElapsedH} formatY={(v) => v.toFixed(0)} /></div>
          {/if}
          {#if gpuMemSeries.length >= 2}
            <div class="rail-card"><LineChart title="gpu mem MiB" data={gpuMemSeries} height={92} formatX={fmtElapsedH} formatY={(v) => v.toFixed(0)} /></div>
          {/if}
          {#if gpuHoursSeries.length >= 2}
            <div class="rail-card"><LineChart title="gpu hours (cumulative)" data={gpuHoursSeries} height={92} color="var(--yellow, #fbbf24)" formatX={fmtElapsedH} formatY={(v) => v.toFixed(2)} /></div>
          {/if}
        </div>
      {/if}
    </section>

    <section class="flex flex-col gap-[10px]">
      <div class="flex items-center justify-between gap-[12px] flex-wrap">
        <button
          class="flex items-center gap-[8px] [border:0] [background:none] p-0 [font:inherit] text-left cursor-pointer"
          onclick={toggleWorkspace}
          aria-expanded={wsOpen}
        >
          <h3 class="m-0 text-(--text-bright) text-[15px] font-medium">Workspace</h3>
          <span class="text-[12px] text-(--muted)">
            {wsTree
              ? `${wsTree.total_files} files · ${fmtBytes(wsTree.total_bytes)}`
              : wsMissing
                ? "no snapshot"
                : ""}
          </span>
          <span class="text-(--muted) transition-transform" class:rotate-180={wsOpen}>
            <ChevronDown size={15} strokeWidth={2.1} />
          </span>
        </button>
        {#if wsOpen && wsTree}
          <input
            class="search-input max-w-[280px]"
            type="search"
            placeholder="Filter files…"
            bind:value={wsFilter}
            aria-label="Filter workspace files"
          />
        {/if}
      </div>
      {#if wsOpen}
        {#if wsLoading && !wsTree}
          <div class="page-empty"><Loading /> Loading workspace snapshot…</div>
        {:else if wsMissing}
          <div class="page-empty">
            No workspace snapshot was uploaded for this run (the observatory
            ingests one with --archive-workspace or the live watcher).
          </div>
        {:else if wsError}
          <div class="page-empty">Failed to load workspace: {wsError}</div>
        {:else if wsTree}
          <div class="ws-grid">
            <div class="scroll-window ws-tree" style="max-height:480px;">
              {#if wsMatches}
                {#each wsMatches as file (file.path)}
                  <button
                    class="ws-file"
                    class:ws-file-active={wsSelected?.path === file.path}
                    disabled={!file.inline}
                    onclick={() => openWorkspaceFile(file)}
                    title={file.inline ? file.path : `${file.path} — content not in snapshot (over the 64 KB per-file inline cap); the full file is in the observatory workspace archive`}
                  >
                    <span class="ws-file-name">{file.path}</span>
                    <span class="ws-file-size">{fmtBytes(file.size)}</span>
                  </button>
                {/each}
                {#if !wsMatches.length}
                  <div class="p-[10px] text-[12px] text-(--muted)">No files match.</div>
                {/if}
              {:else if wsRoot}
                {@render wsDirNode(wsRoot)}
              {/if}
            </div>
            <div class="scroll-window ws-viewer" style="max-height:480px;">
              {#if wsSelected}
                <div class="ws-viewer-head">
                  <span class="[font-family:var(--font-mono)] text-[11.5px] text-(--text-bright) break-all">{wsSelected.path}</span>
                  <span class="text-[11px] text-(--muted) whitespace-nowrap">{fmtBytes(wsSelected.size)}{wsSelected.truncated ? " · truncated" : ""}</span>
                </div>
                {#if wsFileLoading && wsSelected.content == null}
                  <div class="p-[10px]"><Loading /></div>
                {:else}
                  <pre class="ws-content">{wsSelected.content}</pre>
                {/if}
              {:else}
                <div class="p-[12px] text-[12px] text-(--muted)">
                  Select a file to view its snapshot. Grayed-out files were too
                  large to inline; the full archive lives in the observatory.
                </div>
              {/if}
            </div>
          </div>
        {/if}
      {/if}
    </section>

    {#if scoreSeries.length >= 2}
      <section class="rounded-[6px] [background:rgba(255,255,255,0.03)] p-[16px]">
        <LineChart
          title="Dev score over checkpoints"
          data={scoreSeries}
          height={160}
          formatX={fmtChartDate}
          formatY={(value) => fmtScore(value)}
        />
      </section>
    {/if}

    <section class="flex flex-col gap-[10px]">
      <div class="flex items-center justify-between gap-[12px] flex-wrap">
        <button
          class="flex items-center gap-[8px] [border:0] [background:none] p-0 [font:inherit] text-left cursor-pointer"
          onclick={() => (logOpen = !logOpen)}
          aria-expanded={logOpen}
        >
          <h3 class="m-0 text-(--text-bright) text-[15px] font-medium">Learning research log</h3>
          <span class="text-[12px] text-(--muted)">{learningLog.length} entries</span>
          <span class="text-(--muted) transition-transform" class:rotate-180={logOpen}>
            <ChevronDown size={15} strokeWidth={2.1} />
          </span>
        </button>
        {#if logOpen && learningLog.length}
          <div class="flex items-center gap-[6px] flex-wrap">
            {#each ["checkpoint", "submission", "note"] as kind (kind)}
              {#if logKindCounts[kind]}
                <button
                  class={`log-filter-chip ${kindClass({ kind })}`}
                  class:log-filter-off={logKindFilter.size && !logKindFilter.has(kind)}
                  onclick={() => toggleLogKind(kind)}
                  aria-pressed={!logKindFilter.size || logKindFilter.has(kind)}
                >
                  {kind} {logKindCounts[kind]}
                </button>
              {/if}
            {/each}
            <button class="trace-page-button" onclick={toggleAllLog}>
              {allLogExpanded ? "Collapse all" : "Expand all"}
            </button>
            <label class="window-select-label">
              window
              <select class="window-select" bind:value={logWindow}>
                <option value="320">S</option>
                <option value="480">M</option>
                <option value="720">L</option>
                <option value="auto">Full</option>
              </select>
            </label>
          </div>
        {/if}
      </div>
      {#if logOpen}
        {#if !learningLog.length}
          <div class="page-empty">
            {#if labPillStatus(runState) === "running"}
              No log entries yet — the run is live and this refreshes automatically.
              Agents usually write their first entry after the baseline eval.
            {:else}
              This run never wrote to runs/LEARNING_LOG.jsonl (short, smoke, or
              failed runs don't get far enough to experiment). Use the "Log
              entries" column on the runs list to find runs with a notebook.
            {/if}
          </div>
        {:else}
          <ol
            class="scroll-window m-0 p-0 list-none flex flex-col rounded-[6px] [background:rgba(255,255,255,0.03)]"
            style={windowStyle(logWindow)}
          >
            {#each filteredLog as entry (entry._i)}
              {@const expanded = expandedLog.has(entry._i)}
              {@const delta = logScoreDeltas.get(entry._i)}
              {@const isBest = bestLogEntryIndex === entry._i}
              {@const gymRun = expanded && kindLabel(entry) === "checkpoint" ? matchGymRun(entry) : null}
              <li class="log-row-item">
                <button class="log-row" onclick={() => toggleLogEntry(entry._i)} aria-expanded={expanded}>
                  <span class={`log-kind-pill ${kindClass(entry)}`}>{kindLabel(entry)}</span>
                  {#if entry.tag}
                    <span class="[font-family:var(--font-mono)] text-[12px] text-(--text-bright) whitespace-nowrap">{entry.tag}</span>
                  {/if}
                  {#if typeof entry.dev_score === "number" && Number.isFinite(entry.dev_score)}
                    <span class="text-[12px] [font-variant-numeric:tabular-nums] text-(--green,#4ade80) whitespace-nowrap">{fmtScore(entry.dev_score)}</span>
                    {#if delta != null && delta !== 0}
                      <span class="text-[11.5px] [font-variant-numeric:tabular-nums] whitespace-nowrap" class:text-(--green,#4ade80)={delta > 0} class:text-(--red,#f87171)={delta < 0}>{fmtDelta(delta)}</span>
                    {/if}
                    {#if isBest}
                      <span class="log-best">★ best</span>
                    {/if}
                  {/if}
                  <span class="log-what" class:log-what-open={expanded}>{entry.what || entry.result || ""}</span>
                  <span class="text-[11.5px] text-(--muted) ml-auto whitespace-nowrap" title={entry.ts || ""}>
                    {entry._ts ? fmtDate(entry._ts) : "—"}
                  </span>
                  <span class="text-(--muted) transition-transform shrink-0" class:rotate-180={expanded}>
                    <ChevronDown size={13} strokeWidth={2.1} />
                  </span>
                </button>
                {#if expanded}
                  <div class="log-row-detail">
                    {#if entry.what}
                      <div class="text-[13.5px] leading-[20px] text-(--text) whitespace-pre-wrap [overflow-wrap:anywhere]">{entry.what}</div>
                    {/if}
                    {#if entry.why}
                      <div class="text-[12.5px] leading-[18px] text-(--muted)"><span class="uppercase text-[10.5px] tracking-[0.06em]">why</span> {entry.why}</div>
                    {/if}
                    {#if entry.result && entry.what !== entry.result}
                      <div class="text-[12.5px] leading-[18px] text-(--muted)"><span class="uppercase text-[10.5px] tracking-[0.06em]">result</span> {entry.result}</div>
                    {/if}
                    {#if entry.model_path || (Array.isArray(entry.artifacts) && entry.artifacts.length) || gymRun}
                      <div class="flex items-center gap-[8px] flex-wrap pt-[2px]">
                        {#if entry.model_path}
                          <span class="log-artifact" title={entry.model_path}>{entry.model_path}</span>
                        {/if}
                        {#each entry.artifacts || [] as artifact (artifact)}
                          <span class="log-artifact" title={artifact}>{artifact}</span>
                        {/each}
                        {#if gymRun}
                          <a
                            class="inline-flex items-center gap-[4px] text-[12px] text-(--green,#4ade80) no-underline hover:underline"
                            href={`/training/${encodeURIComponent(gymRun.run_id)}`}
                          >
                            training run {gymRun.run_id}
                            <ExternalLink size={11} strokeWidth={2.1} />
                          </a>
                        {/if}
                      </div>
                    {/if}
                  </div>
                {/if}
              </li>
            {/each}
          </ol>
        {/if}
      {/if}
    </section>

    <section class="flex flex-col gap-[10px]">
      <div class="flex items-center justify-between gap-[12px] flex-wrap">
        <button
          class="flex items-center gap-[8px] [border:0] [background:none] p-0 [font:inherit] text-left cursor-pointer"
          onclick={openTrace}
          aria-expanded={traceOpen}
        >
          <h3 class="m-0 text-(--text-bright) text-[15px] font-medium">Agent trace</h3>
          <span class="text-[12px] text-(--muted)">
            {eventsLoadedOnce ? `${eventsTotal} events` : `${row.num_events ?? "—"} events`}
          </span>
          <span class="text-(--muted) transition-transform" class:rotate-180={traceOpen}>
            <ChevronDown size={15} strokeWidth={2.1} />
          </span>
        </button>
        <div class="flex items-center gap-[8px] flex-wrap">
          {#if traceOpen && events.length}
            <label class="window-select-label">
              window
              <select class="window-select" bind:value={traceWindow}>
                <option value="400">S</option>
                <option value="560">M</option>
                <option value="800">L</option>
                <option value="auto">Full</option>
              </select>
            </label>
          {/if}
          <a
            class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_9px] no-underline text-(--muted) text-[12px] font-medium ghost-hover"
            href={`/api/learning-runs/${encodeURIComponent(runId)}/trajectory?download=true`}
            title="Download the full trace as a Harbor ATIF-v1.7 trajectory.json"
          >
            <span>Download ATIF trajectory</span>
            <ExternalLink size={11} strokeWidth={2.1} />
          </a>
        </div>
      </div>
      {#if traceOpen}
        {#if eventsError && !events.length}
          <div class="page-empty">Failed to load trace: {eventsError}</div>
        {:else if eventsLoading && !events.length}
          <div class="page-empty"><Loading /> Loading trace…</div>
        {:else if !events.length}
          <div class="page-empty">No trace events yet.</div>
        {:else}
          <div class="flex items-center gap-[8px] flex-wrap">
            {#if eventsOffset > 0}
              <button class="trace-page-button" onclick={jumpToStart} disabled={eventsLoading}>
                Jump to start
              </button>
              <button class="trace-page-button" onclick={loadEarlierEvents} disabled={eventsLoading}>
                {eventsLoading ? "Loading…" : `Load earlier events (${eventsOffset} before this)`}
              </button>
            {/if}
            {#if eventsOffset + events.length < eventsTotal}
              <button class="trace-page-button" onclick={loadEventsTail} disabled={eventsLoading}>
                Jump to latest
              </button>
            {/if}
          </div>
          <ol class="scroll-window m-0 p-0 list-none flex flex-col gap-[8px]" style={windowStyle(traceWindow)}>
            {#each events as event (event.i ?? event.uuid ?? JSON.stringify(event).slice(0, 80))}
              <li class="rounded-[6px] [background:rgba(255,255,255,0.03)] p-[10px_14px] flex flex-col gap-[6px]">
                <div class="flex items-center gap-[10px] text-[11.5px] text-(--muted)">
                  <span class="[font-variant-numeric:tabular-nums]">#{event.i}</span>
                  <span class={`trace-type trace-type-${event.type}`}>{eventLabel(event)}{event.subtype ? ` · ${event.subtype}` : ""}</span>
                  {#if event.turn != null}
                    <span>turn {event.turn}</span>
                  {/if}
                  <span class="ml-auto whitespace-nowrap" title={event.ts || ""}>{eventTime(event)}</span>
                </div>
                {#each event.blocks || [] as block, blockIndex (blockIndex)}
                  {#if block.type === "text"}
                    <div class="text-[13px] leading-[19px] text-(--text) whitespace-pre-wrap [overflow-wrap:anywhere]">{block.text}</div>
                  {:else if block.type === "thinking"}
                    <details class="trace-details">
                      <summary class="trace-thinking-summary">✦ thinking ({(block.thinking || "").length} chars)</summary>
                      <div class="trace-thinking-text pt-[6px]">{block.thinking}</div>
                    </details>
                  {:else if block.type === "tool_use"}
                    <div class="trace-tool-use">
                      <span class="text-(--green,#4ade80)">$</span>
                      <span class="whitespace-pre-wrap [overflow-wrap:anywhere]">{toolCommand(block)}</span>
                    </div>
                  {:else if block.type === "tool_result"}
                    {@const text = toolResultText(block)}
                    <details class="trace-details">
                      <summary class="trace-summary" class:trace-summary-error={block.is_error}>
                        {block.is_error ? "error output" : "output"} ({text.length} chars)
                      </summary>
                      <pre class="trace-output">{text.length > 20000 ? text.slice(0, 20000) + "\n… (truncated)" : text}</pre>
                    </details>
                  {/if}
                {/each}
                {#if event.type === "result" && event.result}
                  <div class="text-[13px] leading-[19px] text-(--text) whitespace-pre-wrap [overflow-wrap:anywhere]">{event.result}</div>
                {/if}
                {#if event.type === "system" && systemMessage(event)}
                  <div
                    class="text-[12.5px] leading-[18px] whitespace-pre-wrap [overflow-wrap:anywhere]"
                    class:text-(--red,#f87171)={event.subtype === "error"}
                    class:text-(--muted)={event.subtype !== "error"}
                  >{systemMessage(event)}</div>
                {/if}
              </li>
            {/each}
          </ol>
          {#if eventsOffset + events.length < eventsTotal}
            <button class="trace-page-button" onclick={loadNewerEvents} disabled={eventsLoading}>
              {eventsLoading ? "Loading…" : `Load newer events (${eventsTotal - eventsOffset - events.length} after this)`}
            </button>
          {/if}
          <div class="text-[12px] text-(--muted)">
            Showing events {eventsOffset}–{eventsOffset + events.length - 1} of {eventsTotal}.
            {#if labPillStatus(runState) === "running"}New events append automatically while the run is live.{/if}
          </div>
        {/if}
      {/if}
    </section>


    {#if results.length}
      <section class="flex flex-col gap-[10px]">
        <h3 class="m-0 text-(--text-bright) text-[15px] font-medium">Eval results</h3>
        <div class="table-wrap">
          <table class="minimal-table">
            <thead>
              <tr>
                <th>Tag</th>
                <th>Split</th>
                <th>Budget</th>
                <th>Mean</th>
                <th>CI95</th>
                <th>n</th>
                <th>Canonical</th>
              </tr>
            </thead>
            <tbody>
              {#each results as result, i (`${result.tag}-${result.split}-${result.budget}-${i}`)}
                <tr>
                  <td class="[font-family:var(--font-mono)] text-[12px]">{result.tag || "—"}</td>
                  <td>{result.split || "—"}</td>
                  <td class="[font-variant-numeric:tabular-nums]">{result.budget ?? "—"}</td>
                  <td class="[font-variant-numeric:tabular-nums]">{fmtScore(result.mean)}</td>
                  <td class="[font-variant-numeric:tabular-nums]">
                    {Array.isArray(result.bootstrap_ci95)
                      ? `[${fmtScore(result.bootstrap_ci95[0])}, ${fmtScore(result.bootstrap_ci95[1])}]`
                      : "—"}
                  </td>
                  <td class="[font-variant-numeric:tabular-nums]">{result.n ?? "—"}</td>
                  <td>{result.canonical === true ? "yes" : result.canonical === false ? "no" : "—"}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}
  {/if}
</section>

{#snippet wsDirNode(node)}
  {#each sortedDirs(node) as [name, child] (name)}
    <details class="ws-dir">
      <summary class="ws-dir-name">{name}/</summary>
      <div class="ws-indent">
        {@render wsDirNode(child)}
      </div>
    </details>
  {/each}
  {#each sortedFiles(node) as file (file.path)}
    <button
      class="ws-file"
      class:ws-file-active={wsSelected?.path === file.path}
      disabled={!file.inline}
      onclick={() => openWorkspaceFile(file)}
      title={file.inline ? file.path : `${file.path} — content not in snapshot (over the 64 KB per-file inline cap); the full file is in the observatory workspace archive`}
    >
      <span class="ws-file-name">{file.name}</span>
      <span class="ws-file-size">{fmtBytes(file.size)}</span>
    </button>
  {/each}
{/snippet}

<style>
  .log-kind-pill {
    display: inline-flex;
    align-items: center;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 11px;
    line-height: 16px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid var(--border, #2f2f2f);
  }
  .log-kind-checkpoint {
    color: var(--green, #4ade80);
    border-color: color-mix(in srgb, var(--green, #4ade80) 45%, transparent);
    background: color-mix(in srgb, var(--green, #4ade80) 10%, transparent);
  }
  .log-kind-submission {
    color: var(--blue, #60a5fa);
    border-color: color-mix(in srgb, var(--blue, #60a5fa) 45%, transparent);
    background: color-mix(in srgb, var(--blue, #60a5fa) 10%, transparent);
  }
  .log-kind-note {
    color: var(--muted, #9ca3af);
    background: color-mix(in srgb, var(--panel-alt, #1f1f1f) 70%, transparent);
  }
  .log-artifact {
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 1px 7px;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 999px;
    background: color-mix(in srgb, var(--panel-alt, #1f1f1f) 70%, transparent);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    color: var(--muted, #9ca3af);
  }

  .trace-type {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 10.5px;
  }
  .trace-type-assistant {
    color: var(--green, #4ade80);
  }
  .trace-type-user {
    color: var(--blue, #60a5fa);
  }
  .log-row-item + .log-row-item {
    border-top: 1px solid var(--border, #2f2f2f);
  }
  .log-row {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 8px 14px;
    border: 0;
    background: none;
    font: inherit;
    text-align: left;
    cursor: pointer;
    min-width: 0;
  }
  .log-row:hover {
    background: color-mix(in srgb, white 3%, transparent);
  }
  .log-what {
    min-width: 0;
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
    line-height: 19px;
    color: var(--text, #c9c9c9);
  }
  .log-what-open {
    visibility: hidden; /* full text shows in the detail below */
  }
  .log-row-detail {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0 14px 12px 14px;
  }
  .log-best {
    font-size: 10.5px;
    line-height: 16px;
    padding: 1px 7px;
    border-radius: 999px;
    white-space: nowrap;
    color: var(--yellow, #fbbf24);
    border: 1px solid color-mix(in srgb, var(--yellow, #fbbf24) 45%, transparent);
    background: color-mix(in srgb, var(--yellow, #fbbf24) 10%, transparent);
  }
  .log-filter-chip {
    font: inherit;
    cursor: pointer;
    font-size: 11px;
    line-height: 16px;
    padding: 2px 9px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
    border: 1px solid var(--border, #2f2f2f);
    background: none;
  }
  .log-filter-off {
    opacity: 0.35;
  }

  .trace-thinking-summary {
    cursor: pointer;
    user-select: none;
    font-size: 12px;
    line-height: 18px;
    font-style: italic;
    color: var(--green, #4ade80);
  }
  .trace-thinking-summary:hover {
    color: color-mix(in srgb, var(--green, #4ade80) 70%, white);
  }
  .trace-thinking-text {
    font-size: 12.5px;
    line-height: 19px;
    color: color-mix(in srgb, var(--text, #c9c9c9) 88%, white);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .window-select-label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: var(--muted, #9ca3af);
  }
  .window-select {
    font: inherit;
    font-size: 11.5px;
    color: var(--text, #c9c9c9);
    background: color-mix(in srgb, var(--panel-alt, #1f1f1f) 70%, transparent);
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 3px 6px;
    cursor: pointer;
  }
  .scroll-window {
    overflow-y: auto;
    overscroll-behavior: contain;
  }

  .info-band {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px;
    align-items: start;
  }
  .charts-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
    align-items: start;
  }
  .rail-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .rail-title {
    margin: 0;
    font-size: 11px;
    line-height: 16px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--muted, #9ca3af);
  }
  .rail-row {
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr);
    align-items: baseline;
    gap: 10px;
    font-size: 12px;
    line-height: 17px;
  }
  .rail-row dt {
    color: var(--muted, #9ca3af);
    white-space: nowrap;
  }
  .rail-row dd {
    margin: 0;
    color: var(--text, #c9c9c9);
    min-width: 0;
    overflow-wrap: anywhere;
    font-variant-numeric: tabular-nums;
  }

  .ws-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.5fr);
    gap: 12px;
    align-items: start;
  }
  @media (max-width: 900px) {
    .ws-grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .ws-tree,
  .ws-viewer {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    padding: 8px;
  }
  .ws-dir {
    min-width: 0;
  }
  .ws-dir-name {
    cursor: pointer;
    user-select: none;
    font-size: 12px;
    line-height: 20px;
    color: var(--text-bright, #e5e5e5);
    padding: 1px 6px;
    border-radius: 4px;
  }
  .ws-dir-name:hover {
    background: color-mix(in srgb, white 4%, transparent);
  }
  .ws-indent {
    margin-left: 14px;
    border-left: 1px solid var(--border, #2f2f2f);
    padding-left: 6px;
  }
  .ws-file {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    border: 0;
    background: none;
    font: inherit;
    text-align: left;
    cursor: pointer;
    padding: 1px 6px;
    border-radius: 4px;
    min-width: 0;
  }
  .ws-file:hover:not(:disabled) {
    background: color-mix(in srgb, white 4%, transparent);
  }
  .ws-file:disabled {
    cursor: default;
    opacity: 0.45;
  }
  .ws-file-active {
    background: color-mix(in srgb, var(--green, #4ade80) 12%, transparent);
  }
  .ws-file-name {
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 19px;
    color: var(--text, #c9c9c9);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ws-file-size {
    font-size: 10.5px;
    color: var(--muted, #9ca3af);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .ws-viewer-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    padding: 2px 6px 8px;
    border-bottom: 1px solid var(--border, #2f2f2f);
  }
  .ws-content {
    margin: 0;
    padding: 8px 6px 2px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 17px;
    color: var(--text, #c9c9c9);
  }

  .trace-tool-use {
    display: flex;
    gap: 8px;
    align-items: baseline;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 17px;
    color: var(--text-bright, #e5e5e5);
    background: color-mix(in srgb, black 30%, transparent);
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 6px;
    padding: 7px 10px;
  }
  .trace-details {
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 6px;
    padding: 5px 10px;
  }
  .trace-summary {
    cursor: pointer;
    font-size: 11.5px;
    color: var(--muted, #9ca3af);
    user-select: none;
  }
  .trace-summary-error {
    color: var(--red, #f87171);
  }
  .trace-output {
    margin: 6px 0 2px;
    max-height: 320px;
    overflow: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 16px;
    color: var(--text, #c9c9c9);
  }
  .trace-page-button {
    align-self: flex-start;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 6px 10px;
    font: inherit;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--muted, #9ca3af);
    background: transparent;
    cursor: pointer;
  }
  .trace-page-button:disabled {
    opacity: 0.6;
    cursor: default;
  }
</style>
