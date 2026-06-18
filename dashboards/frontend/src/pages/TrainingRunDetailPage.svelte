<script>
  import { ArrowLeft, ChevronLeft, ChevronRight, Download, ExternalLink, Minimize2, X } from "lucide-svelte";
  import Tabs from "../components/Tabs.svelte";
  import RunSummary from "../components/RunSummary.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import SampleTimeline from "../components/SampleTimeline.svelte";
  import ConversationView from "../components/ConversationView.svelte";
  import { fetchRunRollouts, fetchRollout } from "../lib/api.js";
  import { parseAnsi } from "../lib/ansi.js";

  let {
    runId,
    allRuns,
    modelName,
    getStatus,
    getFrameworkStatus,
    showFrameworkStatus,
    fmtDuration,
    onBack,
    // "Collapse" drops the full detail page back to the list as a summary drawer.
    onCollapse,
    // When rendered inside the expanded run drawer the surrounding UI already
    // shows the header/title/summary, so we hide them and render only the
    // unique rollouts + logs content.
    embedded = false,
  } = $props();

  // `run` is looked up from the parent's `allRuns`, which is replaced wholesale
  // every 5s by the auto-refresh. A transient poll that doesn't include this
  // run (pagination, a flaky fetch, the run briefly aging out) would otherwise
  // flip `run` to null — flashing "Loading run …" and tearing down the live log
  // stream. Latch the last resolved run instead: only drop to null when the
  // *runId itself* changes to one we haven't resolved yet.
  let run = $state(null);
  let latchedRunId = null; // plain (non-reactive) so the effect doesn't self-trigger
  $effect(() => {
    const id = runId;
    const match = (allRuns || []).find((r) => r.run_id === id) || null;
    if (match) {
      run = match; // fresh data for this run
      latchedRunId = id;
    } else if (latchedRunId !== id) {
      run = null; // navigated to a different run we haven't loaded yet
    }
    // match == null but we already latched this runId → keep the latched copy.
  });

  // Status as a primitive so effects depending on it don't re-run every time
  // the auto-refresh hands us a new `run` object with the same status (which
  // would otherwise tear down and rebuild the log stream, flashing the tail).
  let runStatus = $derived(String(run?.status || "").toLowerCase());

  // Active tab: "summary" | "rollouts" | "logs". Each tab loads only its own
  // data — rollout summaries for summary/rollouts, the log stream for logs.
  let activeTab = $state("summary");

  function formatMean(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(3);
  }

  // ── Rollouts (auto-refresh while run is running) ─────────────────────
  let rolloutSummaries = $state([]);
  let rolloutsLoading = $state(false);
  let rolloutsError = $state("");
  let expandedRolloutId = $state(null);
  let expandedRollout = $state(null);
  let expandedRolloutLoading = $state(false);

  // Per-step sample view: a histogram of sample scores. Clicking a bar opens
  // a single-sample viewer scoped to that bucket; ←/→ step through it.
  const BUCKET_COUNT = 12;
  let activeBucket = $state(null); // histogram bucket index, or null
  let activeSamplePos = $state(0); // position within the active bucket's list

  // Bucket the expanded rollout's samples by score.
  let sampleDist = $derived.by(() => {
    const samples = expandedRollout?.samples || [];
    if (!samples.length) return null;
    const scores = samples.map((s) => Number(s.score) || 0);
    const lo = Math.min(...scores);
    const hi = Math.max(...scores);
    // When every sample scored the same, a single bucket reads clearer than a
    // lone bar pinned to one edge.
    const count = lo === hi ? 1 : BUCKET_COUNT;
    const span = hi - lo || 1;
    const buckets = Array.from({ length: count }, () => []);
    samples.forEach((s, i) => {
      const score = Number(s.score) || 0;
      let b = count === 1 ? 0 : Math.floor(((score - lo) / span) * count);
      b = Math.max(0, Math.min(count - 1, b));
      buckets[b].push(i);
    });
    const maxCount = Math.max(...buckets.map((b) => b.length), 1);
    return { lo, hi, count, span, buckets, maxCount, total: samples.length };
  });

  function bucketRange(b) {
    const d = sampleDist;
    if (!d) return "";
    if (d.count === 1) return formatMean(d.lo);
    const step = d.span / d.count;
    return `${formatMean(d.lo + b * step)}–${formatMean(d.lo + (b + 1) * step)}`;
  }

  function openBucket(b) {
    const d = sampleDist;
    if (!d || !d.buckets[b]?.length) return;
    activeBucket = b;
    activeSamplePos = 0;
  }

  function closeBucket() {
    activeBucket = null;
    activeSamplePos = 0;
  }

  function stepSample(delta) {
    const d = sampleDist;
    if (!d || activeBucket == null) return;
    const list = d.buckets[activeBucket] || [];
    if (!list.length) return;
    activeSamplePos = Math.max(0, Math.min(list.length - 1, activeSamplePos + delta));
  }

  // The sample currently shown in the viewer (or null when no bucket is open).
  let activeSample = $derived.by(() => {
    const d = sampleDist;
    if (!d || activeBucket == null) return null;
    const list = d.buckets[activeBucket] || [];
    const idx = list[activeSamplePos];
    if (idx == null) return null;
    return {
      sample: expandedRollout.samples[idx],
      pos: activeSamplePos,
      count: list.length,
    };
  });

  function onSampleKeydown(e) {
    if (activeBucket == null) return;
    const tag = (e.target?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepSample(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      stepSample(1);
    }
  }

  function sampleToPayload(s) {
    return {
      score: s.score,
      prompt: s.prompt || null,
      response: s.response || null,
      thinking: s.thinking || null,
      raw_response: s.raw_response || null,
      raw_prompt: s.raw_prompt || null,
      trace: s.trace || null,
      metadata: s.metadata || null,
    };
  }

  function downloadSampleTrajectory() {
    if (!activeSample) return;
    const payload = sampleToPayload(activeSample.sample);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const rollout = expandedRolloutId ?? 0;
    a.download = `trajectory_r${rollout}_s${activeSample.pos}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadAllTrajectories() {
    if (!expandedRollout?.samples?.length) return;
    const rollout = expandedRolloutId ?? 0;
    const payload = {
      training_run_id: runId,
      rollout_id: rollout,
      total: expandedRollout.samples.length,
      mean: expandedRollout.samples.reduce((a, s) => a + (s.score || 0), 0) / expandedRollout.samples.length,
      samples: expandedRollout.samples.map(sampleToPayload),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rollout_${runId}_r${rollout}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function loadRollouts(signal) {
    if (!runId) return;
    try {
      const rows = await fetchRunRollouts(runId, { signal });
      if (signal?.aborted) return;
      rolloutSummaries = rows;
      rolloutsError = "";
    } catch (err) {
      if (signal?.aborted) return;
      // Keep the rollouts we already have on a transient poll failure — only
      // surface the error when there's nothing to show, so the charts/table
      // don't flip to an error message (and back) every flaky 5s poll.
      if (!rolloutSummaries.length) rolloutsError = String(err?.message || err);
    } finally {
      rolloutsLoading = false;
    }
  }

  // Reset rollout state when the run changes (separate from the fetch effect
  // so flipping between the summary/rollouts tabs doesn't clear what's loaded).
  $effect(() => {
    runId;
    rolloutSummaries = [];
    rolloutsError = "";
    expandedRolloutId = null;
    expandedRollout = null;
    closeBucket();
  });

  // Lazy load: only fetch rollout summaries while a tab that needs them is
  // active (the chart on Summary, the table on Rollouts).
  $effect(() => {
    const id = runId;
    const tab = activeTab;
    if (!id || (tab !== "summary" && tab !== "rollouts")) return;

    const controller = new AbortController();
    rolloutsLoading = true;
    void loadRollouts(controller.signal);

    // Poll while the run is active so new rollouts stream in.
    const interval = window.setInterval(() => {
      const status = String(run?.status || "").toLowerCase();
      if (status && status !== "running") return;
      void loadRollouts(controller.signal);
    }, 5000);

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  });

  async function toggleRolloutDetail(rolloutId) {
    if (!runId) return;
    if (expandedRolloutId === rolloutId) {
      expandedRolloutId = null;
      expandedRollout = null;
      closeBucket();
      return;
    }
    expandedRolloutId = rolloutId;
    expandedRollout = null;
    closeBucket();
    expandedRolloutLoading = true;
    try {
      const detail = await fetchRollout(runId, rolloutId);
      if (expandedRolloutId === rolloutId) {
        expandedRollout = detail;
        // Preselect the first populated bucket so a sample is shown right away.
        const d = sampleDist;
        const first = d ? d.buckets.findIndex((b) => b.length > 0) : -1;
        if (first >= 0) openBucket(first);
      }
    } finally {
      if (expandedRolloutId === rolloutId) {
        expandedRolloutLoading = false;
      }
    }
  }

  // ── Live Modal log stream (SSE, pure pass-through) ───────────────────
  const LOG_BUFFER_MAX = 2000;
  let logLines = $state([]); // [{task_id, line, ts}]
  let logState = $state("idle"); // idle | streaming | paused | done | error | reconnecting
  let logError = $state("");
  let logDropped = $state(0); // server-side rate-capped lines (cumulative since reconnect)
  let logTailEl = $state(null);

  // User controls
  let logPaused = $state(false);
  let logSearch = $state("");
  let logSearchInput = $state(""); // debounced into logSearch
  let logRateCap = $state(0); // 0 = no cap
  let logFollow = $state(true); // auto-scroll to bottom

  // The backend emits one SSE event per log line. Mutating `logLines` (and
  // auto-scrolling) on every message means hundreds of synchronous reactive
  // updates per second under a chatty run, which freezes the tab. Instead we
  // buffer incoming lines and flush them into `logLines` once per animation
  // frame — coalescing a burst into a single render + scroll.
  let pendingLogLines = []; // plain array, not reactive
  let logFlushHandle = null; // requestAnimationFrame handle
  let logSeq = 0; // monotonic id for stable keying

  function flushPendingLogs() {
    logFlushHandle = null;
    if (!pendingLogLines.length) return;
    const next = logLines.length
      ? logLines.concat(pendingLogLines)
      : pendingLogLines;
    pendingLogLines = [];
    logLines = next.length > LOG_BUFFER_MAX ? next.slice(-LOG_BUFFER_MAX) : next;
    if (logFollow && logTailEl) {
      // Scroll after Svelte has flushed the new rows to the DOM.
      queueMicrotask(() => {
        if (logTailEl) logTailEl.scrollTop = logTailEl.scrollHeight;
      });
    }
  }

  function scheduleLogFlush() {
    if (logFlushHandle != null) return;
    logFlushHandle = requestAnimationFrame(flushPendingLogs);
  }

  function resetLogBuffer() {
    if (logFlushHandle != null) {
      cancelAnimationFrame(logFlushHandle);
      logFlushHandle = null;
    }
    pendingLogLines = [];
  }

  // Debounce search input → URL
  $effect(() => {
    const value = logSearchInput;
    const handle = window.setTimeout(() => {
      logSearch = value.trim();
    }, 350);
    return () => window.clearTimeout(handle);
  });

  $effect(() => {
    const id = runId;
    const tab = activeTab;
    // Depend on the primitive status (not `run`) so a same-status refresh
    // doesn't tear down and rebuild the stream.
    const status = runStatus;
    // Re-create the EventSource whenever any of the connection params change.
    const search = logSearch;
    const rate = logRateCap;
    const paused = logPaused;

    resetLogBuffer();
    logLines = [];
    logState = "idle";
    logError = "";
    logDropped = 0;
    // Lazy load: only open the log stream while the Logs tab is active.
    if (tab !== "logs" || !id || status !== "running" || paused) {
      if (tab === "logs" && paused) logState = "paused";
      return;
    }

    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (rate > 0) params.set("max_lines_per_sec", String(rate));
    const qs = params.toString();
    const url =
      `/api/runs/${encodeURIComponent(id)}/logs/stream` +
      (qs ? `?${qs}` : "");

    let es;
    try {
      es = new EventSource(url);
    } catch (err) {
      logError = String(err?.message || err || "could not open log stream");
      logState = "error";
      return;
    }
    logState = "streaming";

    es.onopen = () => {
      logState = "streaming";
      logError = "";
    };

    es.onmessage = (evt) => {
      if (logState !== "streaming") {
        logState = "streaming";
        logError = "";
      }
      try {
        const payload = JSON.parse(evt.data);
        const line = String(payload.line || "");
        if (!line) return;
        const parts = line.split(/\r?\n/);
        const task_id = payload.task_id || "";
        const ts = payload.ts || Date.now();
        for (const p of parts) {
          if (!p.length) continue;
          // Parse ANSI color/style codes once at ingestion so the render path
          // stays cheap (lines are immutable once buffered).
          pendingLogLines.push({ id: logSeq++, task_id, line: p, ts, segments: parseAnsi(p) });
        }
        if (pendingLogLines.length > LOG_BUFFER_MAX) {
          pendingLogLines = pendingLogLines.slice(-LOG_BUFFER_MAX);
        }
        scheduleLogFlush();
      } catch {
        // ignore malformed payloads
      }
    };

    es.addEventListener("done", () => {
      logState = "done";
      es.close();
    });

    es.addEventListener("reconnect", (evt) => {
      try {
        const { reason } = JSON.parse(evt.data || "{}");
        logError = String(reason || "");
      } catch {
        logError = "";
      }
      logState = "reconnecting";
    });

    es.addEventListener("dropped", (evt) => {
      try {
        const { dropped } = JSON.parse(evt.data || "{}");
        logDropped += Number(dropped) || 0;
      } catch {}
    });

    es.addEventListener("error", (evt) => {
      try {
        const { error } = JSON.parse(evt.data || "{}");
        logError = String(error || "");
      } catch {
        logError = "";
      }
      logState = "error";
      es.close();
    });

    es.onerror = () => {
      if (logState === "streaming") logState = "reconnecting";
    };

    return () => {
      resetLogBuffer();
      try {
        es.close();
      } catch {}
    };
  });

  function toggleLogPaused() {
    logPaused = !logPaused;
  }

  function clearLogs() {
    resetLogBuffer();
    logLines = [];
    logDropped = 0;
  }

  // Build an SVG polyline path for a per-rollout-step series (x = rollout_id).
  function _rolloutLinePath(getY) {
    const points = rolloutSummaries.map((r) => ({
      x: Number(r.rollout_id) || 0,
      y: getY(r),
    }));
    if (!points.length) return "";
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xMin = Math.min(...xs);
    const xSpan = Math.max(...xs) - xMin || 1;
    const yMin = Math.min(...ys);
    const ySpan = Math.max(...ys) - yMin || 1;
    const W = 640;
    const H = 140;
    return points
      .map((p, i) => {
        const x = ((p.x - xMin) / xSpan) * W;
        const y = H - ((p.y - yMin) / ySpan) * (H - 4) - 2;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  }

  function _seriesStats(getY) {
    if (!rolloutSummaries.length) return null;
    const values = rolloutSummaries.map(getY);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      latest: values[values.length - 1],
    };
  }

  let chartPath = $derived(_rolloutLinePath((r) => Number(r.mean) || 0));
  let chartStats = $derived(_seriesStats((r) => Number(r.mean) || 0));

  // Score-distribution comparison: the first rollout (step 0) vs the most
  // recent one, to see how sample scores shifted over training.
  let firstRolloutId = $derived(
    rolloutSummaries.length
      ? Math.min(...rolloutSummaries.map((r) => Number(r.rollout_id) || 0))
      : null,
  );
  let lastRolloutId = $derived(
    rolloutSummaries.length
      ? Math.max(...rolloutSummaries.map((r) => Number(r.rollout_id) || 0))
      : null,
  );
  let scoreDist = $state(null);

  function buildScoreDist(firstSamples, lastSamples, firstId, lastId) {
    const scores = [...firstSamples, ...lastSamples].map((s) => Number(s.score) || 0);
    if (!scores.length) return null;
    const lo = Math.min(...scores);
    const hi = Math.max(...scores);
    const n = lo === hi ? 1 : 12;
    const span = hi - lo || 1;
    const bins = Array.from({ length: n }, (_, i) => ({
      lo: lo + (i / n) * span,
      hi: lo + ((i + 1) / n) * span,
      first: 0,
      last: 0,
    }));
    const idx = (s) => Math.max(0, Math.min(n - 1, Math.floor(((s - lo) / span) * n)));
    for (const s of firstSamples) bins[idx(Number(s.score) || 0)].first += 1;
    for (const s of lastSamples) bins[idx(Number(s.score) || 0)].last += 1;
    const max = Math.max(1, ...bins.map((b) => Math.max(b.first, b.last)));
    return { bins, max, lo, hi, firstId, lastId };
  }

  // Fetch the two rollouts' samples only when the endpoints change (a new step
  // lands), not on every 5s poll — the payloads are large.
  $effect(() => {
    if (activeTab !== "summary") return;
    const id = runId;
    const fId = firstRolloutId;
    const lId = lastRolloutId;
    if (!id || fId == null || lId == null) {
      scoreDist = null;
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const first = await fetchRollout(id, fId);
        const last = fId === lId ? first : await fetchRollout(id, lId);
        if (cancelled) return;
        scoreDist = buildScoreDist(first?.samples || [], last?.samples || [], fId, lId);
      } catch {
        if (!cancelled) scoreDist = null;
      }
    })();
    return () => {
      cancelled = true;
    };
  });
</script>

<svelte:window onkeydown={onSampleKeydown} />

<section class="detail" class:embedded>
  {#if !embedded}
    <header class="detail-header">
      <button class="back-button" onclick={onBack}>
        <ArrowLeft size={14} strokeWidth={2.1} />
        <span>Back to runs</span>
      </button>
      <div class="detail-header-actions">
        {#if onCollapse}
          <button class="detail-collapse-button" onclick={onCollapse} title="Collapse to drawer">
            <Minimize2 size={12} strokeWidth={2.1} />
            <span>Collapse</span>
          </button>
        {/if}
        {#if run?.train_result?.wandb_url || run?.config_summary?.wandb_project}
          <a
            class="header-link wandb-link"
            href={run.train_result?.wandb_url || `https://wandb.ai/home?search=${encodeURIComponent(run.config_summary.wandb_project)}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <span>Open in W&B</span>
            <ExternalLink size={12} strokeWidth={2.1} />
          </a>
        {/if}
        {#if run?.modal_app_url}
          <a
            class="header-link"
            href={run.modal_app_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            <span>Open in Modal</span>
            <ExternalLink size={12} strokeWidth={2.1} />
          </a>
        {/if}
      </div>
    </header>
  {/if}

  {#if !run}
    <div class="empty">Loading run {runId}…</div>
  {:else}
    {#if !embedded}
    <div class="detail-title-row">
      <h1 class="detail-title" title={run.run_id}>{run.run_id}</h1>
      <StatusPill status={getStatus(run)} />
    </div>
    {/if}

    <Tabs
      bind:active={activeTab}
      tabs={[
        { value: "summary", label: "Summary" },
        { value: "rollouts", label: "Rollouts", count: rolloutSummaries.length || undefined },
        { value: "logs", label: "Logs" },
      ]}
    />

    {#if activeTab === "summary"}
      <div class="summary-tab">
        <div class="summary-tab-main">
          {#if rolloutsLoading && !rolloutSummaries.length}
            <div class="empty">Loading rollouts…</div>
          {:else if rolloutsError}
            <div class="empty">Failed to load rollouts: {rolloutsError}</div>
          {:else if !rolloutSummaries.length}
            <div class="empty">No rollouts recorded yet.</div>
          {:else}
            <div class="rollout-chart">
              <div class="rollout-chart-title">Reward</div>
              {#if rolloutSummaries.length >= 2}
                <svg viewBox="0 0 640 140" preserveAspectRatio="none" aria-hidden="true">
                  <path d={chartPath} fill="none" stroke="var(--accent)" stroke-width="1.5" />
                </svg>
              {/if}
              {#if chartStats}
                <div class="rollout-chart-meta">
                  <span>min {formatMean(chartStats.min)}</span>
                  <span>latest {formatMean(chartStats.latest)}</span>
                  <span>max {formatMean(chartStats.max)}</span>
                </div>
              {/if}
            </div>
            <div class="rollout-chart">
              <div class="rollout-chart-title">Score distribution</div>
              {#if scoreDist}
                <div class="dist-legend">
                  <span class="dist-legend-item">
                    <span class="dist-swatch swatch-first"></span>
                    rollout {scoreDist.firstId}
                  </span>
                  {#if scoreDist.firstId !== scoreDist.lastId}
                    <span class="dist-legend-item">
                      <span class="dist-swatch swatch-last"></span>
                      latest (rollout {scoreDist.lastId})
                    </span>
                  {/if}
                </div>
                <div class="dist-compare">
                  {#each scoreDist.bins as bin, i (i)}
                    <div
                      class="dist-compare-bin"
                      title={`reward ${formatMean(bin.lo)}–${formatMean(bin.hi)} · rollout ${scoreDist.firstId}: ${bin.first}, latest: ${bin.last}`}
                    >
                      <div
                        class="dist-compare-bar swatch-first"
                        style:height={`${(bin.first / scoreDist.max) * 100}%`}
                      ></div>
                      {#if scoreDist.firstId !== scoreDist.lastId}
                        <div
                          class="dist-compare-bar swatch-last"
                          style:height={`${(bin.last / scoreDist.max) * 100}%`}
                        ></div>
                      {/if}
                    </div>
                  {/each}
                </div>
                <div class="dist-axis">
                  <span>{formatMean(scoreDist.lo)}</span>
                  <span class="dist-axis-label">reward</span>
                  <span>{formatMean(scoreDist.hi)}</span>
                </div>
              {:else}
                <div class="empty">Loading distribution…</div>
              {/if}
            </div>
          {/if}
        </div>
        <aside class="summary-tab-side">
          <RunSummary
            {run}
            {getStatus}
            {showFrameworkStatus}
            {getFrameworkStatus}
            {modelName}
            {fmtDuration}
          />
        </aside>
      </div>
    {:else if activeTab === "rollouts"}
      <div class="tab-panel">
      {#if rolloutsLoading && !rolloutSummaries.length}
        <div class="empty">Loading rollouts…</div>
      {:else if rolloutsError}
        <div class="empty">Failed to load rollouts: {rolloutsError}</div>
      {:else if !rolloutSummaries.length}
        <div class="empty">No rollouts recorded yet.</div>
      {:else}
        <table class="rollout-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Mean reward</th>
              <th>Samples</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {#each rolloutSummaries as r (r.rollout_id)}
              <tr
                class:expanded={expandedRolloutId === r.rollout_id}
                class:rollout-error={r.error_summary?.verdict === "all_infra_failure"}
                class:rollout-warn={r.error_summary?.verdict === "partial_infra_failure"}
                onclick={() => toggleRolloutDetail(r.rollout_id)}
              >
                <td>#{r.rollout_id}</td>
                <td class="rollout-mean">
                  {formatMean(r.mean)}
                  {#if r.error_summary?.verdict === "all_infra_failure"}
                    <span class="rollout-error-badge" title="All samples failed due to infrastructure error">infra failure</span>
                  {:else if r.error_summary?.verdict === "partial_infra_failure"}
                    <span class="rollout-warn-badge" title="Some samples failed due to infrastructure error">partial failure</span>
                  {/if}
                </td>
                <td>{r.total}</td>
                <td>
                  <TimeAgo timestamp={r.created_at} showJustNow falsyRepresentation="—" />
                </td>
              </tr>
              {#if expandedRolloutId === r.rollout_id}
                <tr class="rollout-detail-row">
                  <td colspan="4">
                    {#if expandedRolloutLoading}
                      <div class="empty">Loading samples…</div>
                    {:else if !expandedRollout || !sampleDist}
                      <div class="empty">No samples recorded.</div>
                    {:else}
                      {#if expandedRollout.metrics && Object.keys(expandedRollout.metrics).length}
                        {@const m = expandedRollout.metrics}
                        {@const remoteErr = Number(m["agent/exit_status/remoteerror_sample_count"]) || 0}
                        {@const responseMissing = Number(m["agent/response_missing_sample_count"]) || 0}
                        {@const infraInvalid = Number(m["agent/invalid_infra_sample_count"]) || 0}
                        {@const limitsExceeded = Number(m["agent/limits_exceeded_sample_count"]) || 0}
                        {@const totalSamples = Number(m["agent/valid_sample_count"]) || sampleDist.total || 0}
                        {@const hasErrors = remoteErr > 0 || responseMissing > 0 || infraInvalid > 0}
                        {#if hasErrors}
                          <div class="rollout-diagnostics" class:diag-critical={remoteErr >= totalSamples}>
                            <div class="diag-title">
                              {#if remoteErr >= totalSamples}
                                All {totalSamples} samples failed — infrastructure error
                              {:else}
                                {remoteErr + infraInvalid} / {totalSamples} samples hit infrastructure errors
                              {/if}
                            </div>
                            <div class="diag-details">
                              {#if remoteErr}
                                <span class="diag-tag">RemoteError: {remoteErr}</span>
                              {/if}
                              {#if responseMissing}
                                <span class="diag-tag">Response missing: {responseMissing}</span>
                              {/if}
                              {#if infraInvalid}
                                <span class="diag-tag">Infra invalid: {infraInvalid}</span>
                              {/if}
                              {#if limitsExceeded}
                                <span class="diag-tag">Limits exceeded: {limitsExceeded}</span>
                              {/if}
                            </div>
                            {#if remoteErr >= totalSamples}
                              <div class="diag-hint">
                                Check the Modal app logs for sandbox/image build errors. Common cause: the environment image failed to build.
                              </div>
                            {/if}
                          </div>
                        {/if}
                      {/if}
                      <div class="dist">
                        <div class="dist-toolbar">
                          <button
                            class="download-all-btn"
                            onclick={downloadAllTrajectories}
                            title="Download all samples as JSON"
                          >
                            <Download size={13} />
                            Download all ({sampleDist.total} samples)
                          </button>
                        </div>
                        <div
                          class="dist-bars"
                          role="group"
                          aria-label="Sample score distribution"
                        >
                          {#each sampleDist.buckets as bucket, b (b)}
                            <button
                              class="dist-bar"
                              class:active={activeBucket === b}
                              class:is-empty={!bucket.length}
                              style:height={`${(bucket.length / sampleDist.maxCount) * 100}%`}
                              disabled={!bucket.length}
                              title={`${bucket.length} sample${bucket.length === 1 ? "" : "s"} · reward ${bucketRange(b)}`}
                              onclick={() => openBucket(b)}
                            >
                              <span class="dist-bar-count">{bucket.length || ""}</span>
                            </button>
                          {/each}
                        </div>
                        <div class="dist-axis">
                          <span>{formatMean(sampleDist.lo)}</span>
                          <span class="dist-axis-label">reward · {sampleDist.total} samples</span>
                          <span>{formatMean(sampleDist.hi)}</span>
                        </div>
                      </div>

                      {#if activeSample}
                        <div class="rollout-sample sample-viewer">
                          <div class="sample-viewer-header">
                            <div class="sample-viewer-nav">
                              <button
                                class="sample-nav-btn"
                                onclick={() => stepSample(-1)}
                                disabled={activeSample.pos === 0}
                                aria-label="Previous sample"
                              >
                                <ChevronLeft size={14} />
                              </button>
                              <span class="sample-viewer-pos">
                                Sample {activeSample.pos + 1} / {activeSample.count}
                              </span>
                              <button
                                class="sample-nav-btn"
                                onclick={() => stepSample(1)}
                                disabled={activeSample.pos === activeSample.count - 1}
                                aria-label="Next sample"
                              >
                                <ChevronRight size={14} />
                              </button>
                              <span class="sample-viewer-hint">← / → to navigate</span>
                            </div>
                            <div class="sample-viewer-meta">
                              <span class="rollout-sample-score">
                                reward {formatMean(activeSample.sample.score)}
                              </span>
                              <button
                                class="sample-nav-btn"
                                onclick={downloadSampleTrajectory}
                                aria-label="Download trajectory JSON"
                                title="Download trajectory"
                              >
                                <Download size={14} />
                              </button>
                              <button
                                class="sample-nav-btn"
                                onclick={closeBucket}
                                aria-label="Close sample viewer"
                              >
                                <X size={14} />
                              </button>
                            </div>
                          </div>
                          {#if activeSample.sample.metadata?._metadata_type === "audio" || activeSample.sample.metadata?.audio}
                            <div class="rollout-sample-label">audio</div>
                            <audio
                              class="sample-audio"
                              controls
                              preload="none"
                              src={activeSample.sample.metadata.audio}
                            ></audio>
                          {/if}
                          {#if activeSample.sample.prompt}
                            <div class="rollout-sample-label">prompt</div>
                            <pre class="rollout-sample-text">{activeSample.sample.prompt}</pre>
                          {/if}
                          <div class="rollout-sample-label">conversation</div>
                          <ConversationView
                            messages={activeSample.sample.metadata?.trajectory_messages}
                            response={activeSample.sample.response || ""}
                            thinking={activeSample.sample.thinking || ""}
                            evalReport={activeSample.sample.metadata?.eval_report}
                          />
                          {#if activeSample.sample.metadata?.reference}
                            <div class="rollout-sample-label">reference</div>
                            <pre class="rollout-sample-text">{activeSample.sample.metadata.reference}</pre>
                          {/if}
                          {#each Object.entries(activeSample.sample.metadata?.metrics ?? {}) as [name, value]}
                            <div class="rollout-sample-label">{name}</div>
                            <span class="rollout-sample-metric">
                              {typeof value === "number" ? value.toFixed(3) : value}
                            </span>
                          {/each}
                          {#if activeSample.sample.metadata?.exit_status}
                            <div class="rollout-sample-label">exit status</div>
                            <span class="rollout-sample-metric sample-exit-status" class:exit-ok={activeSample.sample.metadata.exit_status === "ok"} class:exit-err={activeSample.sample.metadata.exit_status !== "ok"}>
                              {activeSample.sample.metadata.exit_status}
                            </span>
                          {/if}
                          {#if activeSample.sample.trace?.length}
                            <div class="rollout-sample-label">trajectory timeline</div>
                            <SampleTimeline trace={activeSample.sample.trace} />
                          {/if}
                        </div>
                      {:else}
                        <div class="dist-hint">Click a bar to inspect its samples.</div>
                      {/if}
                    {/if}
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </table>
      {/if}
      </div>
    {:else if activeTab === "logs"}
      <div class="tab-panel">
      <div class="logs-statusbar">
        <span class="logs-status">
          {#if logState === "streaming"}
            <span class="dot dot-live"></span> live
          {:else if logState === "paused"}
            <span class="dot dot-dim"></span> paused
          {:else if logState === "reconnecting"}
            <span class="dot dot-warn"></span> reconnecting…{#if logError} <span class="log-reconnect-reason">({logError})</span>{/if}
          {:else if logState === "done"}
            <span class="dot dot-dim"></span> finished
          {:else if logState === "error"}
            <span class="dot dot-err"></span> error
          {:else if String(run?.status || "").toLowerCase() !== "running"}
            <span class="dot dot-dim"></span> run not active
          {:else}
            <span class="dot dot-dim"></span> idle
          {/if}
        </span>
      </div>

      <div class="logs-controls">
        <button
          class="log-button"
          onclick={toggleLogPaused}
          disabled={String(run?.status || "").toLowerCase() !== "running"}
        >
          {logPaused ? "Resume" : "Pause"}
        </button>
        <button class="log-button" onclick={clearLogs} disabled={!logLines.length}>
          Clear
        </button>
        <input
          class="log-search"
          type="search"
          placeholder="filter substring…"
          bind:value={logSearchInput}
          aria-label="Filter log lines"
        />
        <label class="log-rate">
          <span>Rate cap</span>
          <select bind:value={logRateCap} aria-label="Lines per second cap">
            <option value={0}>off</option>
            <option value={10}>10/s</option>
            <option value={50}>50/s</option>
            <option value={200}>200/s</option>
            <option value={1000}>1000/s</option>
          </select>
        </label>
        <label class="log-follow">
          <input type="checkbox" bind:checked={logFollow} />
          <span>Follow tail</span>
        </label>
      </div>

      {#if logState === "error" && logError}
        <div class="empty">Log stream error: {logError}</div>
      {/if}

      {#if !logLines.length}
        <div class="empty">
          {#if String(run?.status || "").toLowerCase() !== "running"}
            Logs only stream while the run is active.
          {:else if logPaused}
            Stream paused.
          {:else if logSearch}
            Waiting for log output matching "{logSearch}"…
          {:else}
            Waiting for log output…
          {/if}
        </div>
      {:else}
        <div class="log-tail" bind:this={logTailEl}>
          {#each logLines as entry (entry.id)}
            <div class="log-row">
              <span class="log-task">{entry.task_id || ""}</span>
              <span class="log-line">{#each entry.segments as seg, i (i)}<span style={seg.style}>{seg.text}</span>{/each}</span>
            </div>
          {/each}
        </div>
        <div class="log-meta">
          <span>
            Showing last {logLines.length} line{logLines.length === 1 ? "" : "s"} (cap {LOG_BUFFER_MAX})
          </span>
          {#if logDropped > 0}
            <span class="log-meta-drop">
              · {logDropped} dropped by rate cap
            </span>
          {/if}
        </div>
      {/if}
      </div>
    {/if}
  {/if}
</section>

<style>
  .detail {
    padding: 0 0 24px;
    color: var(--text);
  }

  /* Inside the expanded drawer the drawer owns padding/width, so drop the
     page chrome and let the rollouts + logs fill the wide drawer. */
  .detail.embedded {
    padding: 0;
    max-width: none;
    margin: 0;
  }

  .detail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .back-button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: 0;
    color: var(--muted);
    cursor: pointer;
    font-size: 13px;
    padding: 4px 8px;
    border-radius: 6px;
  }

  .back-button:hover {
    color: var(--text);
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .detail-header-actions {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  .detail-collapse-button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 6px;
    background: none;
    color: var(--muted);
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 8px;
  }

  .detail-collapse-button:hover {
    color: var(--text-bright);
    border-color: var(--border-strong, #4a4a4a);
  }

  .header-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
    text-decoration: none;
  }

  .header-link:hover {
    color: var(--accent);
  }

  .wandb-link {
    color: var(--yellow, #fbbf24);
  }

  .wandb-link:hover {
    color: var(--yellow, #fbbf24);
    opacity: 0.8;
  }

  .detail-title-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
  }

  .detail-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-bright);
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ── Tab panels ───────────────────────────────────────────────────────── */
  .tab-panel {
    padding-top: 20px;
  }

  /* Summary tab: chart on the left, run summary metadata on the right. */
  .summary-tab {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
    gap: 32px;
    align-items: start;
    padding-top: 20px;
  }

  .summary-tab-side {
    border-left: 1px solid var(--border, #2f2f2f);
    padding-left: 24px;
  }

  .logs-statusbar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 8px;
  }

  @media (max-width: 900px) {
    .summary-tab {
      grid-template-columns: 1fr;
      gap: 20px;
    }

    .summary-tab-side {
      border-left: 0;
      padding-left: 0;
      border-top: 1px solid var(--border, #2f2f2f);
      padding-top: 16px;
    }
  }

  .rollout-chart {
    margin-bottom: 20px;
  }

  .rollout-chart-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-bright);
    margin-bottom: 6px;
  }

  /* Score-distribution comparison (first vs latest rollout). */
  .swatch-first {
    background: var(--color-c-gray-40, #5e5e5e);
  }

  .swatch-last {
    background: var(--accent);
  }

  .dist-legend {
    display: flex;
    gap: 16px;
    margin-bottom: 8px;
    font-size: 11px;
    color: var(--muted);
  }

  .dist-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .dist-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }

  .dist-compare {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 120px;
    padding-top: 8px;
    border-bottom: 1px solid var(--border, #2f2f2f);
  }

  .dist-compare-bin {
    flex: 1;
    height: 100%;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 2px;
  }

  .dist-compare-bar {
    flex: 1;
    min-height: 1px;
    border-radius: 2px 2px 0 0;
  }

  .rollout-chart svg {
    width: 100%;
    height: 140px;
    background: var(--color-c-gray-08, #1c1c1c);
    border-radius: 6px;
  }

  .rollout-chart-meta {
    display: flex;
    gap: 16px;
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .rollout-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  .rollout-table th {
    text-align: left;
    color: var(--muted);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border, #2f2f2f);
  }

  .rollout-table tbody tr {
    cursor: pointer;
  }

  .rollout-table tbody tr:hover {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .rollout-table tbody tr.expanded {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .rollout-table td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border, #2f2f2f);
    font-variant-numeric: tabular-nums;
  }

  .rollout-mean {
    color: var(--text-bright);
  }

  .rollout-detail-row td {
    padding: 12px 10px;
    background: var(--color-c-gray-08, #1c1c1c);
    cursor: default;
  }

  .rollout-sample {
    border-left: 2px solid var(--accent);
    padding: 8px 12px;
    margin-bottom: 12px;
    background: var(--color-c-gray-10, #2f2f2f);
    border-radius: 0 4px 4px 0;
  }

  /* ── Per-step sample score distribution ──────────────────────────────── */
  .dist {
    margin-bottom: 16px;
  }

  .dist-toolbar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 6px;
  }

  .download-all-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
  }

  .download-all-btn:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .dist-bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 120px;
    padding-top: 14px;
    border-bottom: 1px solid var(--border, #2f2f2f);
  }

  .dist-bar {
    position: relative;
    flex: 1;
    min-height: 2px;
    padding: 0;
    border: 0;
    border-radius: 2px 2px 0 0;
    background: var(--color-c-gray-30, #4a4a4a);
    cursor: pointer;
    transition:
      background 0.12s ease,
      opacity 0.12s ease;
  }

  .dist-bar:hover:not(:disabled) {
    background: var(--color-c-gray-40, #5e5e5e);
  }

  .dist-bar.active {
    background: var(--accent);
  }

  .dist-bar.is-empty {
    background: var(--color-c-gray-10, #2f2f2f);
    cursor: default;
    opacity: 0.5;
  }

  .dist-bar-count {
    position: absolute;
    top: -14px;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 10px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .dist-axis {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .dist-axis-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .dist-hint {
    font-size: 12px;
    color: var(--muted);
    padding: 4px 0;
  }

  /* ── Single-sample viewer (bucket drill-in) ──────────────────────────── */
  .sample-viewer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 6px;
  }

  .sample-viewer-nav,
  .sample-viewer-meta {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  .sample-viewer-pos {
    font-size: 12px;
    color: var(--text-bright);
    font-variant-numeric: tabular-nums;
  }

  .sample-viewer-hint {
    font-size: 11px;
    color: var(--muted);
  }

  .sample-nav-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    background: none;
    color: var(--muted);
    cursor: pointer;
  }

  .sample-nav-btn:hover:not(:disabled) {
    color: var(--text-bright);
    border-color: var(--border-strong, #4a4a4a);
  }

  .sample-nav-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .rollout-sample-score {
    color: var(--text-bright);
    font-variant-numeric: tabular-nums;
  }

  .rollout-sample-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-top: 8px;
    margin-bottom: 2px;
  }

  .sample-audio {
    display: block;
    width: 100%;
    max-width: 400px;
    margin: 4px 0 8px;
    border-radius: 4px;
  }

  .rollout-sample-metric {
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    color: var(--text-bright);
  }

  .rollout-sample-text {
    margin: 0;
    padding: 8px;
    background: var(--color-c-gray-08, #1c1c1c);
    border-radius: 4px;
    font-size: 12px;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 240px;
    overflow: auto;
  }

  .sample-exit-status {
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 500;
  }
  .exit-ok {
    background: rgba(74, 222, 128, 0.12);
    color: #4ade80;
  }
  .exit-err {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
  }

  .empty {
    color: var(--muted);
    font-size: 13px;
    padding: 16px 0;
  }

  .logs-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 9999px;
    background: var(--muted);
  }

  .dot-live {
    background: #4ade80;
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.18);
  }

  .dot-warn {
    background: #fbbf24;
  }

  .dot-err {
    background: #f87171;
  }

  .dot-dim {
    background: #6b7280;
  }

  .log-tail {
    background: var(--color-c-gray-08, #0e0e0e);
    border-radius: 6px;
    padding: 8px 12px;
    max-height: 420px;
    overflow-y: auto;
    overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text);
  }

  .log-row {
    display: flex;
    gap: 10px;
    white-space: pre;
  }

  .log-task {
    flex-shrink: 0;
    color: var(--muted);
    font-size: 10px;
    min-width: 64px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .log-line {
    flex: 1;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .log-meta {
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    display: flex;
    gap: 6px;
  }

  .log-meta-drop {
    color: #fbbf24;
  }

  .logs-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .log-button {
    background: var(--color-c-gray-10, #2f2f2f);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
  }

  .log-button:hover:not(:disabled) {
    background: var(--color-c-gray-12, #3a3a3a);
  }

  .log-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .log-search {
    flex: 1;
    min-width: 160px;
    background: var(--color-c-gray-08, #1c1c1c);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    font-family: inherit;
  }

  .log-rate,
  .log-follow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 11px;
  }

  .log-rate select {
    background: var(--color-c-gray-08, #1c1c1c);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
  }

  /* ── Rollout error indicators ──────────────────────────────────────── */

  .rollout-error td {
    background: rgba(239, 68, 68, 0.06);
  }
  .rollout-warn td {
    background: rgba(251, 191, 36, 0.06);
  }

  .rollout-error-badge,
  .rollout-warn-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 500;
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
  }
  .rollout-error-badge {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.25);
  }
  .rollout-warn-badge {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.25);
  }

  /* ── Rollout diagnostics banner ────────────────────────────────────── */

  .rollout-diagnostics {
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 12px;
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.2);
  }
  .rollout-diagnostics.diag-critical {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.2);
  }

  .diag-title {
    font-size: 13px;
    font-weight: 600;
    color: #fbbf24;
    margin-bottom: 6px;
  }
  .diag-critical .diag-title {
    color: #ef4444;
  }

  .diag-details {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 4px;
  }
  .diag-tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.06);
    color: var(--text, #d1d1d1);
    font-variant-numeric: tabular-nums;
  }

  .diag-hint {
    font-size: 11px;
    color: var(--muted, #a3a3a3);
    margin-top: 6px;
  }
</style>
