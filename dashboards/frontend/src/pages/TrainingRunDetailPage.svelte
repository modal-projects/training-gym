<script>
  import { onMount, tick } from "svelte";
  import { ArrowLeft, ChevronLeft, ChevronRight, Download, ExternalLink, Minimize2, X } from "lucide-svelte";
  import Tabs from "../components/Tabs.svelte";
  import RunSummary from "../components/RunSummary.svelte";
  import StepTimings from "../components/StepTimings.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import InferenceStats from "../components/InferenceStats.svelte";
  import SampleTimeline from "../components/SampleTimeline.svelte";
  import ConversationView from "../components/ConversationView.svelte";
  import AdvantageViolins from "../components/AdvantageViolins.svelte";
  import AdvantageSpreadChart from "../components/AdvantageSpreadChart.svelte";
  import ComparativeBarChart from "../components/ComparativeBarChart.svelte";
  import ChartSkeleton from "../components/ChartSkeleton.svelte";
  import LineChart from "../components/LineChart.svelte";
  import ResizableTable from "../components/ResizableTable.svelte";
  import {
    fetchRunRollouts,
    fetchRollout,
    fetchRunAdvantages,
    fetchRunAdvantageStep,
    fetchRunLogs,
    fetchSubstepTimings,
  } from "../lib/api.js";

  // Number of historical log lines requested per page.
  const HIST_PAGE = 500;
  // Maximum number of historical log lines retained in the browser.
  const HIST_BUFFER_MAX = 2000;

  /** @typedef {"summary" | "rollouts" | "logs"} TabId */
  const DETAIL_TABS = new Set(["summary", "rollouts", "logs"]);
  const DEFAULT_TAB = "summary";

  function parseTabFromUrl() {
    if (typeof window === "undefined") return DEFAULT_TAB;
    const raw = new URLSearchParams(window.location.search).get("tab");
    return DETAIL_TABS.has(raw) ? /** @type {TabId} */ (raw) : DEFAULT_TAB;
  }

  function urlForTab(tab) {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", DETAIL_TABS.has(tab) ? tab : DEFAULT_TAB);
    return `${url.pathname}${url.search}${url.hash}`;
  }

  function locationKey() {
    return `${window.location.pathname}${window.location.search}${window.location.hash}`;
  }

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

  let run = $derived.by(() =>
    (allRuns || []).find((r) => r.run_id === runId) || null
  );

  // Status as a primitive so effects depending on it don't re-run every time
  // the auto-refresh hands us a new `run` object with the same status (which
  // would otherwise tear down and rebuild the log stream, flashing the tail).
  let runStatus = $derived(String(run?.status || "").toLowerCase());
  let wandbUrl = $derived.by(() => {
    const directUrl = run?.train_result?.wandb_url || run?.config_summary?.wandb_url || "";
    if (directUrl) return directUrl;

    const project = run?.config_summary?.wandb_project || "";
    return project ? `https://wandb.ai/home?search=${encodeURIComponent(project)}` : "";
  });
  let wandbLinks = $derived.by(() =>
    run?.wandb_links?.length
      ? run.wandb_links
      : wandbUrl
        ? [{ label: "Open in W&B", url: wandbUrl }]
        : [],
  );

  // Active tab: "summary" | "rollouts" | "logs". One-way sync with the URL:
  // init/popstate/runId read URL → activeTab; selectTab writes pushState.
  let activeTab = $state(/** @type {TabId} */ (DEFAULT_TAB));

  function selectTab(tab) {
    const next = DETAIL_TABS.has(tab) ? /** @type {TabId} */ (tab) : DEFAULT_TAB;
    activeTab = next;
    if (embedded || typeof window === "undefined") return;
    const target = urlForTab(next);
    if (target !== locationKey()) {
      history.pushState({}, "", target);
    }
  }

  onMount(() => {
    if (embedded) {
      activeTab = DEFAULT_TAB;
    } else {
      const tab = parseTabFromUrl();
      activeTab = tab;
      const target = urlForTab(tab);
      if (target !== locationKey()) {
        history.replaceState({}, "", target);
      }
    }

    const onPopState = () => {
      if (embedded) return;
      activeTab = parseTabFromUrl();
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  });

  $effect(() => {
    runId;
    if (embedded || typeof window === "undefined") return;
    activeTab = parseTabFromUrl();
  });

  function formatMean(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(3);
  }

  // Map a rollout to its step timing. Step keys are 1-indexed; rollout ids are
  // 0-indexed, so step N corresponds to rollout N-1 (fall back to a direct match).
  function stepTimingForRollout(rolloutId) {
    const st = displayedStepTimes;
    const sub = displayedSubstepTimes;
    if (!st && !sub) return null;
    const candidates = [String(Number(rolloutId) + 1), String(rolloutId)];
    const key = candidates.find((k) => (st && st[k]) || (sub && sub[k]));
    if (!key) return null;
    return {
      stepTimes: st && st[key] ? { [key]: st[key] } : null,
      substepTimes: sub && sub[key] ? { [key]: sub[key] } : null,
    };
  }

  function resumeBadge(run) {
    const state = run?.resume_state;
    if (!state) return "";
    const parts = [];
    if (state.attempt_count > 1) parts.push(`attempt ${state.attempt_count}`);
    if (state.resumed_from_checkpoint) {
      parts.push(
        state.resume_from_iteration != null
          ? `resumed @ ${state.resume_from_iteration}`
          : "resumed",
      );
    }
    return parts.join(" · ");
  }

  // ── Rollouts (auto-refresh while run is running) ─────────────────────
  let rolloutSummaries = $state([]);
  let substepTimingByAttemptAndRollout = $state({});
  let pendingSubstepTimingKeys = new Set();
  let rolloutLoadVersion = 0;
  let rolloutsLoading = $state(false);
  let rolloutsError = $state("");
  let expandedRolloutId = $state(null);
  let expandedRolloutAttempt = $state(null);
  let expandedRollout = $state(null);
  let expandedRolloutLoading = $state(false);
  let liveStepTimes = $derived.by(() => {
    const steps = {};
    for (const rollout of rolloutSummaries) {
      const timing =
        substepTimingByAttemptAndRollout[
          `${rollout.training_attempt}:${rollout.rollout_id}`
        ];
      if (!timing) continue;
      const key = String(Number(rollout.rollout_id) + 1);
      steps[key] = {
        start: timing.started_at_unix_s,
        end: timing.started_at_unix_s + timing.duration_s,
        duration_s: timing.duration_s,
      };
    }
    return steps;
  });
  let liveSubstepTimes = $derived.by(() => {
    const steps = {};
    for (const rollout of rolloutSummaries) {
      const timing =
        substepTimingByAttemptAndRollout[
          `${rollout.training_attempt}:${rollout.rollout_id}`
        ];
      if (!timing) continue;
      const phases = {};
      for (const role of timing.roles || []) {
        for (const phase of role.phases || []) {
          if (phase.phase === "full_step") continue;
          phases[`${phase.phase} (${role.role})`] = {
            start: phase.started_at_unix_s,
            end: phase.started_at_unix_s + phase.duration_s,
            duration_s: phase.duration_s,
            intervals: phase.intervals || [],
            timeline_group: phase.timeline_group,
            activity_kind: phase.activity_kind,
            display_name: phase.display_name,
            parent_phase: phase.parent_phase,
            activity_rollout_id:
              role.role === "rollout"
                ? timing.source_rollout_id ?? timing.rollout_id
                : timing.training_rollout_id ?? timing.rollout_id,
            activity_rollout_kind:
              role.role === "rollout" ? "source" : "training",
            source_rollout_id:
              timing.source_rollout_id ?? timing.rollout_id,
            training_rollout_id:
              timing.training_rollout_id ?? timing.rollout_id,
            clock_uncertainty_s: role.clock_uncertainty_s,
            execution_sequence: role.execution_sequence,
          };
        }
      }
      steps[String(Number(rollout.rollout_id) + 1)] = phases;
    }
    return steps;
  });
  let visibleLegacyTimingKeys = $derived.by(() => {
    const hasAttemptScopedRollouts = rolloutSummaries.some(
      (rollout) => rollout.training_attempt != null,
    );
    const retryHasNoVisibleRollouts =
      rolloutSummaries.length === 0 &&
      Number(run?.resume_state?.attempt_count) > 1;
    if (!hasAttemptScopedRollouts && !retryHasNoVisibleRollouts) return null;
    const keys = new Set();
    for (const rollout of rolloutSummaries) {
      if (rollout.training_attempt != null) continue;
      const nextKey = String(Number(rollout.rollout_id) + 1);
      const directKey = String(rollout.rollout_id);
      if (run?.step_times?.[nextKey] || run?.substep_times?.[nextKey]) {
        keys.add(nextKey);
      } else if (
        run?.step_times?.[directKey] ||
        run?.substep_times?.[directKey]
      ) {
        keys.add(directKey);
      }
    }
    return keys;
  });
  let displayedStepTimes = $derived.by(() => {
    const legacy = Object.fromEntries(
      Object.entries(run?.step_times || {}).filter(
        ([key]) =>
          visibleLegacyTimingKeys === null || visibleLegacyTimingKeys.has(key),
      ),
    );
    const merged = { ...legacy, ...liveStepTimes };
    return Object.keys(merged).length ? merged : null;
  });
  let displayedSubstepTimes = $derived.by(() => {
    const legacy = Object.fromEntries(
      Object.entries(run?.substep_times || {}).filter(
        ([key]) =>
          visibleLegacyTimingKeys === null || visibleLegacyTimingKeys.has(key),
      ),
    );
    const merged = { ...legacy, ...liveSubstepTimes };
    return Object.keys(merged).length ? merged : null;
  });
  const rolloutColumns = [
    { key: "step", label: "Step", width: 72, minWidth: 56 },
    { key: "mean", label: "Mean reward", width: 118, minWidth: 96 },
    { key: "samples", label: "Samples", width: 80, minWidth: 64 },
    { key: "when", label: "When", width: 88, minWidth: 64 },
  ];

  // Per-step advantage distribution summaries (one row per step, each with the
  // step's overall stats + quantiles) — drives the advantage fan chart.
  let advantageSteps = $state([]);
  let hasAdvantages = $derived(advantageSteps.length > 0);

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
    // Loop instead of Math.min(...arr): a single rollout's per-sample array can
    // exceed the engine's max argument count and make the spread throw a
    // RangeError (same failure class buildDist avoids).
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of scores) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
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
    const loadVersion = ++rolloutLoadVersion;
    try {
      const rows = await fetchRunRollouts(runId, { signal });
      if (signal?.aborted || loadVersion !== rolloutLoadVersion) return;
      const wasEmpty = rolloutSummaries.length === 0;
      rolloutSummaries = rows;
      if (expandedRolloutId !== null) {
        const selected = rows.find(
          (row) => row.rollout_id === expandedRolloutId,
        );
        if (!selected) {
          expandedRolloutId = null;
          expandedRolloutAttempt = null;
          expandedRollout = null;
        } else if (
          (selected.training_attempt ?? null) !== expandedRolloutAttempt
        ) {
          void loadExpandedRollout(
            expandedRolloutId,
            selected.training_attempt,
          );
        }
      }
      rolloutsError = "";

      // Reveal the first rollout
      if (wasEmpty && rolloutSummaries.length > 0 && expandedRolloutId === null) {
        toggleRolloutDetail(rolloutSummaries[0].rollout_id);
      }
    } catch (err) {
      if (signal?.aborted || loadVersion !== rolloutLoadVersion) return;
      // Keep the rollouts we already have on a transient poll failure — only
      // surface the error when there's nothing to show, so the charts/table
      // don't flip to an error message (and back) every flaky 5s poll.
      if (!rolloutSummaries.length) rolloutsError = String(err?.message || err);
    } finally {
      if (loadVersion === rolloutLoadVersion) rolloutsLoading = false;
    }
  }

  async function loadAdvantages(signal) {
    if (!runId) return;
    try {
      const rows = await fetchRunAdvantages(runId, { signal });
      if (signal?.aborted) return;
      advantageSteps = rows;
    } catch {
      // Advantage data is optional (only present once the slime hook has
      // reported a step) — keep whatever we have on a transient failure.
    }
  }

  // Reset rollout state when the run changes (separate from the fetch effect
  // so flipping between the summary/rollouts tabs doesn't clear what's loaded).
  $effect(() => {
    runId;
    rolloutLoadVersion += 1;
    rolloutSummaries = [];
    substepTimingByAttemptAndRollout = {};
    pendingSubstepTimingKeys = new Set();
    rolloutsError = "";
    expandedRolloutId = null;
    expandedRolloutAttempt = null;
    expandedRollout = null;
    advantageSteps = [];
    closeBucket();
  });

  // Load advantage distributions while the Summary tab is active; poll so new
  // steps stream in on a running run.
  $effect(() => {
    const id = runId;
    if (!id || activeTab !== "summary") return;

    const controller = new AbortController();
    void loadAdvantages(controller.signal);
    const interval = window.setInterval(() => {
      const status = String(run?.status || "").toLowerCase();
      if (status && status !== "running") return;
      void loadAdvantages(controller.signal);
    }, 5000);

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
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

  $effect(() => {
    const id = runId;
    const pendingKeys = pendingSubstepTimingKeys;
    const missing = rolloutSummaries
      .filter((rollout) => rollout.training_attempt != null)
      .filter(
        (rollout) =>
          !substepTimingByAttemptAndRollout[
            `${rollout.training_attempt}:${rollout.rollout_id}`
          ] &&
          !pendingKeys.has(
            `${rollout.training_attempt}:${rollout.rollout_id}`,
          ),
      )
      .map((rollout) => ({
        key: `${rollout.training_attempt}:${rollout.rollout_id}`,
        training_attempt: rollout.training_attempt,
        rollout_id: rollout.rollout_id,
      }));
    if (!id || !missing.length) return;

    for (const item of missing) pendingKeys.add(item.key);
    void (async () => {
      try {
        const requests = [];
        for (let offset = 0; offset < missing.length; offset += 512) {
          requests.push(
            fetchSubstepTimings(
              id,
              missing.slice(offset, offset + 512).map((item) => ({
                training_attempt: item.training_attempt,
                rollout_id: item.rollout_id,
              })),
            ),
          );
        }
        const timings = (await Promise.all(requests)).flat();
        if (id !== runId || !timings.length) return;
        substepTimingByAttemptAndRollout = {
          ...substepTimingByAttemptAndRollout,
          ...Object.fromEntries(
            timings.map((timing) => [
              `${timing.training_attempt}:${timing.rollout_id}`,
              timing,
            ]),
          ),
        };
      } catch {
        return;
      } finally {
        for (const item of missing) pendingKeys.delete(item.key);
      }
    })();
  });

  async function toggleRolloutDetail(rolloutId) {
    if (!runId) return;
    if (expandedRolloutId === rolloutId) {
      expandedRolloutId = null;
      expandedRolloutAttempt = null;
      expandedRollout = null;
      closeBucket();
      return;
    }
    expandedRolloutId = rolloutId;
    const summary = rolloutSummaries.find(
      (rollout) => rollout.rollout_id === rolloutId,
    );
    await loadExpandedRollout(rolloutId, summary?.training_attempt);
  }

  async function loadExpandedRollout(rolloutId, trainingAttempt) {
    if (!runId || expandedRolloutId !== rolloutId) return;
    expandedRolloutAttempt = trainingAttempt ?? null;
    expandedRollout = null;
    closeBucket();
    expandedRolloutLoading = true;
    try {
      const detail = await fetchRollout(runId, rolloutId, trainingAttempt);
      if (
        expandedRolloutId === rolloutId &&
        expandedRolloutAttempt === (trainingAttempt ?? null)
      ) {
        expandedRollout = detail;
        const d = sampleDist;
        const first = d ? d.buckets.findIndex((b) => b.length > 0) : -1;
        if (first >= 0) openBucket(first);
      }
    } finally {
      if (
        expandedRolloutId === rolloutId &&
        expandedRolloutAttempt === (trainingAttempt ?? null)
      ) {
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
    if (tab !== "logs" || !id || !status || status !== "running" || paused) {
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

    const es = new EventSource(url);
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
          pendingLogLines.push({ id: logSeq++, task_id, line: p, ts });
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

  // ── Historical logs ─────
  let isRunning = $derived(runStatus === "running");

  let histLines = $state([]), histNewerWindows = $state([]);
  let histLoading = $state(false),
    histLoadingOlder = $state(false),
    histLoadingNewer = $state(false);
  let histError = $state(""), histHasMore = $state(false);
  let histNextUntil = $state(null), histTailEl = $state(null);
  let histSeq = 0, histLastScrollTop = 0;
  let histController = null, histRestoringScroll = false;

  let histRangeInput = $state({ since: "", until: "" });
  let histRange = $state({ since: "", until: "" });

  function epochToLocalInput(epoch) {
    if (!epoch) return "";
    const d = new Date(Number(epoch) * 1000);
    const p = (x) => String(x).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }
  function localInputToEpoch(str) {
    if (!str || !str.trim()) return null;
    const ms = new Date(str.trim().replace(" ", "T")).getTime();
    return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
  }

  function histRangeFromText(sinceText, untilText) {
    const since = localInputToEpoch(sinceText);
    const until = localInputToEpoch(untilText);
    return {
      since: since != null ? String(since) : "",
      // The picker has minute precision, so include the entire final minute.
      until: until != null ? String(until + 59.999999999) : "",
    };
  }

  $effect(() => {
    const { since, until } = histRangeInput;
    const handle = window.setTimeout(() => {
      histRange = histRangeFromText(since, until);
    }, 350);
    return () => window.clearTimeout(handle);
  });

  // Seed the pickers with the run's lifetime the first time we view a finished
  // run.
  let histPrefilledFor = null;
  $effect(() => {
    const id = runId;
    const running = isRunning;
    if (!id || running || histPrefilledFor === id) return;

    const startedAt = run?.started_at || run?.created_at || 0;
    const endedAt = run?.ended_at || run?.completed_at || 0;
    if (!startedAt && !endedAt) return;

    histPrefilledFor = id;
    const sinceText = epochToLocalInput(startedAt);
    const untilText = epochToLocalInput(endedAt);
    const range = histRangeFromText(sinceText, untilText);
    histRangeInput = { since: sinceText, until: untilText };
    histRange = range;
  });

  // Expand server entries into per-line rows (a single ClickHouse entry can
  // carry an embedded newline), matching how the live stream splits lines.
  function pushHistRows(target, entries) {
    for (const entry of entries) {
      const task_id = entry.task_id || "";
      const ts = entry.ts || 0;
      const ts_ns = entry.ts_ns || 0;
      for (const part of String(entry.line ?? "").split(/\r?\n/)) {
        if (!part.length) continue;
        target.push({ id: histSeq++, task_id, line: part, ts, ts_ns });
      }
    }
  }

  function resetHist() {
    histLines = [];
    histError = "";
    histHasMore = false;
    histNewerWindows = [];
    histNextUntil = null;
    histLoading = false;
    histLoadingOlder = false;
    histLoadingNewer = false;
    histLastScrollTop = 0;
    histRestoringScroll = false;
  }

  function histRowTime(row) {
    if (!row) return 0;
    if (row.ts_ns) return Number(row.ts_ns) / 1_000_000_000;
    return Number(row.ts) || 0;
  }

  function histCursorBefore(row) {
    if (!row) return null;
    if (row.ts_ns) return (Number(row.ts_ns) - 1) / 1_000_000_000;
    const ts = Number(row.ts) || 0;
    return ts > 0 ? ts - 0.000000001 : null;
  }

  // Capture a visible row and its exact viewport position. Restoring this
  // after replacing the bounded window avoids any perceptible jump.
  function captureHistAnchor() {
    const el = histTailEl;
    if (!el) return null;
    const containerTop = el.getBoundingClientRect().top;
    const rows = el.querySelectorAll("[data-hist-id]");
    for (const node of rows) {
      const rect = node.getBoundingClientRect();
      if (rect.bottom >= containerTop) {
        return {
          id: node.dataset.histId,
          top: rect.top,
          scrollTop: el.scrollTop,
        };
      }
    }
    return null;
  }

  async function restoreHistAnchor(anchor) {
    if (!histTailEl || !anchor) return;
    histRestoringScroll = true;
    await tick();
    if (!histTailEl) {
      histRestoringScroll = false;
      return;
    }
    const node = histTailEl.querySelector(`[data-hist-id="${anchor.id}"]`);
    if (node) {
      histTailEl.scrollTop += node.getBoundingClientRect().top - anchor.top;
    } else {
      histTailEl.scrollTop = anchor.scrollTop;
    }
    histLastScrollTop = histTailEl.scrollTop;
    requestAnimationFrame(() => {
      if (histTailEl) histLastScrollTop = histTailEl.scrollTop;
      histRestoringScroll = false;
    });
  }

  function rememberTrimmedNewer(kept, dropped) {
    if (!kept.length || !dropped.length) return;
    const since = histRowTime(kept[kept.length - 1]);
    const until = histRowTime(dropped[dropped.length - 1]);
    if (since > 0 && until >= since) {
      histNewerWindows = [...histNewerWindows, { since, until }];
    }
  }

  function markTrimmedOlder(kept, dropped) {
    if (!kept.length || !dropped.length) return;
    histHasMore = true;
    histNextUntil = histCursorBefore(kept[0]);
  }

  async function loadHistInitial(id, { search, since, until }, signal) {
    histLoading = true;
    histError = "";
    try {
      const data = await fetchRunLogs(id, {
        maxLines: HIST_PAGE,
        search,
        since,
        until,
        signal,
      });
      if (signal?.aborted) return;
      histNewerWindows = [];
      const rows = [];
      pushHistRows(rows, data.logs);
      const droppedOlder =
        rows.length > HIST_BUFFER_MAX
          ? rows.slice(0, rows.length - HIST_BUFFER_MAX)
          : [];
      histLines =
        rows.length > HIST_BUFFER_MAX
          ? rows.slice(-HIST_BUFFER_MAX)
          : rows;
      if (droppedOlder.length) {
        markTrimmedOlder(histLines, droppedOlder);
      } else {
        histHasMore = data.hasMore;
        histNextUntil = data.nextUntil;
      }
      // Land on the newest line, like the live tail does.
      queueMicrotask(() => {
        if (histTailEl) {
          histRestoringScroll = true;
          histTailEl.scrollTop = histTailEl.scrollHeight;
          histLastScrollTop = histTailEl.scrollTop;
          requestAnimationFrame(() => {
            histRestoringScroll = false;
          });
        }
      });
    } catch (err) {
      if (signal?.aborted) return;
      histError = String(err?.message || err);
    } finally {
      if (!signal?.aborted) histLoading = false;
    }
  }

  async function loadHistPage(direction, { since, until }) {
    const loadingOlder = direction === "older";
    if (loadingOlder) {
      histLoadingOlder = true;
    } else {
      histLoadingNewer = true;
    }
    histError = "";
    const anchor = captureHistAnchor();
    const signal = histController?.signal;
    try {
      const data = await fetchRunLogs(runId, {
        since,
        until,
        maxLines: HIST_PAGE,
        search: logSearch,
        signal,
      });
      if (signal?.aborted) return;
      const rows = [];
      pushHistRows(rows, data.logs);
      if (loadingOlder && !rows.length) {
        histHasMore = false;
        histNextUntil = null;
        return;
      }

      const adjacent =
        rows.length <= HIST_PAGE
          ? rows
          : loadingOlder
            ? rows.slice(-HIST_PAGE)
            : rows.slice(0, HIST_PAGE);
      const merged = loadingOlder
        ? adjacent.concat(histLines)
        : histLines.concat(adjacent);

      if (loadingOlder) {
        const dropped = merged.slice(HIST_BUFFER_MAX);
        histLines = merged.slice(0, HIST_BUFFER_MAX);
        rememberTrimmedNewer(histLines, dropped);
        if (rows.length > HIST_PAGE) {
          histHasMore = true;
          histNextUntil = histCursorBefore(adjacent[0]);
        } else {
          histHasMore = data.hasMore;
          histNextUntil = data.nextUntil;
        }
      } else {
        const dropped = merged.slice(
          0,
          Math.max(0, merged.length - HIST_BUFFER_MAX),
        );
        histLines = merged.slice(-HIST_BUFFER_MAX);
        markTrimmedOlder(histLines, dropped);
        histNewerWindows = histNewerWindows.slice(0, -1);
      }

      await restoreHistAnchor(anchor);
    } catch (err) {
      if (signal?.aborted) return;
      histError = String(err?.message || err);
    } finally {
      if (!signal?.aborted) {
        if (loadingOlder) {
          histLoadingOlder = false;
        } else {
          histLoadingNewer = false;
        }
      }
    }
  }

  function loadHistOlder() {
    if (
      !histHasMore ||
      histNextUntil == null ||
      histLoadingOlder ||
      histLoadingNewer ||
      histLoading
    ) {
      return;
    }
    return loadHistPage("older", {
      since: histRange.since,
      until: histNextUntil,
    });
  }

  function loadHistNewer() {
    if (
      !histNewerWindows.length ||
      histLoadingNewer ||
      histLoadingOlder ||
      histLoading
    ) {
      return;
    }
    const range = histNewerWindows[histNewerWindows.length - 1];
    return loadHistPage("newer", range);
  }

  // Load only in the direction the user moved. Scroll restoration after a
  // page swap is ignored, preventing older/newer fetches from ping-ponging.
  function onHistScroll() {
    const el = histTailEl;
    if (!el || histRestoringScroll) return;
    const top = el.scrollTop;
    const delta = top - histLastScrollTop;
    histLastScrollTop = top;
    if (delta < 0 && top <= 40) {
      void loadHistOlder();
      return;
    }
    if (delta > 0) {
      const distanceFromBottom = el.scrollHeight - top - el.clientHeight;
      if (distanceFromBottom <= 40) void loadHistNewer();
    }
  }

  function jumpHistToLatest() {
    if (!runId || histLoading || histLoadingOlder || histLoadingNewer) return;
    void loadHistInitial(
      runId,
      {
        search: logSearch,
        since: histRange.since,
        until: histRange.until,
      },
      histController?.signal,
    );
  }

  function resetHistRange() {
    const startedAt = run?.started_at || run?.created_at || 0;
    const endedAt = run?.ended_at || run?.completed_at || 0;
    histRangeInput = {
      since: epochToLocalInput(startedAt),
      until: epochToLocalInput(endedAt),
    };
  }

  // Load the newest page when the Logs tab opens on a finished run, and reload
  // when the debounced search/since/until change.
  $effect(() => {
    const id = runId;
    const tab = activeTab;
    const status = runStatus;
    const running = isRunning;
    const search = logSearch;
    const { since, until } = histRange;

    resetHist();
    if (tab !== "logs" || !id || !status || running) return;

    const controller = new AbortController();
    histController = controller;
    void loadHistInitial(id, { search, since, until }, controller.signal);
    return () => {
      controller.abort();
      if (histController === controller) histController = null;
    };
  });

  function _seriesStats(getY) {
    if (!rolloutSummaries.length) return null;
    const values = rolloutSummaries.map(getY);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      latest: values[values.length - 1],
    };
  }

  let chartStats = $derived(_seriesStats((r) => Number(r.mean) || 0));
  let rewardChartData = $derived(
    rolloutSummaries.map((r) => ({
      x: Number(r.rollout_id) || 0,
      y: Number(r.mean) || 0,
      rollout_id: Number(r.rollout_id) || 0,
    })),
  );

  // Custom reward-function tags: any numeric sample.metadata key a reward fn
  // sets (e.g. sample.metadata["step_K"] = k) gets aggregated per rollout on
  // the backend (TrainingRolloutResult.tag_stats) and charted here the same
  // way as the reward line above — one chart per discovered tag name.
  let customTagNames = $derived(
    Array.from(
      new Set(rolloutSummaries.flatMap((r) => Object.keys(r.tag_stats || {}))),
    ).sort(),
  );

  function tagChartData(tag) {
    return rolloutSummaries
      .filter((r) => r.tag_stats?.[tag])
      .map((r) => ({
        x: Number(r.rollout_id) || 0,
        y: Number(r.tag_stats[tag].mean) || 0,
        rollout_id: Number(r.rollout_id) || 0,
      }));
  }

  function tagChartStats(tag) {
    const values = rolloutSummaries
      .filter((r) => r.tag_stats?.[tag])
      .map((r) => Number(r.tag_stats[tag].mean) || 0);
    if (!values.length) return null;
    return { min: Math.min(...values), max: Math.max(...values), latest: values[values.length - 1] };
  }

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

  // Grouped-bar histogram of two value sets (first vs latest), sharing a common
  // [lo, hi] range so the bars line up. Drives both the score and advantage
  // before/after comparison charts.
  function buildDist(firstValues, lastValues, firstId, lastId) {
    if (!firstValues.length && !lastValues.length) return null;
    // Loop instead of Math.min(...arr): advantage arrays can exceed the
    // engine's max argument count and make the spread throw a RangeError.
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of firstValues) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    for (const v of lastValues) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const n = lo === hi ? 1 : 12;
    const span = hi - lo || 1;
    const bins = Array.from({ length: n }, (_, i) => ({
      lo: lo + (i / n) * span,
      hi: lo + ((i + 1) / n) * span,
      first: 0,
      last: 0,
    }));
    const idx = (v) => Math.max(0, Math.min(n - 1, Math.floor(((v - lo) / span) * n)));
    for (const v of firstValues) bins[idx(v)].first += 1;
    for (const v of lastValues) bins[idx(v)].last += 1;
    const max = Math.max(1, ...bins.map((b) => Math.max(b.first, b.last)));
    return { bins, max, lo, hi, firstId, lastId };
  }

  function buildScoreDist(firstSamples, lastSamples, firstId, lastId) {
    return buildDist(
      firstSamples.map((s) => Number(s.score) || 0),
      lastSamples.map((s) => Number(s.score) || 0),
      firstId,
      lastId,
    );
  }

  // Adapt a buildDist result into ComparativeBarChart inputs: one category per
  // bin (value range, shown only in the tooltip — the endpoint axis is drawn
  // separately), and a series per rollout endpoint. The latest series is dropped
  // when the first and latest rollout are the same one.
  function distCategories(dist) {
    return dist.bins.map((b) => `${formatMean(b.lo)}–${formatMean(b.hi)}`);
  }

  function distSeries(dist) {
    const series = [
      {
        name: `rollout ${dist.firstId}`,
        color: "var(--color-c-gray-40)",
        values: dist.bins.map((b) => b.first),
      },
    ];
    if (dist.firstId !== dist.lastId) {
      series.push({
        name: `latest (rollout ${dist.lastId})`,
        color: "var(--accent)",
        values: dist.bins.map((b) => b.last),
      });
    }
    return series;
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
        const firstAttempt = rolloutSummaries.find(
          (rollout) => rollout.rollout_id === fId,
        )?.training_attempt;
        const lastAttempt = rolloutSummaries.find(
          (rollout) => rollout.rollout_id === lId,
        )?.training_attempt;
        const first = await fetchRollout(id, fId, firstAttempt);
        const last =
          fId === lId
            ? first
            : await fetchRollout(id, lId, lastAttempt);
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

  // Advantage-distribution comparison: raw advantages of the first vs latest
  // step, bucketed the same way as the score comparison. The merged-step
  // payloads are large, so only refetch when the endpoint ids change.
  let advantageDist = $state(null);

  function mergedAdvantages(merged) {
    return (merged?.groups || []).flatMap((g) =>
      (g?.advantages || []).map((v) => Number(v) || 0),
    );
  }

  $effect(() => {
    if (activeTab !== "summary") return;
    const id = runId;
    const fId = firstRolloutId;
    const lId = lastRolloutId;
    if (!id || fId == null || lId == null) {
      advantageDist = null;
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const first = await fetchRunAdvantageStep(id, fId);
        const last = fId === lId ? first : await fetchRunAdvantageStep(id, lId);
        if (cancelled) return;
        advantageDist = buildDist(
          mergedAdvantages(first),
          mergedAdvantages(last),
          fId,
          lId,
        );
      } catch {
        if (!cancelled) advantageDist = null;
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
    <header class="flex flex-wrap items-center gap-x-[10px] gap-y-[8px] p-[0_24px] mb-[16px] max-[900px]:p-[0_16px]">
      <button type="button" class="inline-flex items-center gap-[6px] [background:none] [border:0] text-(--muted) cursor-pointer text-[13px] leading-[16px] min-h-[32px] p-[4px_8px] rounded-[6px] hover:text-(--text) hover:bg-(--color-c-gray-10,#2f2f2f) max-[900px]:basis-full" onclick={onBack}>
        <ArrowLeft size={14} strokeWidth={2.1} />
        <span>Back to runs</span>
      </button>
      {#if onCollapse}
        <button type="button" class="inline-flex items-center gap-[6px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[6px] [background:none] text-(--muted) cursor-pointer [font:inherit] text-[12px] font-medium leading-[16px] min-h-[32px] p-[4px_8px] hover:text-(--text-bright) hover:border-(--border-strong,#4a4a4a)" onclick={onCollapse} title="Collapse to drawer">
          <Minimize2 size={12} strokeWidth={2.1} />
          <span>Collapse</span>
        </button>
      {/if}
      {#each wandbLinks as link (link.url)}
        <a
          class="header-link wandb-link inline-flex items-center gap-[6px] min-h-[32px] leading-[16px]"
          href={link.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span>{link.label}</span>
          <ExternalLink size={12} strokeWidth={2.1} />
        </a>
      {/each}
      {#if run?.modal_app_url}
        <a
          class="header-link inline-flex items-center gap-[6px] min-h-[32px] leading-[16px]"
          href={run.modal_app_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span>Open in Modal</span>
          <ExternalLink size={12} strokeWidth={2.1} />
        </a>
      {/if}
    </header>
  {/if}

  {#if !run}
    <div class="detail-empty px-[24px]">Loading run {runId}…</div>
  {:else}
    {#if !embedded}
    <div class="flex items-center gap-[16px] p-[0_24px] mb-[16px] max-[900px]:p-[0_16px] min-w-0">
      <h1 class="text-[22px] font-[600] text-(--text-bright) m-0 overflow-hidden text-ellipsis whitespace-nowrap min-w-0" title={run.run_id}>{run.run_id}</h1>
      <StatusPill status={getStatus(run)} />
      {#if resumeBadge(run)}
        <span class="[border:1px_solid_color-mix(in_srgb,var(--yellow,#fbbf24)_42%,transparent)] rounded-[999px] bg-[color-mix(in_srgb,var(--yellow,#fbbf24)_10%,transparent)] text-(--yellow,#fbbf24) text-[12px] leading-[16px] p-[2px_8px] whitespace-nowrap">{resumeBadge(run)}</span>
      {/if}
    </div>
    {/if}

    <Tabs
      active={activeTab}
      onSelect={selectTab}
      tabs={[
        { value: "summary", label: "Summary" },
        { value: "rollouts", label: "Rollouts", count: rolloutSummaries.length || undefined },
        { value: "logs", label: "Logs" },
      ]}
    />

    {#if activeTab === "summary"}
      <div class="summary-tab">
        <div class="summary-tab-main">
          {#if run?.error_message}
            <div class="mb-[20px]">
              <div class="text-(--red,#f87171) text-[12px] font-[600] tracking-[0.02em] mb-[6px] uppercase">Error</div>
              <pre class="[border:1px_solid_color-mix(in_srgb,var(--red,#f87171)_45%,transparent)] rounded-[8px] bg-[color-mix(in_srgb,var(--red,#f87171)_12%,transparent)] text-(--red,#f87171) [font-family:var(--font-mono)] text-[12px] leading-[17px] m-0 max-h-[320px] overflow-auto p-[12px_14px] whitespace-pre-wrap [word-break:break-word]">{run.error_message}</pre>
            </div>
          {/if}
          {#if displayedStepTimes || displayedSubstepTimes}
            <div class="rollout-chart">
              <div class="rollout-chart-title">Framework activity timeline</div>
              <div class="chart-scroll">
                <StepTimings
                  stepTimes={displayedStepTimes}
                  substepTimes={displayedSubstepTimes}
                  rolloutStats={rolloutSummaries}
                  layout="timeline"
                  downloadName={`step_substep_times_${runId}.json`}
                />
              </div>
            </div>
          {/if}
          {#if rolloutsLoading && !rolloutSummaries.length}
            <div class="rollout-chart">
              <ChartSkeleton variant="line" height={140} showTitle />
            </div>
            <div class="rollout-chart">
              <div class="rollout-chart-title">Score distribution</div>
              <ChartSkeleton variant="bars" height={120} />
            </div>
            <div class="chart-grid">
              <div class="rollout-chart">
                <div class="rollout-chart-title">Advantage spread over time</div>
                <ChartSkeleton variant="line" height={200} />
              </div>
              <div class="rollout-chart">
                <div class="rollout-chart-title">Advantage distribution over time</div>
                <ChartSkeleton variant="violins" height={210} />
              </div>
            </div>
          {:else if rolloutsError}
            <div class="detail-empty">Failed to load rollouts: {rolloutsError}</div>
          {:else if !rolloutSummaries.length}
            <div class="detail-empty">No rollouts recorded yet.</div>
          {:else}
            <div class="rollout-chart">
              <div class="chart-scroll">
                <LineChart
                  title="Reward"
                  data={rewardChartData}
                  formatX={(row) => `rollout ${row.rollout_id}`}
                  formatY={(value) => formatMean(value)}
                  ariaLabel="Reward chart"
                />
              </div>
              {#if chartStats}
                <div class="flex flex-wrap gap-[16px] mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
                  <span>min {formatMean(chartStats.min)}</span>
                  <span>latest {formatMean(chartStats.latest)}</span>
                  <span>max {formatMean(chartStats.max)}</span>
                </div>
              {/if}
            </div>

            <!-- Score distribution: second graph, above the advantage graphs. -->
            <div class="rollout-chart">
              <div class="rollout-chart-title">Score distribution</div>
              <div class="chart-scroll">
                {#if scoreDist}
                  <ComparativeBarChart
                    categories={distCategories(scoreDist)}
                    series={distSeries(scoreDist)}
                    height={120}
                    showCategoryLabels={false}
                    format={(v) => `${v}`}
                  />
                  <div class="dist-axis">
                    <span>{formatMean(scoreDist.lo)}</span>
                    <span class="dist-axis-label">reward</span>
                    <span>{formatMean(scoreDist.hi)}</span>
                  </div>
                {:else}
                  <ChartSkeleton variant="bars" height={120} />
                {/if}
              </div>
            </div>

            {#if hasAdvantages}
              <div class="chart-grid">
                <div class="rollout-chart">
                  <div class="rollout-chart-title">Advantage spread over time</div>
                  <AdvantageSpreadChart steps={advantageSteps} />
                </div>
                <div class="rollout-chart">
                  <div class="rollout-chart-title">Advantage distribution over time</div>
                  <AdvantageViolins steps={advantageSteps} />
                </div>
                <div class="rollout-chart">
                  <div class="rollout-chart-title">Advantage distribution: rollout {firstRolloutId} vs latest</div>
                  {#if advantageDist}
                    <ComparativeBarChart
                      categories={distCategories(advantageDist)}
                      series={distSeries(advantageDist)}
                      height={120}
                      showCategoryLabels={false}
                      format={(v) => `${v}`}
                    />
                    <div class="dist-axis">
                      <span>{formatMean(advantageDist.lo)}</span>
                      <span class="dist-axis-label">advantage</span>
                      <span>{formatMean(advantageDist.hi)}</span>
                    </div>
                  {:else}
                    <ChartSkeleton variant="bars" height={120} />
                  {/if}
                </div>
              </div>
            {/if}

            {#if customTagNames.length}
              <div class="chart-grid">
                {#each customTagNames as tag (tag)}
                  <div class="rollout-chart">
                    <LineChart
                      title={`${tag} (mean)`}
                      data={tagChartData(tag)}
                      formatX={(row) => `rollout ${row.rollout_id}`}
                      formatY={(value) => formatMean(value)}
                      ariaLabel={`${tag} chart`}
                    />
                    {#if tagChartStats(tag)}
                      <div class="flex gap-[16px] mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
                        <span>min {formatMean(tagChartStats(tag).min)}</span>
                        <span>latest {formatMean(tagChartStats(tag).latest)}</span>
                        <span>max {formatMean(tagChartStats(tag).max)}</span>
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
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
        <div class="detail-empty">Loading rollouts…</div>
      {:else if rolloutsError}
        <div class="detail-empty">Failed to load rollouts: {rolloutsError}</div>
      {:else if !rolloutSummaries.length}
        <div class="detail-empty">No rollouts recorded yet.</div>
      {:else}
        <div class="table-wrap">
        <ResizableTable class="rollout-table" columns={rolloutColumns}>
          <tbody>
            {#each rolloutSummaries as r (r.rollout_id)}
              <tr
                class:expanded={expandedRolloutId === r.rollout_id}
                class:rollout-error={r.error_summary?.verdict === "all_infra_failure"}
                class:rollout-warn={r.error_summary?.verdict === "partial_infra_failure"}
                onclick={() => toggleRolloutDetail(r.rollout_id)}
              >
                <td>#{r.rollout_id}</td>
                <td class="text-(--text-bright)">
                  {formatMean(r.mean)}
                  {#if r.error_summary?.verdict === "all_infra_failure"}
                    <span class="inline-block text-[10px] font-medium p-[1px_6px] rounded-[3px] ml-[6px] align-middle bg-[rgba(239,68,68,0.15)] text-[#ef4444] [border:1px_solid_rgba(239,68,68,0.25)]" title="All samples failed due to infrastructure error">infra failure</span>
                  {:else if r.error_summary?.verdict === "partial_infra_failure"}
                    <span class="inline-block text-[10px] font-medium p-[1px_6px] rounded-[3px] ml-[6px] align-middle bg-[rgba(251,191,36,0.15)] text-[#fbbf24] [border:1px_solid_rgba(251,191,36,0.25)]" title="Some samples failed due to infrastructure error">partial failure</span>
                  {/if}
                </td>
                <td>{r.total}</td>
                <td>
                  <TimeAgo timestamp={r.created_at} showJustNow falsyRepresentation="—" />
                </td>
              </tr>
              {#if expandedRolloutId === r.rollout_id}
                <tr>
                  <td class="p-[12px_10px] bg-(--color-c-gray-08,#1c1c1c) cursor-default" colspan={rolloutColumns.length}>
                    {#if expandedRolloutLoading}
                      <div class="detail-empty">Loading samples…</div>
                    {:else if !expandedRollout || !sampleDist}
                      <div class="detail-empty">No samples recorded.</div>
                    {:else}
                      {@const stepTiming = stepTimingForRollout(r.rollout_id)}
                      {#if stepTiming}
                        <div class="rollout-chart">
                          <div class="rollout-chart-title">Step timing</div>
                          <div class="chart-scroll">
                            <StepTimings
                              stepTimes={stepTiming.stepTimes}
                              substepTimes={stepTiming.substepTimes}
                              layout="rows"
                            />
                          </div>
                        </div>
                      {/if}
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
                            <div class="flex flex-wrap gap-[6px] mb-[4px]">
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
                              <div class="text-[11px] text-(--muted,#a3a3a3) mt-[6px]">
                                Check the Modal app logs for sandbox/image build errors. Common cause: the environment image failed to build.
                              </div>
                            {/if}
                          </div>
                        {/if}
                      {/if}
                      <div class="mb-[16px]">
                        <div class="flex justify-end mb-[6px]">
                          <button
                            type="button"
                            class="inline-flex items-center gap-[5px] [background:none] [border:1px_solid_var(--border,#2f2f2f)] rounded-[4px] text-(--muted) text-[11px] p-[3px_8px] cursor-pointer hover:text-(--text) hover:border-(--border-strong,#4a4a4a)"
                            onclick={downloadAllTrajectories}
                            title="Download all samples as JSON"
                          >
                            <Download size={13} />
                            Download all ({sampleDist.total} samples)
                          </button>
                        </div>
                        <div class="chart-scroll">
                          <div
                            class="flex items-end gap-[2px] h-[120px] pt-[14px] min-w-[280px] [border-bottom:1px_solid_var(--border,#2f2f2f)]"
                            role="group"
                            aria-label="Sample score distribution"
                          >
                            {#each sampleDist.buckets as bucket, b (b)}
                              <button
                                type="button"
                                class="dist-bar"
                                class:detail-active={activeBucket === b}
                                class:is-empty={!bucket.length}
                                style:height={`${(bucket.length / sampleDist.maxCount) * 100}%`}
                                disabled={!bucket.length}
                                title={`${bucket.length} sample${bucket.length === 1 ? "" : "s"} · reward ${bucketRange(b)}`}
                                onclick={() => openBucket(b)}
                              >
                                <span class="absolute top-[-14px] left-0 right-0 text-center text-[10px] text-(--muted) [font-variant-numeric:tabular-nums]">{bucket.length || ""}</span>
                              </button>
                            {/each}
                          </div>
                          <div class="dist-axis">
                            <span>{formatMean(sampleDist.lo)}</span>
                            <span class="dist-axis-label">reward · {sampleDist.total} samples</span>
                            <span>{formatMean(sampleDist.hi)}</span>
                          </div>
                        </div>
                      </div>

                      {#if activeSample}
                        <div class="sample-viewer">
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
                              <span class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
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
                            <div class="sample-viewer-actions">
                              <span class="text-(--text-bright) [font-variant-numeric:tabular-nums]">
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
                          {#if activeSample.sample.metadata?.inference}
                            <div class="rollout-sample-label">inference</div>
                            <InferenceStats inference={activeSample.sample.metadata.inference} />
                          {/if}
                          {#if activeSample.sample.metadata?._metadata_type === "audio" || activeSample.sample.metadata?.audio}
                            <div class="rollout-sample-label">audio</div>
                            <audio
                              class="block w-full max-w-[400px] m-[4px_0_8px] rounded-[4px]"
                              controls
                              preload="none"
                              src={activeSample.sample.metadata.audio}
                            ></audio>
                          {/if}
                          {#if activeSample.sample.metadata?._metadata_type === "image" || activeSample.sample.metadata?.image}
                            <div class="rollout-sample-label">image</div>
                            <img
                              class="block w-full max-w-[400px] h-auto m-[4px_0_8px] rounded-[4px] [border:1px_solid_var(--border)]"
                              src={activeSample.sample.metadata.image}
                              alt="rollout input"
                              loading="lazy"
                            />
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
                            <span class="rollout-sample-metric p-[2px_8px] rounded-[3px] text-[11px]! font-medium" class:exit-ok={activeSample.sample.metadata.exit_status === "ok"} class:exit-err={activeSample.sample.metadata.exit_status !== "ok"}>
                              {activeSample.sample.metadata.exit_status}
                            </span>
                          {/if}
                          {#if activeSample.sample.metadata?.eval_detail}
                            <div class="rollout-sample-label">failure reason</div>
                            <pre class="rollout-sample-text">{activeSample.sample.metadata.eval_detail}</pre>
                          {/if}
                          <!-- Catch-all: any other tag a custom reward/rollout function set on
                               sample.metadata (e.g. sample.metadata["guessing"] = {...}) that
                               isn't one of the known keys rendered explicitly above. -->
                          {#each Object.entries(activeSample.sample.metadata ?? {}).filter(
                            ([key]) =>
                              ![
                                "inference",
                                "_metadata_type",
                                "audio",
                                "image",
                                "trajectory_messages",
                                "eval_report",
                                "reference",
                                "metrics",
                                "exit_status",
                                "eval_detail",
                                "response_length",
                                "prompt_length",
                                "rollout_id",
                                "rollout_idx",
                              ].includes(key),
                          ) as [name, value] (name)}
                            <div class="rollout-sample-label">{name}</div>
                            {#if value !== null && typeof value === "object"}
                              <pre class="rollout-sample-text">{JSON.stringify(value, null, 2)}</pre>
                            {:else}
                              <span class="rollout-sample-metric">{String(value)}</span>
                            {/if}
                          {/each}
                          {#if activeSample.sample.trace?.length}
                            <div class="rollout-sample-label">trajectory timeline</div>
                            <div class="chart-scroll">
                              <SampleTimeline trace={activeSample.sample.trace} />
                            </div>
                          {/if}
                        </div>
                      {:else}
                        <div class="text-[12px] text-(--muted) p-[4px_0]">Click a bar to inspect its samples.</div>
                      {/if}
                    {/if}
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </ResizableTable>
        </div>
      {/if}
      </div>
    {:else if activeTab === "logs"}
      <div class="tab-panel">
      {#if isRunning}
      <div class="flex justify-end mb-[8px]">
        <span class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em]">
          {#if logState === "streaming"}
            <span class="dot bg-[#4ade80]! shadow-[0_0_0_2px_rgba(74,222,128,0.18)]"></span> live
          {:else if logState === "paused"}
            <span class="dot dot-dim"></span> paused
          {:else if logState === "reconnecting"}
            <span class="dot bg-[#fbbf24]!"></span> reconnecting…{#if logError} <span class="log-reconnect-reason">({logError})</span>{/if}
          {:else if logState === "done"}
            <span class="dot dot-dim"></span> finished
          {:else if logState === "error"}
            <span class="dot bg-[#f87171]!"></span> error
          {:else if String(run?.status || "").toLowerCase() !== "running"}
            <span class="dot dot-dim"></span> run not active
          {:else}
            <span class="dot dot-dim"></span> idle
          {/if}
        </span>
      </div>

      <div class="flex flex-wrap items-center gap-[8px] mb-[8px]">
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
          class="[flex:1] min-w-[160px] bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[4px] p-[4px_8px] text-[12px] [font-family:inherit]"
          type="search"
          placeholder="filter substring…"
          bind:value={logSearchInput}
          aria-label="Filter log lines"
        />
        <label class="inline-flex items-center gap-[6px] text-(--muted) text-[11px]">
          <span>Rate cap</span>
          <select class="bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[4px] p-[2px_6px] text-[12px]" bind:value={logRateCap} aria-label="Lines per second cap">
            <option value={0}>off</option>
            <option value={10}>10/s</option>
            <option value={50}>50/s</option>
            <option value={200}>200/s</option>
            <option value={1000}>1000/s</option>
          </select>
        </label>
        <label class="inline-flex items-center gap-[6px] text-(--muted) text-[11px]">
          <input type="checkbox" bind:checked={logFollow} />
          <span>Follow tail</span>
        </label>
      </div>

      {#if logState === "error" && logError}
        <div class="detail-empty">Log stream error: {logError}</div>
      {/if}

      {#if !logLines.length}
        <div class="detail-empty">
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
        <div class="bg-(--color-c-gray-08,#0e0e0e) rounded-[6px] p-[8px_12px] max-h-[420px] overflow-y-auto overflow-x-auto [font-family:ui-monospace,SFMono-Regular,Menlo,monospace] text-[12px] leading-[1.45] text-(--text)" bind:this={logTailEl}>
          {#each logLines as entry (entry.id)}
            <div class="flex gap-[10px] whitespace-pre">
              <span class="shrink-0 text-(--muted) text-[10px] min-w-[64px] overflow-hidden text-ellipsis">{entry.task_id || ""}</span>
              <span class="flex-1 whitespace-pre-wrap break-all">{entry.line}</span>
            </div>
          {/each}
        </div>
        <div class="mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums] flex gap-[6px]">
          <span>
            Showing last {logLines.length} line{logLines.length === 1 ? "" : "s"} (cap {LOG_BUFFER_MAX})
          </span>
          {#if logDropped > 0}
            <span class="text-[#fbbf24]">
              · {logDropped} dropped by rate cap
            </span>
          {/if}
        </div>
      {/if}
      {:else}
        <!-- Finished run: page through the durable copy on demand. -->
        <div class="flex flex-col gap-[12px] mb-[12px] p-[12px_14px] rounded-[8px] bg-(--color-c-gray-08,#161616) [border:1px_solid_var(--border,#2f2f2f)]">
          <div class="flex items-center gap-[12px]">
            <input
              class="flex-1 min-w-0 bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
              type="search"
              placeholder="filter substring…"
              bind:value={logSearchInput}
              aria-label="Filter log lines"
            />
            <span class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em] shrink-0">
              <span class="dot dot-dim"></span> stored logs
            </span>
          </div>
          <div class="flex flex-wrap items-center gap-[10px]">
            <span class="text-(--muted) text-[11px] uppercase tracking-[0.04em]">Time range</span>
            <input
              class="w-[160px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] [font-variant-numeric:tabular-nums] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
              type="text"
              placeholder="YYYY-MM-DD HH:MM"
              bind:value={histRangeInput.since}
              aria-label="Show logs since"
            />
            <span class="text-(--muted-strong) text-[13px]">→</span>
            <input
              class="w-[160px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] [font-variant-numeric:tabular-nums] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
              type="text"
              placeholder="YYYY-MM-DD HH:MM"
              bind:value={histRangeInput.until}
              aria-label="Show logs until"
            />
            <button
              class="log-button text-[11px] px-[10px] py-[4px]"
              onclick={resetHistRange}
              title="Reset to the run's time range"
            >
              Reset
            </button>
          </div>
        </div>

        {#if histError}
          <div class="detail-empty">Failed to load logs: {histError}</div>
        {/if}

        {#if histLoading && !histLines.length}
          <div class="detail-empty">Loading logs…</div>
        {:else if !histLines.length}
          <div class="detail-empty">
            {#if logSearch}
              No log lines matching "{logSearch}".
            {:else}
              No logs recorded for this run.
            {/if}
          </div>
        {:else}
          <div class="bg-(--color-c-gray-08,#0e0e0e) rounded-[6px] p-[8px_12px] max-h-[420px] overflow-y-auto overflow-x-auto [overflow-anchor:none] [font-family:ui-monospace,SFMono-Regular,Menlo,monospace] text-[12px] leading-[1.45] text-(--text)" bind:this={histTailEl} onscroll={onHistScroll}>
            {#if histHasMore}
              <div class="text-center text-[10px] text-(--muted) pb-[6px]">
                {histLoadingOlder ? "Loading older lines…" : "Scroll up for older lines"}
              </div>
            {/if}
            {#each histLines as entry (entry.id)}
              <div class="flex gap-[10px] whitespace-pre" data-hist-id={entry.id}>
                <span class="shrink-0 text-(--muted) text-[10px] min-w-[64px] overflow-hidden text-ellipsis">{entry.task_id || ""}</span>
                <span class="flex-1 whitespace-pre-wrap break-all">{entry.line}</span>
              </div>
            {/each}
            {#if histNewerWindows.length}
              <div class="text-center text-[10px] text-(--muted) pt-[6px]">
                {histLoadingNewer ? "Loading newer lines…" : "Scroll down for newer lines"}
              </div>
            {/if}
          </div>
          <div class="mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums] flex items-center gap-[8px]">
            <span>Showing {histLines.length} line{histLines.length === 1 ? "" : "s"}</span>
            {#if histNewerWindows.length}
              <span class="h-[12px] w-px bg-(--border)"></span>
              <button
                type="button"
                class="text-(--accent) underline-offset-2 hover:underline bg-transparent border-0 p-0 cursor-pointer text-[11px]"
                onclick={jumpHistToLatest}
              >
                Jump to latest
              </button>
            {/if}
          </div>
        {/if}
      {/if}
      </div>
    {/if}
  {/if}
</section>
