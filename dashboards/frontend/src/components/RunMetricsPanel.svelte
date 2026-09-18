<script>
  // Metrics tab body for a training run: the scalar series mirrored from
  // wandb/trackio, laid out like W&B's workspace — one panel group per key
  // prefix (`train/`, `rollout/`, ...), a LineChart per key, a search box
  // that narrows every group at once, and a shared step window so brushing
  // one chart zooms them all.
  import ChartSkeleton from "./ChartSkeleton.svelte";
  import GroupSection from "./GroupSection.svelte";
  import LineChart from "./LineChart.svelte";
  import { fetchRunMetrics } from "../lib/api.js";
  import {
    formatMetricValue,
    groupMetricKeys,
    seriesStats,
    seriesToRows,
  } from "../lib/metricSeries.js";

  let { runId, isRunning = false } = $props();

  const POLL_MS = 5000;
  // Rendering hundreds of SVG charts at once is what makes W&B workspaces
  // crawl; past this many panels we ask for a narrower search instead.
  const MAX_CHARTS = 60;

  let payload = $state(null);
  let loading = $state(true);
  let error = $state("");
  let searchInput = $state("");
  let search = $state("");
  let collapsed = $state(new Set());
  let stepDomain = $state(null); // [lo, hi] steps while zoomed; null → all

  $effect(() => {
    const value = searchInput;
    const timer = window.setTimeout(() => {
      search = value.trim();
    }, 150);
    return () => window.clearTimeout(timer);
  });

  let controller = null;
  let inflight = false;

  async function load(id) {
    if (inflight) return;
    inflight = true;
    const signal = controller?.signal;
    try {
      const next = await fetchRunMetrics(id, { signal });
      if (signal?.aborted) return;
      payload = next;
      error = "";
    } catch (err) {
      if (signal?.aborted || err?.name === "AbortError") return;
      error = err instanceof Error ? err.message : String(err);
    } finally {
      inflight = false;
      if (!signal?.aborted) loading = false;
    }
  }

  // Initial load, re-done from scratch when the run changes.
  $effect(() => {
    const id = runId;
    if (!id) return;
    payload = null;
    loading = true;
    error = "";
    collapsed = new Set();
    stepDomain = null;
    controller = new AbortController();
    load(id);
    return () => {
      controller?.abort();
      controller = null;
    };
  });

  // Poll while the run is live; a finished run's history is fetched once.
  $effect(() => {
    const id = runId;
    if (!id || !isRunning) return;
    const interval = window.setInterval(() => load(id), POLL_MS);
    return () => window.clearInterval(interval);
  });

  let allKeys = $derived(payload?.keys ?? []);
  let groups = $derived(groupMetricKeys(allKeys, search));
  let visibleKeyCount = $derived(groups.reduce((n, g) => n + g.keys.length, 0));

  // Chart rows are memoised per key so a poll that leaves a series untouched
  // doesn't rebuild every panel.
  let rowsByKey = $derived.by(() => {
    const out = new Map();
    const series = payload?.series ?? {};
    for (const key of allKeys) out.set(key, seriesToRows(series[key]));
    return out;
  });

  let stepRange = $derived.by(() => {
    const range = payload?.step_range;
    if (!Array.isArray(range) || range.length !== 2) return null;
    const [lo, hi] = range.map(Number);
    return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : null;
  });

  // Charts rendered so far, in group order, capped at MAX_CHARTS.
  let panels = $derived.by(() => {
    let budget = MAX_CHARTS;
    return groups.map((group) => {
      const keys = collapsed.has(group.name) ? [] : group.keys.slice(0, Math.max(0, budget));
      budget -= keys.length;
      return { ...group, shown: keys, truncated: group.keys.length - keys.length };
    });
  });
  let hiddenByCap = $derived(
    panels.reduce((n, g) => n + (collapsed.has(g.name) ? 0 : g.truncated), 0),
  );

  function toggleGroup(name) {
    const next = new Set(collapsed);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    collapsed = next;
  }

  function setAllCollapsed(value) {
    collapsed = value ? new Set(groups.map((g) => g.name)) : new Set();
  }

  function onDomainChange(domain) {
    if (!domain) {
      stepDomain = null;
      return;
    }
    const lo = Math.floor(domain[0]);
    const hi = Math.ceil(domain[1]);
    stepDomain = hi > lo ? [lo, hi] : null;
  }

  function formatStep(row) {
    return `step ${row?.x ?? ""}`;
  }

  function subtitleFor(group) {
    const n = group.keys.length;
    return `${n} ${n === 1 ? "metric" : "metrics"}`;
  }
</script>

<div class="flex flex-col gap-[16px] min-w-0">
  {#if loading && !payload}
    <div class="chart-grid" aria-busy="true" aria-label="Loading metrics">
      {#each [0, 1, 2, 3] as i (i)}
        <div class="rollout-chart">
          <ChartSkeleton variant="line" height={140} showTitle />
        </div>
      {/each}
    </div>
  {:else if error && !payload}
    <div class="detail-empty" role="alert">
      Couldn't load metrics: {error}
    </div>
  {:else if !allKeys.length}
    <div class="detail-empty">
      <div class="text-(--text-bright) mb-[6px]">No metrics reported yet.</div>
      <div class="max-w-[64ch] leading-[1.5]">
        Scalars the framework logs through <code>wandb.log</code> show up here
        for every metric provider.
        {#if isRunning}The first points arrive once training logs its first step.{/if}
      </div>
    </div>
  {:else}
    <div class="flex flex-wrap items-center gap-[10px]">
      <input
        class="flex-1 min-w-[200px] bg-(--color-c-gray-08,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
        type="search"
        placeholder="filter metrics… (e.g. loss, rollout/)"
        bind:value={searchInput}
        aria-label="Filter metrics by name"
      />
      <span class="text-(--muted) text-[11px] uppercase tracking-[0.04em] [font-variant-numeric:tabular-nums]">
        {#if search}{visibleKeyCount} of {allKeys.length}{:else}{allKeys.length}{/if}
        {allKeys.length === 1 ? "metric" : "metrics"}
        {#if stepRange}· steps {stepRange[0]}–{stepRange[1]}{/if}
      </span>
      {#if stepDomain}
        <button class="log-button" type="button" onclick={() => onDomainChange(null)}>
          Reset zoom
        </button>
      {/if}
      <button
        class="log-button"
        type="button"
        onclick={() => setAllCollapsed(collapsed.size < groups.length)}
        disabled={!groups.length}
      >
        {collapsed.size < groups.length ? "Collapse all" : "Expand all"}
      </button>
      {#if isRunning}
        <span class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em] shrink-0">
          <span class="dot bg-[#4ade80]! shadow-[0_0_0_2px_rgba(74,222,128,0.18)]"></span> live
        </span>
      {:else if payload?.stale}
        <span
          class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em] shrink-0"
          title="Newer points are still buffered by the dashboard and will appear shortly."
        >
          <span class="dot dot-dim"></span> syncing
        </span>
      {/if}
    </div>

    {#if error}
      <div class="text-[#fbbf24] text-[12px]" role="status">
        Metrics may be out of date: {error}
      </div>
    {/if}

    {#if !groups.length}
      <div class="detail-empty">No metrics match “{search}”.</div>
    {/if}

    {#each panels as group (group.name)}
      <GroupSection
        title={group.name}
        subtitle={subtitleFor(group)}
        expanded={!collapsed.has(group.name)}
        onToggle={() => toggleGroup(group.name)}
      >
        <div class="p-[0_14px_14px]">
          <div class="chart-grid" style:margin-bottom="0">
            {#each group.shown as key (key)}
              {@const rows = rowsByKey.get(key) ?? []}
              {@const stats = seriesStats(rows)}
              <div class="rollout-chart">
                <LineChart
                  title={key}
                  data={rows}
                  height={140}
                  label="value"
                  smoothable
                  ariaLabel={`${key} over training steps`}
                  formatX={formatStep}
                  formatY={formatMetricValue}
                  xDomain={stepDomain}
                  onChangeDomainX={onDomainChange}
                />
                {#if stats}
                  <div class="flex flex-wrap gap-[12px] mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
                    <span>latest <span class="text-(--text)">{formatMetricValue(stats.latest)}</span></span>
                    <span>min <span class="text-(--text)">{formatMetricValue(stats.min)}</span></span>
                    <span>max <span class="text-(--text)">{formatMetricValue(stats.max)}</span></span>
                    <span>{stats.count} {stats.count === 1 ? "point" : "points"}</span>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
          {#if group.truncated > 0 && group.shown.length < group.keys.length}
            <div class="text-(--muted) text-[12px] mt-[10px]">
              {group.truncated} more in this group — narrow the filter to see them.
            </div>
          {/if}
        </div>
      </GroupSection>
    {/each}

    {#if hiddenByCap > 0}
      <div class="text-(--muted) text-[12px]" role="status">
        Showing {MAX_CHARTS} of {visibleKeyCount} charts. Filter by name or collapse groups to see the rest.
      </div>
    {/if}
  {/if}
</div>
