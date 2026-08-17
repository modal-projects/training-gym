<script>
  import { onMount } from "svelte";
  import {
    AdvantageSpreadChart,
    AdvantageViolins,
    ChartSkeleton,
    ComparativeBarChart,
    LineChart,
    RunSummary,
    StepTimings,
  } from "$host/components";
  import { fetchRun, fetchRunAdvantages, fetchRunRollouts, fetchRollout } from "$host/data";

  let {
    runId,
    initialRun = null,
    getStatus = (run) => run?.status || "pending",
    showFrameworkStatus = () => false,
    modelName = (run) => run?.model || "—",
    fmtDuration = () => "—",
  } = $props();

  let run = $state(null);
  let rollouts = $state([]);
  let advantageSteps = $state([]);
  let scoreDist = $state(null);
  let loading = $state(true);
  let error = $state("");

  $effect(() => {
    if (initialRun?.run_id === runId) run = initialRun;
  });

  onMount(() => {
    let disposed = false;
    async function refresh() {
      try {
        const [nextRun, nextRollouts, nextAdvantages] = await Promise.all([
          fetchRun(runId),
          fetchRunRollouts(runId),
          fetchRunAdvantages(runId),
        ]);
        if (disposed) return;
        run = nextRun;
        rollouts = nextRollouts;
        advantageSteps = nextAdvantages;
        if (nextRollouts.length) {
          const firstId = Math.min(...nextRollouts.map((row) => Number(row.rollout_id)));
          const lastId = Math.max(...nextRollouts.map((row) => Number(row.rollout_id)));
          const first = await fetchRollout(runId, firstId);
          const last = firstId === lastId ? first : await fetchRollout(runId, lastId);
          scoreDist = buildDist(
            (first?.samples || []).map((sample) => Number(sample.score ?? sample.reward ?? 0)),
            (last?.samples || []).map((sample) => Number(sample.score ?? sample.reward ?? 0)),
            firstId,
            lastId,
          );
        } else {
          scoreDist = null;
        }
        error = "";
      } catch (reason) {
        if (!disposed) error = reason?.message || String(reason);
      } finally {
        if (!disposed) loading = false;
      }
    }
    void refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  });

  let rewardData = $derived(
    rollouts.map((row) => ({
      x: Number(row.rollout_id) || 0,
      y: Number(row.mean) || 0,
      rollout_id: row.rollout_id,
    })),
  );
  let customTags = $derived(
    Array.from(new Set(rollouts.flatMap((row) => Object.keys(row.tag_stats || {})))).sort(),
  );

  let chartStats = $derived.by(() => {
    const values = rollouts.map((row) => Number(row.mean) || 0);
    return values.length
      ? { min: Math.min(...values), latest: values.at(-1), max: Math.max(...values) }
      : null;
  });

  function formatMean(value) {
    return Number(value || 0).toFixed(3);
  }

  function buildDist(firstValues, lastValues, firstId, lastId) {
    const values = [...firstValues, ...lastValues];
    if (!values.length) return null;
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const count = lo === hi ? 1 : 12;
    const span = hi - lo || 1;
    const bins = Array.from({ length: count }, (_, index) => ({
      lo: lo + (index / count) * span,
      hi: lo + ((index + 1) / count) * span,
      first: 0,
      last: 0,
    }));
    const indexFor = (value) =>
      Math.max(0, Math.min(count - 1, Math.floor(((value - lo) / span) * count)));
    for (const value of firstValues) bins[indexFor(value)].first += 1;
    for (const value of lastValues) bins[indexFor(value)].last += 1;
    return { bins, lo, hi, firstId, lastId };
  }

  function distCategories(dist) {
    return dist.bins.map((bin) => `${formatMean(bin.lo)}–${formatMean(bin.hi)}`);
  }

  function distSeries(dist) {
    const series = [
      {
        name: `rollout ${dist.firstId}`,
        color: "var(--color-c-gray-40)",
        values: dist.bins.map((bin) => bin.first),
      },
    ];
    if (dist.firstId !== dist.lastId) {
      series.push({
        name: `latest (rollout ${dist.lastId})`,
        color: "var(--accent)",
        values: dist.bins.map((bin) => bin.last),
      });
    }
    return series;
  }

  let tagChartData = (tag) =>
    rollouts
      .filter((row) => row.tag_stats?.[tag])
      .map((row) => ({ x: row.rollout_id, y: row.tag_stats[tag].mean }));

  let tagChartStats = (tag) => {
    const values = rollouts
      .filter((row) => row.tag_stats?.[tag])
      .map((row) => Number(row.tag_stats[tag].mean) || 0);
    return values.length
      ? { min: Math.min(...values), latest: values.at(-1), max: Math.max(...values) }
      : null;
  };
</script>

{#if error}
  <div class="detail-empty">Failed to load summary: {error}</div>
{:else if !run}
  <div class="detail-empty">Loading run {runId}…</div>
{:else}
  <div class="summary-tab">
      <div class="summary-tab-main">
        {#if run.error_message}
          <div class="mb-[20px]">
            <div class="text-(--red,#f87171) text-[12px] font-[600] tracking-[0.02em] mb-[6px] uppercase">Error</div>
            <pre class="[border:1px_solid_color-mix(in_srgb,var(--red,#f87171)_45%,transparent)] rounded-[8px] bg-[color-mix(in_srgb,var(--red,#f87171)_12%,transparent)] text-(--red,#f87171) [font-family:var(--font-mono)] text-[12px] leading-[17px] m-0 max-h-[320px] overflow-auto p-[12px_14px] whitespace-pre-wrap [word-break:break-word]">{run.error_message}</pre>
          </div>
        {/if}
        {#if run.step_times || run.substep_times}
          <div class="rollout-chart">
            <div class="rollout-chart-title">Step &amp; substep timeline</div>
            <div class="chart-scroll">
              <StepTimings stepTimes={run.step_times} substepTimes={run.substep_times} layout="timeline" downloadName={`step_substep_times_${runId}.json`} />
            </div>
          </div>
        {/if}
        {#if loading && !rollouts.length}
          <div class="rollout-chart"><ChartSkeleton variant="line" height={140} showTitle /></div>
        {:else if rollouts.length}
          <div class="rollout-chart">
            <div class="chart-scroll">
              <LineChart
                title="Reward"
                data={rewardData}
                formatX={(row) => `rollout ${row.rollout_id}`}
                formatY={formatMean}
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
          <div class="rollout-chart">
            <div class="rollout-chart-title">Score distribution</div>
            <div class="chart-scroll">
              {#if scoreDist}
                <ComparativeBarChart
                  categories={distCategories(scoreDist)}
                  series={distSeries(scoreDist)}
                  height={120}
                  showCategoryLabels={false}
                  format={(value) => `${value}`}
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
          {#if advantageSteps.length}
            <div class="chart-grid">
              <div class="rollout-chart">
                <div class="rollout-chart-title">Advantage spread over time</div>
                <AdvantageSpreadChart steps={advantageSteps} />
              </div>
              <div class="rollout-chart">
                <div class="rollout-chart-title">Advantage distribution over time</div>
                <AdvantageViolins steps={advantageSteps} />
              </div>
            </div>
          {/if}
          {#each customTags as tag (tag)}
            <div class="rollout-chart">
              <LineChart
                title={`${tag} (mean)`}
                data={tagChartData(tag)}
                formatX={(row) => `rollout ${row.x}`}
                formatY={formatMean}
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
        {:else}
          <div class="detail-empty">No rollouts recorded yet.</div>
        {/if}
      </div>
      <aside class="summary-tab-side">
        <RunSummary {run} {getStatus} {showFrameworkStatus} {modelName} {fmtDuration} />
      </aside>
  </div>
{/if}
