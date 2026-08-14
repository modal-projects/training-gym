<script>
  import { onMount } from "svelte";
  import { ChartSkeleton, LineChart, RunSummary, StepTimings } from "$host/components";
  import { fetchRun, fetchRunRollouts } from "$host/data";

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
  let loading = $state(true);
  let error = $state("");

  $effect(() => {
    if (initialRun?.run_id === runId) run = initialRun;
  });

  onMount(() => {
    let disposed = false;
    async function refresh() {
      try {
        const [nextRun, nextRollouts] = await Promise.all([
          fetchRun(runId),
          fetchRunRollouts(runId),
        ]);
        if (disposed) return;
        run = nextRun;
        rollouts = nextRollouts;
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
            <div class="rollout-chart-title">Reward over time</div>
            <LineChart title="Reward over time" data={rewardData} height={180} formatY={(value) => Number(value).toFixed(3)} />
          </div>
          {#each customTags as tag (tag)}
            <div class="rollout-chart">
              <div class="rollout-chart-title">{tag}</div>
              <LineChart
                title={tag}
                data={rollouts.filter((row) => row.tag_stats?.[tag]).map((row) => ({ x: row.rollout_id, y: row.tag_stats[tag].mean }))}
                height={150}
              />
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
