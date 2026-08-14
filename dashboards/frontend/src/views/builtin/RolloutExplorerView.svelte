<script>
  import { onMount } from "svelte";
  import { ChevronLeft, ChevronRight, Download, X } from "$host/icons";
  import {
    ConversationView,
    ResizableTable,
    SampleTimeline,
    StepTimings,
    TimeAgo,
  } from "$host/components";
  import { fetchRunRollouts, fetchRollout } from "$host/data";

  let { runId } = $props();
  let rows = $state([]);
  let selected = $state(null);
  let loading = $state(true);
  let error = $state("");
  let activeSampleIndex = $state(0);

  onMount(() => {
    let disposed = false;
    async function refresh() {
      try {
        const next = await fetchRunRollouts(runId);
        if (!disposed) {
          rows = next;
          if (!selected && next.length) {
            void openRollout(next[0]);
          }
          error = "";
        }
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

  async function openRollout(row) {
    selected = { loading: true, row };
    activeSampleIndex = 0;
    const detail = await fetchRollout(runId, row.rollout_id);
    selected = { row, detail, loading: false };
  }

  let samples = $derived(selected?.detail?.samples || []);
  function close() {
    selected = null;
    activeSampleIndex = 0;
  }

  let sampleDist = $derived.by(() => {
    const samples = selected?.detail?.samples || [];
    if (!samples.length) return null;
    const values = samples.map((sample) => Number(sample.score ?? sample.reward ?? 0));
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const count = lo === hi ? 1 : 12;
    const span = hi - lo || 1;
    const buckets = Array.from({ length: count }, () => []);
    for (const [index, score] of values.entries()) {
      const bucket = Math.max(0, Math.min(count - 1, Math.floor(((score - lo) / span) * count)));
      buckets[bucket].push({ index, score });
    }
    return { lo, hi, span, count, buckets, maxCount: Math.max(...buckets.map((bucket) => bucket.length), 1) };
  });

  function downloadAll() {
    if (!selected?.detail) return;
    const blob = new Blob([JSON.stringify(selected.detail, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `rollout_${selected.row.rollout_id}_samples.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  function download() {
    if (!selected?.detail) return;
    const blob = new Blob([JSON.stringify(selected.detail, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `rollout_${selected.row.rollout_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="tab-panel">
  {#if error}
    <div class="detail-empty">Failed to load rollouts: {error}</div>
  {:else if loading && !rows.length}
    <div class="detail-empty">Loading rollouts…</div>
  {:else if !rows.length}
    <div class="detail-empty">No rollouts recorded yet.</div>
  {:else}
    <ResizableTable class="rollout-table" columns={[
      { key: "step", label: "Step", width: 72, minWidth: 56 },
      { key: "mean", label: "Mean reward", width: 118, minWidth: 96 },
      { key: "rollouts", label: "Rollouts", width: 80, minWidth: 64 },
      { key: "when", label: "When", width: 88, minWidth: 64 },
    ]}>
      <tbody>
        {#each rows as row (row.rollout_id)}
          <tr class:expanded={selected?.row?.rollout_id === row.rollout_id} onclick={() => openRollout(row)} role="button" tabindex="0" onkeydown={(event) => event.key === "Enter" && openRollout(row)}>
            <td>#{row.rollout_id}</td>
            <td>{Number(row.mean || 0).toFixed(3)}</td>
            <td>—</td>
            <td><TimeAgo timestamp={row.created_at} showJustNow falsyRepresentation="—" /></td>
          </tr>
          {#if selected?.row?.rollout_id === row.rollout_id}
            <tr>
              <td class="p-[12px_10px] bg-(--color-c-gray-08,#1c1c1c) cursor-default" colspan={4}>
                {#if selected.loading}
                  <div class="detail-empty">Loading rollouts…</div>
                {:else if samples.length}
                  {@const sample = samples[activeSampleIndex]}
                  <div class="rollout-chart">
                    <div class="rollout-chart-title">Step timing</div>
                    <div class="chart-scroll">
                      <StepTimings
                        stepTimes={{ 1: { duration_s: selected.row.rollout_time } }}
                        substepTimes={{}}
                        layout="rows"
                      />
                    </div>
                  </div>
                  {#if sampleDist}
                    <div class="mb-[16px]">
                      <div class="flex justify-end mb-[6px]">
                        <button
                          type="button"
                          class="inline-flex items-center gap-[5px] [background:none] [border:1px_solid_var(--border,#2f2f2f)] rounded-[4px] text-(--muted) text-[11px] p-[3px_8px] cursor-pointer hover:text-(--text) hover:border-(--border-strong,#4a4a4a)"
                          onclick={downloadAll}
                          title="Download all samples as JSON"
                        >
                          <Download size={13} />
                          Download all ({samples.length} samples)
                        </button>
                      </div>
                      <div class="chart-scroll">
                        <div
                          class="flex items-end gap-[2px] h-[120px] pt-[14px] min-w-[280px] [border-bottom:1px_solid_var(--border,#2f2f2f)]"
                          role="group"
                          aria-label="Reward distribution"
                        >
                          {#each sampleDist.buckets as bucket, index (index)}
                            <button
                              type="button"
                              class="dist-bar"
                              class:detail-active={bucket.some((entry) => entry.index === activeSampleIndex)}
                              class:is-empty={!bucket.length}
                              style:height={`${(bucket.length / sampleDist.maxCount) * 100}%`}
                              disabled={!bucket.length}
                              onclick={() => (activeSampleIndex = bucket[0].index)}
                              aria-label={`Reward bucket ${index + 1}`}
                            >
                              <span class="absolute top-[-14px] left-0 right-0 text-center text-[10px] text-(--muted) [font-variant-numeric:tabular-nums]">{bucket.length || ""}</span>
                            </button>
                          {/each}
                        </div>
                        <div class="dist-axis">
                          <span>{Number(sampleDist.lo).toFixed(3)}</span>
                          <span class="dist-axis-label">reward · {samples.length} samples</span>
                          <span>{Number(sampleDist.hi).toFixed(3)}</span>
                        </div>
                      </div>
                    </div>
                  {/if}
                  <div class="sample-viewer">
                    <div class="sample-viewer-header">
                      <div class="sample-viewer-nav">
                        <button
                          class="sample-nav-btn"
                          onclick={() => (activeSampleIndex = Math.max(0, activeSampleIndex - 1))}
                          disabled={activeSampleIndex === 0}
                          aria-label="Previous rollout"
                        >
                          <ChevronLeft size={14} />
                        </button>
                        <span>Rollout {activeSampleIndex + 1} / {samples.length}</span>
                        <button
                          class="sample-nav-btn"
                          onclick={() => (activeSampleIndex = Math.min(samples.length - 1, activeSampleIndex + 1))}
                          disabled={activeSampleIndex === samples.length - 1}
                          aria-label="Next rollout"
                        >
                          <ChevronRight size={14} />
                        </button>
                        <span class="sample-viewer-hint">← / → to navigate</span>
                      </div>
                      <div class="sample-viewer-actions">
                        <span class="rollout-sample-metric">reward {Number(sample.score || 0).toFixed(3)}</span>
                        <button class="sample-nav-btn" onclick={download} aria-label="Download trajectory"><Download size={14} /></button>
                        <button class="sample-nav-btn" onclick={close} aria-label="Close rollout viewer"><X size={14} /></button>
                      </div>
                    </div>
                    {#if sample.prompt}<div class="rollout-sample-label">prompt</div><pre class="rollout-sample-text">{sample.prompt}</pre>{/if}
                    <div class="rollout-sample-label">conversation</div>
                    <ConversationView messages={sample.metadata?.trajectory_messages} response={sample.response || ""} thinking={sample.thinking || ""} evalReport={sample.metadata?.eval_report} />
                    {#each Object.entries(sample.metadata || {}).filter(([name]) => ["accuracy", "custom_score", "exit_status"].includes(name)) as [name, value]}
                      <div class="rollout-sample-label">{name}</div>
                      <span class="rollout-sample-metric">{String(value)}</span>
                    {/each}
                    {#if sample.trace?.length}<div class="rollout-sample-label">trajectory timeline</div><div class="chart-scroll"><SampleTimeline trace={sample.trace} /></div>{/if}
                  </div>
                {:else}
                  <div class="detail-empty">No samples recorded for this rollout.</div>
                {/if}
              </td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </ResizableTable>
  {/if}
</div>
