<script>
  import { onMount } from "svelte";
  import { Download, X } from "lucide-svelte";
  import ConversationView from "../../components/ConversationView.svelte";
  import SampleTimeline from "../../components/SampleTimeline.svelte";
  import ResizableTable from "../../components/ResizableTable.svelte";
  import { fetchRunRollouts, fetchRollout } from "../../lib/api.js";

  let { runId } = $props();
  let rows = $state([]);
  let selected = $state(null);
  let loading = $state(true);
  let error = $state("");

  onMount(() => {
    let disposed = false;
    async function refresh() {
      try {
        const next = await fetchRunRollouts(runId);
        if (!disposed) {
          rows = next;
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
    const detail = await fetchRollout(runId, row.rollout_id);
    selected = { row, detail, loading: false };
  }

  let samples = $derived(selected?.detail?.samples || []);
  function close() {
    selected = null;
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
    <ResizableTable columns={[
      { key: "step", label: "Step", width: 72, minWidth: 56 },
      { key: "mean", label: "Mean reward", width: 118, minWidth: 96 },
      { key: "rollouts", label: "Rollouts", width: 80, minWidth: 64 },
    ]}>
      <tbody>
        {#each rows as row (row.rollout_id)}
          <tr class="cursor-pointer" onclick={() => openRollout(row)} role="button" tabindex="0" onkeydown={(event) => event.key === "Enter" && openRollout(row)}>
            <td>{row.rollout_id}</td>
            <td>{Number(row.mean || 0).toFixed(3)}</td>
            <td>{row.total || 0}</td>
          </tr>
        {/each}
      </tbody>
    </ResizableTable>
  {/if}

  {#if selected}
    <div class="sample-viewer">
      <div class="sample-viewer-header">
        <span>Rollout {selected.row.rollout_id}</span>
        <div class="sample-viewer-actions">
          <button class="sample-nav-btn" onclick={download} aria-label="Download trajectory"><Download size={14} /></button>
          <button class="sample-nav-btn" onclick={close} aria-label="Close rollout viewer"><X size={14} /></button>
        </div>
      </div>
      {#if selected.loading}
        <div class="detail-empty">Loading rollout…</div>
      {:else if samples.length}
        {@const sample = samples[0]}
        {#if sample.prompt}<div class="rollout-sample-label">prompt</div><pre class="rollout-sample-text">{sample.prompt}</pre>{/if}
        <div class="rollout-sample-label">conversation</div>
        <ConversationView messages={sample.metadata?.trajectory_messages} response={sample.response || ""} thinking={sample.thinking || ""} evalReport={sample.metadata?.eval_report} />
        {#if sample.trace?.length}<div class="rollout-sample-label">trajectory timeline</div><div class="chart-scroll"><SampleTimeline trace={sample.trace} /></div>{/if}
      {:else}
        <div class="detail-empty">No samples recorded for this rollout.</div>
      {/if}
    </div>
  {/if}
</div>
