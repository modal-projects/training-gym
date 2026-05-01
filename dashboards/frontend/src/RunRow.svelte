<script>
  import { truncateId, fmtCluster, fmtLr } from "./lib/format.js";
  import MetricsChart from "./MetricsChart.svelte";

  let { run } = $props();
  let showMetrics = $state(false);

  let summary = $derived(run.config_summary || {});
  let result = $derived(run.train_result);
  let hasWandb = $derived(!!result?.wandb_url);
</script>

<tr>
  <td class="run-id" title={run.run_id}>{truncateId(run.run_id)}</td>
  <td>
    <div class="model-name">{summary.model_name || "—"}</div>
    {#if run.modal_app_id}
      <div class="app-id">{run.modal_app_id}</div>
    {/if}
  </td>
  <td class="cluster">{fmtCluster(summary)}</td>
  <td class="config-details">
    {#if summary.lr}
      <span class="config-tag">lr {fmtLr(summary.lr)}</span>
    {/if}
    {#if summary.global_batch_size}
      <span class="config-tag">bs {summary.global_batch_size}</span>
    {/if}
    {#if summary.wandb_group}
      <span class="config-tag">{summary.wandb_group}</span>
    {/if}
  </td>
  <td>
    {#if result}
      <span class="result-badge result-completed">Completed</span>
    {:else}
      <span class="result-badge result-pending">Pending</span>
    {/if}
  </td>
  <td>
    {#if result}
      {#if result.checkpoint_dir}
        <span class="tag" title={result.checkpoint_dir}>
          <strong>ckpt</strong>
        </span>
      {/if}
      {#if hasWandb}
        <a class="wandb-link" href={result.wandb_url} target="_blank" rel="noopener">W&B</a>
        <button class="metrics-toggle" onclick={() => showMetrics = !showMetrics}>
          {showMetrics ? "Hide" : "Metrics"}
        </button>
      {/if}
    {/if}
  </td>
</tr>
{#if showMetrics && result?.training_run_id}
  <tr class="metrics-row">
    <td colspan="6">
      <MetricsChart trainingRunId={result.training_run_id} />
    </td>
  </tr>
{/if}

<style>
  td {
    padding: 0.55rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  tr:last-child td {
    border-bottom: none;
  }
  .run-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85em;
    color: var(--accent);
    cursor: default;
  }
  .model-name {
    font-weight: 500;
  }
  .app-id {
    color: var(--muted);
    font-size: 0.8em;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    margin-top: 0.1rem;
  }
  .cluster {
    font-size: 0.9em;
    white-space: nowrap;
  }
  .config-details {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }
  .config-tag {
    display: inline-block;
    padding: 0.1rem 0.4rem;
    background: var(--surface);
    border-radius: 4px;
    font-size: 0.8em;
    color: var(--muted);
    white-space: nowrap;
  }
  .result-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.8em;
    font-weight: 500;
  }
  .result-completed {
    background: rgba(74, 222, 128, 0.12);
    color: var(--green);
  }
  .result-pending {
    background: rgba(251, 191, 36, 0.12);
    color: var(--yellow);
  }
  .tag {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    margin: 0.1rem 0.2rem 0.1rem 0;
    background: var(--surface);
    border-radius: 4px;
    font-size: 0.8em;
    color: var(--muted);
    cursor: default;
  }
  .tag strong {
    color: var(--text);
    font-weight: 500;
  }
  .wandb-link {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin-right: 0.4rem;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.1);
    text-decoration: none;
  }
  .wandb-link:hover {
    background: rgba(251, 191, 36, 0.2);
  }
  .metrics-toggle {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin-right: 0.4rem;
    border: none;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
    color: var(--accent);
    background: var(--accent-dim);
    cursor: pointer;
  }
  .metrics-toggle:hover {
    background: rgba(125, 211, 252, 0.2);
  }
  .metrics-row td {
    padding: 0.25rem 1rem 0.75rem;
    background: var(--panel-alt);
  }
</style>
