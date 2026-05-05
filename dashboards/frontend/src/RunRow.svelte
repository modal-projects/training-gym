<script>
  import { truncateId, fmtCluster, fmtLr } from "./lib/format.js";

  let { run, deployments = [] } = $props();

  let summary = $derived(run.config_summary || {});
  let result = $derived(run.train_result);
  let modalAppUrl = $derived(run.modal_app_url || null);
  let deployment = $derived(
    deployments.find((d) => d.app_name && result?.app_name && d.app_name === result.app_name) || null,
  );

  function openModalApp() {
    if (!modalAppUrl) return;
    window.open(modalAppUrl, "_blank", "noopener,noreferrer");
  }

  function onRowKeydown(event) {
    if (!modalAppUrl) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openModalApp();
    }
  }
</script>

<tr
  class:clickable={!!modalAppUrl}
  onclick={openModalApp}
  onkeydown={onRowKeydown}
  role={modalAppUrl ? "link" : undefined}
  tabindex={modalAppUrl ? "0" : undefined}
  aria-label={modalAppUrl ? `Open Modal app for run ${run.run_id}` : undefined}
>
  <td class="run-id" title={run.run_id}>{truncateId(run.run_id)}</td>
  <td>
    <div class="model-name">{summary.model_name || "—"}</div>
    {#if run.modal_app_id}
      {#if modalAppUrl}
        <a
          class="app-id app-link"
          href={modalAppUrl}
          target="_blank"
          rel="noopener noreferrer"
          onclick={(event) => event.stopPropagation()}
          >{run.modal_app_id}</a
        >
      {:else}
        <div class="app-id">{run.modal_app_id}</div>
      {/if}
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
  <td class="status-cell">
    {#if result}
      <span class="result-badge result-completed">Completed</span>
    {:else}
      <span class="result-badge result-pending">Pending</span>
    {/if}
  </td>
  <td>
    {#if modalAppUrl}
      <a
        class="pill-link pill-modal"
        href={modalAppUrl}
        target="_blank"
        rel="noopener noreferrer"
        onclick={(event) => event.stopPropagation()}>Modal</a
      >
    {/if}
    {#if result}
      {#if result.wandb_url}
        <a
          class="pill-link pill-wandb"
          href={result.wandb_url}
          target="_blank"
          rel="noopener noreferrer"
          onclick={(event) => event.stopPropagation()}>W&B</a
        >
      {/if}
      {#if deployment?.url}
        <a
          class="pill-link pill-deploy"
          href={deployment.url}
          target="_blank"
          rel="noopener noreferrer"
          onclick={(event) => event.stopPropagation()}>Endpoint</a
        >
      {/if}
      {#if result.checkpoint_dir}
        <span class="tag" title={result.checkpoint_dir}>
          <strong>ckpt</strong>
        </span>
      {/if}
    {/if}
  </td>
</tr>

<style>
  td {
    padding: 0.5rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    font-size: 0.79rem;
  }
  tr:hover td {
    background: color-mix(in srgb, var(--text) 4%, transparent);
  }
  .clickable td {
    cursor: pointer;
  }
  .clickable:focus-visible td {
    outline: 1px solid var(--accent-border);
    outline-offset: -1px;
    background: color-mix(in srgb, var(--accent) 8%, transparent);
  }
  tr:last-child td {
    border-bottom: none;
  }
  .run-id {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: color-mix(in srgb, var(--accent) 78%, white);
    cursor: default;
  }
  .model-name {
    font-weight: 500;
    color: var(--text-bright);
  }
  .app-id {
    color: var(--muted-strong);
    font-size: 0.72rem;
    font-family: var(--font-mono);
    margin-top: 0.1rem;
  }
  .app-link {
    text-decoration: none;
  }
  .app-link:hover {
    color: var(--accent);
  }
  .cluster {
    font-size: 0.75rem;
    white-space: nowrap;
    color: var(--text);
  }
  .config-details {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }
  .config-tag {
    display: inline-block;
    padding: 0.12rem 0.4rem;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 9999px;
    font-size: 0.67rem;
    color: var(--muted);
    white-space: nowrap;
  }
  .status-cell {
    white-space: nowrap;
  }
  .result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.12rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.71rem;
    font-weight: 500;
    border: 1px solid transparent;
  }
  .result-badge::before {
    content: "";
    width: 0.38rem;
    height: 0.38rem;
    border-radius: 9999px;
    background: currentColor;
  }
  .result-completed {
    border-color: var(--accent-border);
    background: var(--accent-soft);
    color: var(--green);
  }
  .result-pending {
    border-color: color-mix(in srgb, var(--yellow) 45%, transparent);
    background: color-mix(in srgb, var(--yellow) 10%, transparent);
    color: var(--yellow);
  }
  .tag {
    display: inline-block;
    padding: 0.12rem 0.45rem;
    margin: 0 0.2rem 0.15rem 0;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.68rem;
    color: var(--muted);
    cursor: default;
  }
  .tag strong {
    color: var(--text-bright);
    font-weight: 500;
  }
  .pill-link {
    display: inline-block;
    padding: 0.14rem 0.48rem;
    margin-right: 0.3rem;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 500;
    text-decoration: none;
  }
  .pill-modal {
    color: var(--accent);
    border: 1px solid var(--accent-border);
    background: var(--accent-soft);
  }
  .pill-modal:hover {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
  }
  .pill-wandb {
    color: var(--yellow);
    border: 1px solid color-mix(in srgb, var(--yellow) 45%, transparent);
    background: color-mix(in srgb, var(--yellow) 10%, transparent);
  }
  .pill-wandb:hover {
    background: color-mix(in srgb, var(--yellow) 18%, transparent);
  }
  .pill-deploy {
    color: var(--green);
    border: 1px solid var(--accent-border);
    background: var(--accent-soft);
  }
  .pill-deploy:hover {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
  }
</style>
