<script>
  import { onDestroy } from "svelte";
  import { truncateId, fmtCluster, fmtLr } from "./lib/format.js";

  let { run, deployments = [] } = $props();

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function normalizePath(value) {
    return safeText(value).replace(/\/+$/, "");
  }

  function pathMatches(left, right) {
    const a = normalizePath(left);
    const b = normalizePath(right);
    if (!a || !b) return false;
    return a === b || a.startsWith(`${b}/`) || b.startsWith(`${a}/`);
  }

  let summary = $derived(run.config_summary || {});
  let result = $derived(run.train_result);
  let trainingRunId = $derived(result?.training_run_id || run.run_id || "");
  let modalAppUrl = $derived(run.modal_app_url || null);
  let copiedTrainingRunId = $state(false);
  let copyResetTimer = null;
  let deployment = $derived(
    deployments.find((d) => {
      const deploymentAppName = safeText(d.app_name || "");
      const deploymentModelName = safeText(d.model_name || "");
      const deploymentModelPath = normalizePath(d.model_path || "");
      const deploymentCheckpointPath = normalizePath(d.checkpoint_path || "");
      const runModelName = safeText(result?.model_name || summary.model_name || "");
      const runModelPath = normalizePath(result?.model_path || "");
      const runCheckpointDir = normalizePath(result?.checkpoint_dir || "");

      if (run.deployment_id && deploymentAppName && run.deployment_id === deploymentAppName) {
        return true;
      }
      if (
        deploymentCheckpointPath &&
        (pathMatches(deploymentCheckpointPath, runCheckpointDir) ||
          pathMatches(deploymentCheckpointPath, runModelPath))
      ) {
        return true;
      }
      if (
        deploymentModelPath &&
        (pathMatches(deploymentModelPath, runCheckpointDir) ||
          pathMatches(deploymentModelPath, runModelPath))
      ) {
        return true;
      }
      return !!deploymentModelName && deploymentModelName === runModelName;
    }) || null,
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

  async function copyTrainingRunId(event) {
    event.stopPropagation();
    if (!trainingRunId) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(trainingRunId);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = trainingRunId;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      copiedTrainingRunId = true;
      if (copyResetTimer) {
        clearTimeout(copyResetTimer);
      }
      copyResetTimer = setTimeout(() => {
        copiedTrainingRunId = false;
      }, 1200);
    } catch {
      copiedTrainingRunId = false;
    }
  }

  onDestroy(() => {
    if (copyResetTimer) clearTimeout(copyResetTimer);
  });
</script>

<tr
  class:clickable={!!modalAppUrl}
  onclick={openModalApp}
  onkeydown={onRowKeydown}
  role={modalAppUrl ? "link" : undefined}
  tabindex={modalAppUrl ? "0" : undefined}
  aria-label={modalAppUrl ? `Open Modal app for training run ${run.run_id}` : undefined}
>
  <td class="run-id" title={run.run_id}>
    <div class="mono">{truncateId(run.run_id)}</div>
    {#if run.modal_app_id}
      <div class="app-id" title={run.modal_app_id}>{truncateId(run.modal_app_id)}</div>
    {/if}
  </td>
  <td class="training-run-id-cell">
    {#if trainingRunId}
      <button
        type="button"
        class="copy-chip mono"
        title={`Click to copy ${trainingRunId}`}
        aria-label={`Copy training run id ${trainingRunId}`}
        onclick={copyTrainingRunId}
      >
        <span>{truncateId(trainingRunId)}</span>
        <span class="copy-chip-state">{copiedTrainingRunId ? "Copied" : "Copy"}</span>
      </button>
    {:else}
      —
    {/if}
  </td>
  <td>
    <div class="model-name">{summary.model_name || "—"}</div>
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
  <td class="result-cell">
    {#if result}
      <span class="result-badge result-completed">Completed</span>
      <div class="result-meta mono" title={result.training_run_id}>
        TrainResult {truncateId(result.training_run_id)}
      </div>
      {#if result.checkpoint_dir}
        <div class="result-meta mono" title={result.checkpoint_dir}>
          {truncateId(result.checkpoint_dir)}
        </div>
      {/if}
    {:else if run.status === "stopped" || run.status === "failed"}
      <span class="result-badge result-stopped">No result</span>
    {:else}
      <span class="result-badge result-pending">Pending</span>
    {/if}
  </td>
  <td class="deployment-cell">
    {#if deployment}
      <div class="deployment-name">
        {deployment.app_name || deployment.served_model_name || deployment.model_name || "Deployment"}
      </div>
      {#if deployment.url}
        <a
          class="pill-link pill-deploy"
          href={deployment.url}
          target="_blank"
          rel="noopener noreferrer"
          onclick={(event) => event.stopPropagation()}>Endpoint</a
        >
      {/if}
    {:else}
      <span class="result-badge result-pending">Not deployed</span>
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
  .mono {
    font-family: var(--font-mono);
  }
  .training-run-id-cell {
    white-space: nowrap;
  }
  .copy-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.16rem 0.45rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel-alt);
    color: var(--text);
    font-size: 0.66rem;
    cursor: copy;
  }
  .copy-chip:hover {
    border-color: var(--accent-border);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
  }
  .copy-chip-state {
    color: var(--muted-strong);
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
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
  .result-cell,
  .deployment-cell {
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
  .result-stopped {
    border-color: color-mix(in srgb, #f87171 45%, transparent);
    background: color-mix(in srgb, #f87171 10%, transparent);
    color: #f87171;
  }
  .result-meta {
    margin-top: 0.18rem;
    font-size: 0.7rem;
    color: var(--muted-strong);
  }
  .deployment-name {
    font-weight: 500;
    color: var(--text-bright);
    margin-bottom: 0.15rem;
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
