<script>
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";

  let {
    allDeployments,
    linkedDeployments,
    distinctDeploymentModels,
    loading,
    error,
    deploymentRows,
    deploymentLabel,
    truncateId,
    onOpenTrainingRun,
  } = $props();
</script>

<section class="summary-row">
  <article class="summary-card">
    <span class="summary-label">Total deployments</span>
    <strong>{allDeployments.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Linked runs</span>
    <strong>{linkedDeployments}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Distinct models</span>
    <strong>{distinctDeploymentModels}</strong>
  </article>
</section>

<section class="runs-surface">
  <div class="runs-body">
    {#if loading}
      <div class="table-wrap">
        <MinimalTableSkeleton
          class="runs-table"
          columns={["Deployment", "Training run", "Model", "Checkpoint", "Endpoint", "Links"]}
          rows={6}
        />
      </div>
    {:else if error}
      <div class="empty">Failed to load: {error}</div>
    {:else if !allDeployments.length}
      <div class="empty">No deployments recorded yet.</div>
    {:else}
      <div class="table-wrap">
        <MinimalTable class="runs-table">
          <thead>
            <tr>
              <th>Deployment</th>
              <th>Training run</th>
              <th>Model</th>
              <th>Checkpoint</th>
              <th>Endpoint</th>
              <th>Links</th>
            </tr>
          </thead>
          <tbody>
            {#each deploymentRows as row (row.deployment.url || row.deployment.app_name || row.deployment.model_name)}
              <tr>
                <td>
                  <div class="run-name">{deploymentLabel(row.deployment)}</div>
                  {#if row.deployment.app_name}
                    <div class="mono">{row.deployment.app_name}</div>
                  {/if}
                </td>
                <td class="mono">
                  {#if row.run}
                    <button class="cross-link" onclick={() => onOpenTrainingRun(row.run.run_id)}>
                      {truncateId(row.run.run_id)}
                    </button>
                  {:else}
                    —
                  {/if}
                </td>
                <td>
                  <div>{row.deployment.served_model_name || row.deployment.model_name || "—"}</div>
                  {#if row.deployment.model_name && row.deployment.served_model_name && row.deployment.model_name !== row.deployment.served_model_name}
                    <div class="mono">{row.deployment.model_name}</div>
                  {/if}
                </td>
                <td class="mono" title={row.deployment.checkpoint_path || row.deployment.model_path || ""}>
                  {row.deployment.checkpoint_path || row.deployment.model_path || "—"}
                </td>
                <td>
                  {#if row.deployment.url}
                    <a class="pill-link" href={row.deployment.url} target="_blank" rel="noopener noreferrer">Endpoint</a>
                  {/if}
                </td>
                <td class="links-cell">
                  {#if row.run?.modal_app_url}
                    <a class="pill-link" href={row.run.modal_app_url} target="_blank" rel="noopener noreferrer">Modal</a>
                  {/if}
                  {#if row.run?.train_result?.wandb_url}
                    <a class="pill-link pill-link-wandb" href={row.run.train_result.wandb_url} target="_blank" rel="noopener noreferrer">W&B</a>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </MinimalTable>
      </div>
    {/if}
  </div>
</section>

<style>
  .summary-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-bottom: 24px;
  }

  .summary-card {
    border: 0;
    border-radius: 10px;
    background: color-mix(in srgb, #ffffff 7%, transparent);
    padding: 0.8rem 0.95rem;
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
  }

  .summary-label {
    color: var(--muted);
    font-size: 0.74rem;
    text-transform: none;
  }

  .summary-card strong {
    color: var(--text-bright);
    font-size: 1.36rem;
    font-weight: 600;
  }

  .runs-surface {
    border: 1px solid var(--border);
    background: var(--panel);
  }

  .runs-body {
    padding: 0;
  }

  .table-wrap {
    overflow-x: auto;
  }

  :global(table.runs-table) {
    width: 100%;
    min-width: 1080px;
  }

  .run-name {
    color: var(--text-bright);
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .mono {
    font-family: var(--font-mono);
    color: color-mix(in srgb, var(--accent) 78%, white);
    font-size: 0.72rem;
  }

  .cross-link {
    border: 0;
    background: transparent;
    color: color-mix(in srgb, var(--accent) 80%, white);
    padding: 0;
    font: inherit;
    cursor: pointer;
    font-family: var(--font-mono);
  }

  .cross-link:hover {
    color: var(--accent);
    text-decoration: underline;
  }

  .links-cell {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .pill-link {
    display: inline-block;
    border-radius: 6px;
    padding: 0.14rem 0.46rem;
    text-decoration: none;
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    font-size: 0.68rem;
    font-weight: 500;
  }

  .pill-link:hover {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
  }

  .pill-link-wandb {
    border-color: color-mix(in srgb, var(--yellow) 40%, transparent);
    color: var(--yellow);
    background: color-mix(in srgb, var(--yellow) 12%, transparent);
  }

  .pill-link-wandb:hover {
    background: color-mix(in srgb, var(--yellow) 18%, transparent);
  }

  .empty {
    padding: 2.5rem 2rem;
    color: var(--muted);
    text-align: center;
    font-size: 0.84rem;
  }

  @media (max-width: 1080px) {
    .summary-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 900px) {
    .summary-row {
      grid-template-columns: 1fr;
    }
  }
</style>
