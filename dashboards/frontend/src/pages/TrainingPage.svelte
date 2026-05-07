<script>
  import { ExternalLink, PanelRightClose } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBar from "../components/FilterBar.svelte";
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";

  let {
    allRuns,
    completedTotal,
    runningTotal,
    stoppedTotal,
    failedTotal,
    recipes,
    recipeCounts,
    activeRecipes,
    statuses,
    statusCounts,
    activeStatuses,
    filteredRuns,
    loading,
    error,
    modelName,
    getStatus,
    fmtDuration,
    search = $bindable(),
    onToggleRecipe,
    onToggleAllRecipes,
    onToggleStatus,
  } = $props();

  let selectedRunId = $state(null);

  let selectedRun = $derived.by(
    () => allRuns.find((run) => run.run_id === selectedRunId) || null,
  );

  let selectedRecipeEntries = $derived.by(() => {
    if (!selectedRun) return [];
    const recipe = selectedRun.config?.recipe || selectedRun.config?.preset || {};
    return Object.entries(recipe).filter(
      ([, value]) => value !== undefined && value !== null && String(value) !== "",
    );
  });

  function selectRun(runId) {
    selectedRunId = runId;
  }

  function closeDrawer() {
    selectedRunId = null;
  }

  function formatFieldLabel(field) {
    return field.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function formatFieldValue(value) {
    if (typeof value === "number") {
      return Number.isInteger(value) ? String(value) : value.toExponential(1);
    }
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function runDuration(run) {
    if (typeof run.duration_seconds === "number" && run.duration_seconds >= 0) {
      return fmtDuration(0, run.duration_seconds);
    }
    if (run.started_at) {
      return fmtDuration(run.started_at, run.ended_at);
    }
    return "—";
  }

  $effect(() => {
    if (selectedRunId && !allRuns.some((run) => run.run_id === selectedRunId)) {
      selectedRunId = null;
    }
  });
</script>

<section class="summary-row training-summary">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{allRuns.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Completed runs</span>
    <strong>{completedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending</span>
    <strong>{runningTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Stopped</span>
    <strong>{stoppedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Failed</span>
    <strong>{failedTotal}</strong>
  </article>
</section>

<section class="runs-surface">
  <div class="filters-row">
    <FilterBar
      {recipes}
      {recipeCounts}
      {activeRecipes}
      allRecipesActive={activeRecipes.size === recipes.length}
      {statuses}
      {statusCounts}
      {activeStatuses}
      totalRuns={allRuns.length}
      bind:search
      onToggleRecipe={onToggleRecipe}
      onToggleAllRecipes={onToggleAllRecipes}
      onToggleStatus={onToggleStatus}
    />
  </div>

  <div class="runs-body">
    {#if loading}
      <div class="table-wrap">
        <MinimalTableSkeleton
          class="runs-table"
          columns={["Name", "Status", "Model", "Dataset", "Recipe", ""]}
          rows={8}
        />
      </div>
    {:else if error}
      <div class="empty">Failed to load: {error}</div>
    {:else if !allRuns.length}
      <div class="empty">No training runs found yet.</div>
    {:else if !filteredRuns.length}
      <div class="empty">No runs match the current filters.</div>
    {:else}
      <div class="table-wrap">
        <MinimalTable class="runs-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Model</th>
              <th>Dataset</th>
              <th>Recipe</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each filteredRuns as run (run.run_id)}
              {@const status = getStatus(run)}
              <tr class:row-selected={selectedRunId === run.run_id}>
                <td class="run-cell">
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    <div class="run-name">{run.run_id}</div>
                  </button>
                </td>
                <td>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    <StatusPill status={status} />
                  </button>
                </td>
                <td class="model-cell" title={modelName(run)}>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {modelName(run)}
                  </button>
                </td>
                <td class="dataset-cell" title={run.config_summary?.dataset_name || "—"}>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {run.config_summary?.dataset_name || "—"}
                  </button>
                </td>
                <td>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {run.framework || "—"}
                  </button>
                </td>
                <td class="modal-link-cell">
                  <div class="modal-link-wrap">
                    {#if run.modal_app_url}
                      <a
                        class="open-modal-link"
                        href={run.modal_app_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onclick={(event) => event.stopPropagation()}
                      >
                        <span class="open-modal-link-label">Open in Modal</span>
                        <ExternalLink class="open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </a>
                    {:else}
                      <span class="open-modal-link open-modal-link-disabled">
                        <span class="open-modal-link-label">Open in Modal</span>
                        <ExternalLink class="open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </span>
                    {/if}
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </MinimalTable>
      </div>
    {/if}
  </div>
</section>

{#if selectedRun}
  <Drawer open={!!selectedRun} onclose={closeDrawer}>
    <div class="run-drawer" aria-label={`Training run ${selectedRun.run_id}`}>
      <div class="drawer-header">
        <div>
          <div class="drawer-eyebrow">Training run</div>
          <h2 class="drawer-run-id">{selectedRun.run_id}</h2>
        </div>
        <div class="drawer-actions">
          {#if selectedRun.modal_app_url}
            <a
              class="view-app-link"
              href={selectedRun.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>View in app</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="drawer-close" onclick={closeDrawer} aria-label="Close run drawer">
            <PanelRightClose size={16} />
          </button>
        </div>
      </div>

      <section class="drawer-section">
        <div class="drawer-kv">
          <span class="drawer-key">Status</span>
          <StatusPill status={getStatus(selectedRun)} />
        </div>
        <div class="drawer-kv">
          <span class="drawer-key">Model</span>
          <span class="drawer-value">{modelName(selectedRun)}</span>
        </div>
        <div class="drawer-kv">
          <span class="drawer-key">Recipe</span>
          <span class="drawer-value">{selectedRun.framework || "—"}</span>
        </div>
        <div class="drawer-kv">
          <span class="drawer-key">Duration</span>
          <span class="drawer-value">{runDuration(selectedRun)}</span>
        </div>
        <div class="drawer-kv">
          <span class="drawer-key">Started</span>
          <span class="drawer-value">
            <TimeAgo timestamp={selectedRun.started_at || selectedRun.created_at} showJustNow />
          </span>
        </div>
      </section>

      <section class="drawer-section">
        <h3 class="drawer-section-title">Training recipe</h3>
        {#if selectedRecipeEntries.length}
          {#each selectedRecipeEntries as [field, value] (field)}
            <div class="drawer-kv">
              <span class="drawer-key">{formatFieldLabel(field)}</span>
              <span class="drawer-value drawer-value-mono">{formatFieldValue(value)}</span>
            </div>
          {/each}
        {:else}
          <div class="drawer-empty">No recipe values found for this run.</div>
        {/if}
      </section>
    </div>
  </Drawer>
{/if}

<style>
  .summary-row {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 24px;
    margin-bottom: 24px;
  }

  .training-summary {
    margin-bottom: 24px;
  }

  .summary-card {
    border: 0;
    border-radius: 10px;
    background: color-mix(in srgb, #ffffff 7%, transparent);
    padding: 24px;
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
    border: 0;
    background: transparent;
    display: flex;
    flex-direction: column;
    gap: 24px;
    padding: 24px;
  }

  .filters-row {
    margin: 0;
  }

  .runs-body {
    padding: 0;
  }

  .table-wrap {
    overflow-x: auto;
  }

  :global(table.runs-table) {
    width: 100%;
    min-width: 860px;
  }

  :global(table.runs-table tr.row-selected td) {
    background: color-mix(in srgb, var(--accent) 6%, transparent);
  }

  .cell-open-button {
    border: 0;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 0.81rem;
    cursor: pointer;
    padding: 0;
    text-align: left;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cell-open-button:hover {
    color: var(--accent);
  }

  .run-name {
    color: var(--text-bright);
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .run-cell {
    width: 24%;
  }

  .model-cell {
    width: 20%;
  }

  .dataset-cell {
    width: 18%;
  }

  .modal-link-cell {
    min-width: 9.75rem;
    width: 12.5rem;
  }

  .modal-link-wrap {
    display: flex;
  }

  .open-modal-link {
    display: inline-flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.38rem;
    white-space: nowrap;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.2rem 0.58rem;
    text-decoration: none;
    font-size: 0.72rem;
    font-weight: 500;
    background: var(--panel);
  }

  .open-modal-link-label {
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  :global(.open-modal-link-icon) {
    color: var(--muted-strong);
  }

  .open-modal-link:hover {
    border-color: var(--border-strong);
  }

  .open-modal-link-disabled {
    opacity: 0.5;
    pointer-events: none;
  }

  .run-drawer {
    width: min(420px, calc(100vw - 24px));
    height: 100%;
  }

  .drawer-header {
    border-bottom: 1px solid var(--border);
    padding: 0.92rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
  }

  .drawer-eyebrow {
    color: var(--muted);
    font-size: 0.72rem;
    margin-bottom: 0.35rem;
  }

  .drawer-run-id {
    color: var(--text-bright);
    font-size: 1.1rem;
    font-weight: 500;
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .drawer-actions {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-shrink: 0;
  }

  .view-app-link {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.18rem 0.42rem;
    text-decoration: none;
    color: var(--text);
    font-size: 0.68rem;
  }

  .view-app-link:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .drawer-close {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.18rem;
  }

  .drawer-close:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .drawer-section {
    border-bottom: 1px solid var(--border);
    padding: 0.82rem 1rem 0.9rem;
  }

  .drawer-section-title {
    color: var(--text-bright);
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.55rem;
  }

  .drawer-kv {
    display: grid;
    grid-template-columns: 122px minmax(0, 1fr);
    align-items: baseline;
    gap: 0.55rem;
    padding: 0.2rem 0;
  }

  .drawer-key {
    color: var(--muted);
    font-size: 0.74rem;
  }

  .drawer-value {
    color: var(--text);
    font-size: 0.76rem;
    overflow-wrap: anywhere;
  }

  .drawer-value-mono {
    font-family: var(--font-mono);
    font-size: 0.72rem;
  }

  .drawer-empty {
    color: var(--muted);
    font-size: 0.76rem;
  }

  .empty {
    padding: 24px;
    color: var(--muted);
    text-align: center;
    font-size: 0.84rem;
  }

  @media (max-width: 900px) {
    .summary-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 640px) {
    .summary-row {
      grid-template-columns: 1fr;
    }
  }
</style>
