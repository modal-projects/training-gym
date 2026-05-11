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
    <span class="summary-label">Pending runs</span>
    <strong>{runningTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Stopped runs</span>
    <strong>{stoppedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Failed runs</span>
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
        <MinimalTable class="runs-table training-runs-table">
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
            {#each filteredRuns as run, runIndex (`${run.run_id || "run"}-${run.created_at || 0}-${runIndex}`)}
              {@const runName = run.run_id || "—"}
              {@const status = getStatus(run)}
              <tr class:row-selected={selectedRunId === run.run_id}>
                <td class="run-cell">
                  <button
                    class="cell-open-button"
                    title={runName}
                    onclick={() => selectRun(run.run_id)}
                  >
                    <div class="run-name">{runName}</div>
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
        <div class="drawer-header-left">
          <div class="drawer-eyebrow">Training run</div>
          <h2 class="drawer-run-id" title={selectedRun.run_id}>{selectedRun.run_id}</h2>
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
    gap: 14px;
    margin-bottom: 24px;
  }

  .training-summary {
    margin-bottom: 24px;
  }

  .summary-card {
    border: 0;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.03);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    height: 80px;
  }

  .summary-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 400;
    line-height: 16px;
  }

  .summary-card strong {
    color: var(--text-bright);
    font-size: 20px;
    font-weight: 500;
    line-height: 32px;
  }

  .runs-surface {
    border: 0;
    background: transparent;
    display: flex;
    flex-direction: column;
    gap: 24px;
    padding: 0;
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

  :global(table.training-runs-table) {
    table-layout: fixed;
  }

  :global(table.runs-table tr.row-selected td) {
    background: color-mix(in srgb, var(--accent) 6%, transparent);
  }

  .cell-open-button {
    border: 0;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 14px;
    line-height: 20px;
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
    display: block;
    color: var(--text-bright);
    font-family: var(--font-mono);
    font-weight: 400;
    font-size: 14px;
    line-height: 20px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
    gap: 6px;
    white-space: nowrap;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 4px 8px;
    text-decoration: none;
    font-size: 12px;
    font-weight: 500;
    line-height: 16px;
    background: transparent;
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
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .drawer-header-left {
    min-width: 0;
    overflow: hidden;
  }

  .drawer-eyebrow {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
    margin-bottom: 4px;
  }

  .drawer-run-id {
    color: var(--text-bright);
    font-size: 16px;
    font-weight: 500;
    font-family: var(--font-mono);
    line-height: 24px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .drawer-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .view-app-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 4px 8px;
    text-decoration: none;
    color: var(--muted);
    font-size: 12px;
    font-weight: 500;
    line-height: 16px;
  }

  .view-app-link:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .drawer-close {
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
  }

  .drawer-close:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .drawer-section {
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
    padding: 16px 20px;
  }

  .drawer-section-title {
    color: var(--text-bright);
    font-size: 14px;
    font-weight: 500;
    line-height: 20px;
    margin-bottom: 8px;
  }

  .drawer-kv {
    display: grid;
    grid-template-columns: 100px minmax(0, 1fr);
    align-items: baseline;
    gap: 8px;
    padding: 4px 0;
  }

  .drawer-key {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
  }

  .drawer-value {
    color: var(--text);
    font-size: 14px;
    line-height: 20px;
    overflow-wrap: anywhere;
  }

  .drawer-value-mono {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
  }

  .drawer-empty {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
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
