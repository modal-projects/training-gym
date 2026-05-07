<script>
  import { ExternalLink, PanelRightClose } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBar from "../components/FilterBar.svelte";
  import MinimalTable from "../components/MinimalTable.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";

  let {
    allRuns,
    completedTotal,
    pendingTotal,
    frameworks,
    fwCounts,
    activeFrameworks,
    statuses,
    statusCounts,
    activeStatuses,
    filteredRuns,
    loading,
    error,
    modelName,
    getStatus,
    fmtCluster,
    fmtDuration,
    search = $bindable(),
    onToggleFramework,
    onToggleAllFrameworks,
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
    <strong>{pendingTotal}</strong>
  </article>
</section>

<section class="runs-surface">
  <div class="filters-row">
    <FilterBar
      {frameworks}
      {fwCounts}
      {activeFrameworks}
      allActive={activeFrameworks.size === frameworks.length}
      {statuses}
      {statusCounts}
      {activeStatuses}
      totalRuns={allRuns.length}
      bind:search
      onToggleFramework={onToggleFramework}
      onToggleAllFrameworks={onToggleAllFrameworks}
      onToggleStatus={onToggleStatus}
    />
  </div>

  <div class="runs-body">
    {#if loading}
      <div class="empty">Loading runs...</div>
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
              <th>Run</th>
              <th>Status</th>
              <th>Model</th>
              <th>Cluster</th>
              <th>Started</th>
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
                    <span
                      class="status-pill"
                      class:status-pending={status === "Pending"}
                      class:status-completed={status === "Completed"}
                    >
                      {status}
                    </span>
                  </button>
                </td>
                <td class="model-cell" title={modelName(run)}>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {modelName(run)}
                  </button>
                </td>
                <td>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {fmtCluster(run.config_summary)}
                  </button>
                </td>
                <td>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    <TimeAgo timestamp={run.started_at || run.created_at} showJustNow />
                  </button>
                </td>
                <td class="modal-link-cell">
                  {#if run.modal_app_url}
                    <a
                      class="open-modal-link"
                      href={run.modal_app_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onclick={(event) => event.stopPropagation()}
                    >
                      Open in Modal
                    </a>
                  {:else}
                    <span class="open-modal-link open-modal-link-disabled">
                      Open in Modal
                    </span>
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
          <span
            class="status-pill"
            class:status-pending={getStatus(selectedRun) === "Pending"}
            class:status-completed={getStatus(selectedRun) === "Completed"}
          >
            {getStatus(selectedRun)}
          </span>
        </div>
        <div class="drawer-kv">
          <span class="drawer-key">Model</span>
          <span class="drawer-value">{modelName(selectedRun)}</span>
        </div>
        <div class="drawer-kv">
          <span class="drawer-key">Framework</span>
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
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-bottom: 0.7rem;
  }

  .training-summary {
    margin-bottom: 0.75rem;
  }

  .summary-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
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
    border: 0;
    background: transparent;
  }

  .filters-row {
    margin: 0.05rem 0 0.55rem;
  }

  .runs-body {
    padding: 0;
  }

  .table-wrap {
    overflow-x: auto;
  }

  :global(table.runs-table) {
    width: 100%;
    min-width: 920px;
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
    width: 22%;
  }

  .model-cell {
    width: 24%;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 9999px;
    border: 1px solid transparent;
    padding: 0.13rem 0.56rem;
    font-size: 0.72rem;
    font-weight: 500;
  }

  .status-pending {
    color: #d18be7;
    border-color: color-mix(in srgb, #d18be7 35%, transparent);
    background: color-mix(in srgb, #d18be7 12%, transparent);
  }

  .status-completed {
    color: var(--green);
    border-color: color-mix(in srgb, var(--green) 36%, transparent);
    background: color-mix(in srgb, var(--green) 10%, transparent);
  }

  .modal-link-cell {
    text-align: right;
    width: 1%;
  }

  .open-modal-link {
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.2rem 0.58rem;
    color: var(--text);
    text-decoration: none;
    font-size: 0.72rem;
    font-weight: 500;
    background: var(--panel-alt);
  }

  .open-modal-link:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
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
