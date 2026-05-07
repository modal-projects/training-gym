<script>
  import {
    ChevronDown,
    ExternalLink,
    Filter,
    PanelRightClose,
    Search,
  } from "lucide-svelte";
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";

  let {
    allDeployments,
    allEvals,
    loading,
    error,
    deploymentRows,
    deploymentLabel,
    truncateId,
    getStatus,
    onOpenTrainingRun,
  } = $props();

  let search = $state("");
  let selectedStatus = $state("all");
  let statusMenuOpen = $state(false);
  let selectedDeploymentKey = $state(null);
  let relatedRunExpanded = $state(true);
  let relatedEvalsExpanded = $state(true);

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function normalizePath(value) {
    return safeText(value).replace(/\/+$/, "").toLowerCase();
  }

  function normalizeStatus(row) {
    const raw = safeText(row.deployment.status).toLowerCase();
    if (
      raw === "ready" ||
      raw === "healthy" ||
      raw === "active" ||
      raw === "completed"
    ) {
      return "Ready";
    }
    if (
      raw === "running" ||
      raw === "pending" ||
      raw === "initializing" ||
      raw === "starting" ||
      raw === "deploying"
    ) {
      return "Pending";
    }
    if (
      raw === "inactive" ||
      raw === "failed" ||
      raw === "stopped" ||
      raw === "error"
    ) {
      return "Inactive";
    }
    if (row.deployment.url && row.run?.train_result) return "Ready";
    return "Pending";
  }

  function rowKey(row, rowIndex) {
    return (
      row.deployment.deployment_id ||
      row.deployment.app_name ||
      row.deployment.url ||
      `${row.deployment.model_name || "deployment"}-${rowIndex}`
    );
  }

  function statusTone(status) {
    if (status === "Ready") return "ready";
    if (status === "Inactive") return "inactive";
    return "pending";
  }

  function matchesSearch(row, query) {
    if (!query) return true;
    return [
      row.deployment.deployment_id,
      row.deployment.app_name,
      row.deployment.served_model_name,
      row.deployment.model_name,
      row.run?.run_id,
      row.run?.train_result?.training_run_id,
    ].some((value) => safeText(value).toLowerCase().includes(query));
  }

  function evalScore(evalRun) {
    if (typeof evalRun.mean === "number") return evalRun.mean;
    const rows = Array.isArray(evalRun.rows) ? evalRun.rows : [];
    if (!rows.length) return 0;
    return rows.reduce((sum, row) => sum + (row.score || 0), 0) / rows.length;
  }

  function evalDataset(evalRun) {
    const config = evalRun.config || {};
    return (
      safeText(config.dataset?.name) ||
      safeText(config.dataset?.hf_repo) ||
      safeText(config.dataset?.prompt_data) ||
      "eval"
    );
  }

  function relatedEvals(row) {
    const runId = safeText(row.run?.run_id || row.run?.train_result?.training_run_id);
    const deploymentModel = safeText(
      row.deployment.model_name || row.deployment.served_model_name,
    );
    const deploymentUrl = normalizePath(row.deployment.url);
    return allEvals
      .filter((evalRun) => {
        const config = evalRun.config || {};
        const deploymentConfig = config.deployment || {};
        const evalRunId = safeText(
          evalRun.training_run_id || config.training_run_id || config.run_id,
        );
        if (runId && evalRunId && runId === evalRunId) return true;
        const evalUrl = normalizePath(
          deploymentConfig.url || deploymentConfig.endpoint || "",
        );
        if (deploymentUrl && evalUrl && deploymentUrl === evalUrl) return true;
        const evalModel = safeText(
          deploymentConfig.model_name ||
            deploymentConfig.served_model_name ||
            config.model?.model_name,
        );
        return !!deploymentModel && !!evalModel && deploymentModel === evalModel;
      })
      .map((evalRun) => ({
        evalId: evalRun.eval_id || "",
        dataset: evalDataset(evalRun),
        score: evalScore(evalRun),
        createdAt: evalRun.created_at || 0,
      }))
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, 8);
  }

  function endpointHost(url) {
    const value = safeText(url);
    if (!value) return "—";
    try {
      return new URL(value).host;
    } catch {
      return value.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
    }
  }

  let enrichedRows = $derived.by(() =>
    deploymentRows.map((row, index) => {
      const status = normalizeStatus(row);
      return {
        ...row,
        key: rowKey(row, index),
        status,
        statusTone: statusTone(status),
        baseModel:
          row.deployment.served_model_name || row.deployment.model_name || "—",
      };
    }),
  );

  let statusCounts = $derived.by(() =>
    enrichedRows.reduce((acc, row) => {
      acc[row.status] = (acc[row.status] || 0) + 1;
      return acc;
    }, {}),
  );

  let statusFilters = $derived.by(() => ["Ready", "Pending", "Inactive"]);

  let readyDeployments = $derived(statusCounts.Ready || 0);
  let pendingDeployments = $derived(statusCounts.Pending || 0);

  let filteredRows = $derived.by(() => {
    const query = search.trim().toLowerCase();
    return enrichedRows.filter((row) => {
      if (selectedStatus !== "all" && row.status !== selectedStatus) return false;
      return matchesSearch(row, query);
    });
  });

  let selectedDeployment = $derived.by(
    () => filteredRows.find((row) => row.key === selectedDeploymentKey) || null,
  );

  let selectedDeploymentEvals = $derived.by(() =>
    selectedDeployment ? relatedEvals(selectedDeployment) : [],
  );

  $effect(() => {
    if (!selectedDeploymentKey) return;
    if (!filteredRows.some((row) => row.key === selectedDeploymentKey)) {
      selectedDeploymentKey = null;
    }
  });

  function selectDeployment(row) {
    selectedDeploymentKey = row.key;
  }

  function closeDetails() {
    selectedDeploymentKey = null;
  }

  function switchStatusFilter(status) {
    selectedStatus = status;
    statusMenuOpen = false;
  }
</script>

<svelte:window onclick={() => (statusMenuOpen = false)} />

<section class="summary-row" class:compact={!!selectedDeployment}>
  <article class="summary-card">
    <span class="summary-label">Total deployments</span>
    <strong>{allDeployments.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Deployment-ready</span>
    <strong>{readyDeployments}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending deployments</span>
    <strong>{pendingDeployments}</strong>
  </article>
</section>

<section class="deployments-layout" class:with-panel={!!selectedDeployment}>
  <div class="table-pane">
    <div class="filters-row">
      <label class="search-wrap" aria-label="Search deployments">
        <span class="search-icon"><Search size={13} /></span>
        <input
          type="search"
          class="search-input"
          placeholder="Search"
          bind:value={search}
          autocomplete="off"
          spellcheck="false"
        />
      </label>
      <div class="menu-wrap">
        <button
          class="status-filter"
          class:open={statusMenuOpen}
          onclick={(event) => {
            event.stopPropagation();
            statusMenuOpen = !statusMenuOpen;
          }}
        >
          <Filter size={12} />
          <span>Status</span>
          <ChevronDown size={12} class={statusMenuOpen ? "rotated" : ""} />
        </button>
        {#if statusMenuOpen}
          <div class="status-menu">
            <button class="status-item" onclick={() => switchStatusFilter("all")}>
              <span>All</span>
              <span class="status-count">{allDeployments.length}</span>
            </button>
            {#each statusFilters as status (status)}
              <button class="status-item" onclick={() => switchStatusFilter(status)}>
                <span>{status}</span>
                <span class="status-count">{statusCounts[status] || 0}</span>
              </button>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <div class="runs-body">
    {#if loading}
      <div class="table-wrap">
        <MinimalTableSkeleton
          class="deployments-table"
          columns={["Name", "Training run name", "Status", "Base model", ""]}
          rows={6}
        />
      </div>
    {:else if error}
      <div class="empty">Failed to load: {error}</div>
    {:else if !allDeployments.length}
      <div class="empty">No deployments recorded yet.</div>
    {:else}
      <div class="table-wrap">
        <MinimalTable class="deployments-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Training run name</th>
              <th>Status</th>
              <th>Base model</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each filteredRows as row (row.key)}
              <tr
                class:row-selected={selectedDeployment?.key === row.key}
                onclick={() => selectDeployment(row)}
              >
                <td class="name-cell">
                  <div class="run-name">{row.deployment.deployment_id || deploymentLabel(row.deployment)}</div>
                  {#if row.deployment.app_name && row.deployment.app_name !== row.deployment.deployment_id}
                    <div class="subtle">{row.deployment.app_name}</div>
                  {/if}
                </td>
                <td class="training-cell">
                  {#if row.run}
                    <button
                      class="cross-link"
                      onclick={(event) => {
                        event.stopPropagation();
                        onOpenTrainingRun(row.run.run_id);
                      }}
                    >
                      {row.run.run_id}
                    </button>
                  {:else}
                    —
                  {/if}
                </td>
                <td>
                  <span class={`deployment-status deployment-status-${row.statusTone}`}>
                    <span class="status-dot"></span>
                    <span>{row.status}</span>
                  </span>
                </td>
                <td class="model-cell" title={row.baseModel}>
                  {row.baseModel}
                </td>
                <td class="open-link-cell">
                  {#if row.deployment.modal_app_url || row.run?.modal_app_url}
                    <a
                      class="open-modal-link"
                      href={row.deployment.modal_app_url || row.run?.modal_app_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onclick={(event) => event.stopPropagation()}
                    >
                      <span>Open in Modal</span>
                      <ExternalLink size={12} />
                    </a>
                  {:else}
                    <span class="open-modal-link open-modal-link-disabled">
                      <span>Open in Modal</span>
                      <ExternalLink size={12} />
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
  </div>

  {#if selectedDeployment}
    <aside class="details-pane">
      <div class="details-header">
        <div>
          <div class="details-eyebrow">Deployment</div>
          <h2>{selectedDeployment.deployment.deployment_id || deploymentLabel(selectedDeployment.deployment)}</h2>
        </div>
        <div class="details-actions">
          {#if selectedDeployment.deployment.modal_app_url || selectedDeployment.run?.modal_app_url}
            <a
              class="open-modal-link"
              href={selectedDeployment.deployment.modal_app_url || selectedDeployment.run?.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>Open in Modal</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="close-panel" onclick={closeDetails} aria-label="Close deployment details">
            <PanelRightClose size={15} />
          </button>
        </div>
      </div>

      <section class="details-meta">
        <div class="meta-row">
          <span class="meta-key">Status</span>
          <span class={`deployment-status deployment-status-${selectedDeployment.statusTone}`}>
            <span class="status-dot"></span>
            <span>{selectedDeployment.status}</span>
          </span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Model</span>
          <span class="meta-value">{selectedDeployment.baseModel}</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Training run</span>
          <span class="meta-value">
            {selectedDeployment.run?.run_id || "—"}
          </span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Version</span>
          <span class="meta-value">
            {selectedDeployment.deployment.version ||
              (selectedDeployment.run?.train_result ? "Trained" : "Base")}
          </span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Endpoint</span>
          <span class="meta-value endpoint-value">
            {endpointHost(selectedDeployment.deployment.url)}
          </span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Created</span>
          <span class="meta-value">
            <TimeAgo
              timestamp={selectedDeployment.deployment.created_at || selectedDeployment.run?.created_at}
              showJustNow
              falsyRepresentation="—"
            />
          </span>
        </div>
      </section>

      <section class="details-section">
        <button
          class="section-toggle"
          onclick={() => (relatedRunExpanded = !relatedRunExpanded)}
        >
          <span>Related training run</span>
          <ChevronDown size={13} class={relatedRunExpanded ? "rotated" : ""} />
        </button>
        {#if relatedRunExpanded}
          {#if selectedDeployment.run}
            <button
              class="related-run-row"
              onclick={() => onOpenTrainingRun(selectedDeployment.run.run_id)}
            >
              <StatusPill status={getStatus(selectedDeployment.run)} />
              <span class="mono">{selectedDeployment.run.run_id}</span>
            </button>
          {:else}
            <div class="related-empty">No linked training run.</div>
          {/if}
        {/if}
      </section>

      <section class="details-section">
        <button
          class="section-toggle"
          onclick={() => (relatedEvalsExpanded = !relatedEvalsExpanded)}
        >
          <span>Related evals</span>
          <ChevronDown size={13} class={relatedEvalsExpanded ? "rotated" : ""} />
        </button>
        {#if relatedEvalsExpanded}
          {#if selectedDeploymentEvals.length}
            <div class="eval-list">
              {#each selectedDeploymentEvals as evalRun (`${evalRun.dataset}-${evalRun.evalId}`)}
                <div class="eval-row">
                  <div class="eval-left">
                    <span class="eval-chip">{evalRun.dataset}</span>
                    <span class="mono">{truncateId(evalRun.evalId)}</span>
                  </div>
                  <span class="eval-score">{(evalRun.score * 100).toFixed(1)}%</span>
                </div>
              {/each}
            </div>
          {:else}
            <div class="related-empty">No evals found for this deployment.</div>
          {/if}
        {/if}
      </section>
    </aside>
  {/if}
</section>

<style>
  .summary-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 24px;
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

  .summary-row.compact {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .deployments-layout {
    border: 0;
    background: transparent;
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    min-height: 520px;
    padding: 0;
  }

  .deployments-layout.with-panel {
    grid-template-columns: minmax(0, 1fr) 380px;
  }

  .table-pane {
    min-width: 0;
  }

  .filters-row {
    border-bottom: 0;
    padding: 0 0 24px;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .search-wrap {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--panel);
    min-width: 220px;
    width: min(320px, 100%);
    padding: 0.24rem 0.55rem;
  }

  .search-icon {
    display: inline-flex;
    color: var(--muted);
  }

  .search-input {
    border: 0;
    outline: 0;
    background: transparent;
    color: var(--text);
    width: 100%;
    min-width: 0;
    font: inherit;
    font-size: 0.78rem;
  }

  .search-input::placeholder {
    color: var(--muted);
  }

  .menu-wrap {
    position: relative;
  }

  .status-filter {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--panel);
    color: var(--text);
    font: inherit;
    font-size: 0.76rem;
    padding: 0.25rem 0.58rem;
    cursor: pointer;
  }

  .status-filter.open {
    border-color: var(--border-strong);
    background: var(--panel-alt);
    color: var(--text-bright);
  }

  .status-menu {
    position: absolute;
    top: calc(100% + 0.3rem);
    left: 0;
    z-index: 10;
    width: 175px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
    box-shadow: 0 10px 24px color-mix(in srgb, black 55%, transparent);
    padding: 0.25rem;
  }

  .status-item {
    width: 100%;
    border: 0;
    background: transparent;
    color: var(--text);
    padding: 0.34rem 0.4rem;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font: inherit;
    font-size: 0.74rem;
    text-align: left;
    cursor: pointer;
  }

  .status-item:hover {
    background: color-mix(in srgb, var(--text-bright) 6%, transparent);
    color: var(--text-bright);
  }

  .status-count {
    color: var(--muted);
    font-size: 0.68rem;
  }

  .rotated {
    transform: rotate(180deg);
  }

  .runs-body {
    padding: 0;
  }

  .table-wrap {
    overflow-x: auto;
  }

  :global(table.deployments-table) {
    width: 100%;
    min-width: 840px;
    border: 0;
    border-radius: 0;
  }

  :global(table.deployments-table tbody tr) {
    cursor: pointer;
  }

  :global(table.deployments-table tr.row-selected td) {
    background: color-mix(in srgb, var(--accent) 7%, transparent);
  }

  .name-cell {
    width: 24%;
  }

  .training-cell {
    width: 30%;
  }

  .model-cell {
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-name {
    color: var(--text-bright);
    font-family: var(--font-mono);
    font-size: 0.73rem;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .subtle {
    color: var(--muted-strong);
    font-size: 0.7rem;
    margin-top: 0.1rem;
  }

  .cross-link {
    border: 0;
    background: transparent;
    color: color-mix(in srgb, var(--text) 86%, white);
    padding: 0;
    font: inherit;
    font-size: 0.74rem;
    cursor: pointer;
    text-align: left;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 280px;
  }

  .cross-link:hover {
    color: var(--text-bright);
  }

  .deployment-status {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    border-radius: 999px;
    padding: 0.1rem 0.46rem;
    border: 1px solid transparent;
  }

  .status-dot {
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 999px;
    background: currentColor;
  }

  .deployment-status-ready {
    color: var(--green);
    border-color: color-mix(in srgb, var(--green) 35%, transparent);
    background: color-mix(in srgb, var(--green) 10%, transparent);
  }

  .deployment-status-pending {
    color: var(--yellow);
    border-color: color-mix(in srgb, var(--yellow) 35%, transparent);
    background: color-mix(in srgb, var(--yellow) 10%, transparent);
  }

  .deployment-status-inactive {
    color: var(--red);
    border-color: color-mix(in srgb, var(--red) 35%, transparent);
    background: color-mix(in srgb, var(--red) 10%, transparent);
  }

  .open-link-cell {
    width: 145px;
  }

  .open-modal-link {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.16rem 0.48rem;
    text-decoration: none;
    color: var(--text);
    font-size: 0.7rem;
    white-space: nowrap;
    background: var(--panel);
  }

  .open-modal-link:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }

  .open-modal-link-disabled {
    opacity: 0.5;
    pointer-events: none;
  }

  .details-pane {
    border-left: 1px solid var(--border);
    min-width: 0;
    background: color-mix(in srgb, var(--panel) 70%, var(--bg-depth));
    overflow-y: auto;
  }

  .details-header {
    padding: 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.6rem;
  }

  .details-eyebrow {
    color: var(--muted);
    font-size: 0.72rem;
    margin-bottom: 0.25rem;
  }

  h2 {
    color: var(--text-bright);
    font-size: 1.32rem;
    font-family: var(--font-mono);
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .details-actions {
    display: flex;
    align-items: center;
    gap: 0.38rem;
  }

  .close-panel {
    border: 1px solid var(--border);
    border-radius: 7px;
    background: transparent;
    color: var(--muted);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.24rem;
    cursor: pointer;
  }

  .close-panel:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }

  .details-meta {
    padding: 24px;
    border-bottom: 1px solid var(--border);
  }

  .meta-row {
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr);
    gap: 0.45rem;
    align-items: baseline;
    padding: 0.16rem 0;
  }

  .meta-key {
    color: var(--muted);
    font-size: 0.73rem;
  }

  .meta-value {
    color: var(--text);
    font-size: 0.75rem;
    overflow-wrap: anywhere;
  }

  .endpoint-value {
    font-family: var(--font-mono);
    font-size: 0.72rem;
  }

  .details-section {
    padding: 24px;
    border-bottom: 1px solid var(--border);
  }

  .section-toggle {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel-alt);
    color: var(--text-bright);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.35rem 0.55rem;
    font: inherit;
    font-size: 0.76rem;
    cursor: pointer;
  }

  .related-run-row {
    width: 100%;
    margin-top: 0.45rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--text);
    padding: 0.38rem 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    text-align: left;
  }

  .related-run-row:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }

  .mono {
    font-family: var(--font-mono);
    font-size: 0.7rem;
  }

  .eval-list {
    margin-top: 0.45rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .eval-row {
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 0.28rem 0.45rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.55rem;
    background: color-mix(in srgb, var(--panel-alt) 70%, transparent);
  }

  .eval-left {
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.34rem;
    overflow: hidden;
  }

  .eval-chip {
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--muted);
    padding: 0.08rem 0.32rem;
    font-size: 0.64rem;
    white-space: nowrap;
  }

  .eval-score {
    color: var(--yellow);
    font-size: 0.69rem;
    font-variant-numeric: tabular-nums;
  }

  .related-empty {
    margin-top: 0.46rem;
    color: var(--muted);
    font-size: 0.74rem;
  }

  .empty {
    padding: 24px;
    color: var(--muted);
    text-align: center;
    font-size: 0.84rem;
  }

  @media (max-width: 1240px) {
    .deployments-layout.with-panel {
      grid-template-columns: minmax(0, 1fr);
    }

    .details-pane {
      border-left: 0;
      border-top: 1px solid var(--border);
    }
  }

  @media (max-width: 1080px) {
    .summary-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 760px) {
    .summary-row,
    .summary-row.compact {
      grid-template-columns: 1fr;
    }

    .filters-row {
      flex-direction: column;
      align-items: stretch;
    }

    .search-wrap {
      width: 100%;
    }
  }
</style>
