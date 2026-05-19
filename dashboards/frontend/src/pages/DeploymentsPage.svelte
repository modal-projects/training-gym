<script>
  import {
    ChevronDown,
    ExternalLink,
    Filter,
    PanelRightClose,
    Search,
  } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
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
    focusDeploymentRef,
    onFocusResolved,
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

  function toTimestampSeconds(value) {
    if (value && typeof value === "object" && "value" in value) {
      return toTimestampSeconds(value.value);
    }
    if (typeof value === "number") return Number.isFinite(value) ? value : 0;
    const text = safeText(value).trim();
    if (!text) return 0;
    const numeric = Number(text);
    if (Number.isFinite(numeric)) return numeric;
    const epochMs = Date.parse(text);
    if (Number.isFinite(epochMs)) return Math.floor(epochMs / 1000);
    return 0;
  }

  function rowCreatedAt(row) {
    return (
      toTimestampSeconds(row.deployment.created_at) ||
      toTimestampSeconds(row.run?.created_at) ||
      toTimestampSeconds(row.run?.started_at) ||
      0
    );
  }

  function normalizeStatus(row) {
    const raw = safeText(row.deployment.status).trim().toLowerCase();
    const hasEndpoint = !!safeText(row.deployment.url).trim();
    const createdAt = rowCreatedAt(row);
    const ageSeconds = createdAt > 0 ? Math.floor(Date.now() / 1000) - createdAt : 0;
    const endpointCreationTimedOut = !hasEndpoint && ageSeconds > 3600;
    if (
      raw.includes("ready") ||
      raw.includes("healthy") ||
      raw.includes("active") ||
      raw.includes("available") ||
      raw.includes("serving") ||
      raw.includes("online") ||
      raw.includes("completed") ||
      raw.includes("success")
    ) {
      return "Ready";
    }
    if (
      raw.includes("running") ||
      raw.includes("pending") ||
      raw.includes("initializing") ||
      raw.includes("starting") ||
      raw.includes("deploying") ||
      raw.includes("creating") ||
      raw.includes("provisioning") ||
      raw.includes("building")
    ) {
      return "Pending";
    }
    if (
      raw.includes("inactive") ||
      raw.includes("failed") ||
      raw.includes("stopped") ||
      raw.includes("error") ||
      raw.includes("deleted") ||
      raw.includes("terminated")
    ) {
      return "Inactive";
    }
    if (endpointCreationTimedOut) return "Inactive";
    if (row.run && getStatus(row.run) === "Failed") return "Inactive";
    return "Pending";
  }

  function rowKey(row, rowIndex) {
    const base =
      row.deployment.deployment_id ||
      row.deployment.app_name ||
      row.deployment.url ||
      row.deployment.model_name ||
      "deployment";
    return `${base}-${rowIndex}`;
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

  function deploymentRefMatches(row, targetRef) {
    const rawTarget = safeText(targetRef).trim();
    if (!rawTarget) return false;
    const normalizedTarget = normalizePath(rawTarget);
    const candidates = [
      safeText(row.deployment.deployment_id).trim(),
      safeText(row.deployment.app_name).trim(),
      safeText(row.deployment.modal_app_id).trim(),
      normalizePath(row.deployment.url),
    ];
    return candidates.some((candidate) => {
      const value = safeText(candidate).trim();
      if (!value) return false;
      if (value === rawTarget) return true;
      if (normalizedTarget && normalizePath(value) === normalizedTarget) return true;
      return false;
    });
  }

  let enrichedRows = $derived.by(() =>
    deploymentRows.map((row, index) => {
      const status = normalizeStatus(row);
      return {
        ...row,
        key: rowKey(row, index),
        status,
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

  $effect(() => {
    const targetRef = safeText(focusDeploymentRef).trim();
    if (!targetRef) return;
    const match = enrichedRows.find((row) => deploymentRefMatches(row, targetRef));
    if (match) {
      selectedDeploymentKey = match.key;
      onFocusResolved?.();
      return;
    }
    if (!loading) {
      onFocusResolved?.();
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

<section class="summary-row">
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

<section class="deployments-layout">
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
              {@const deploymentName = row.deployment.deployment_id || deploymentLabel(row.deployment)}
              {@const deploymentAppName =
                row.deployment.app_name && row.deployment.app_name !== row.deployment.deployment_id
                  ? row.deployment.app_name
                  : ""}
              <tr
                class:row-selected={selectedDeployment?.key === row.key}
                onclick={() => selectDeployment(row)}
              >
                <td class="name-cell">
                  <div class="run-name" title={deploymentName}>{deploymentName}</div>
                  {#if deploymentAppName}
                    <div class="subtle" title={deploymentAppName}>{deploymentAppName}</div>
                  {/if}
                </td>
                <td class="training-cell">
                  {#if row.run}
                    <button
                      class="cross-link"
                      title={row.run.run_id}
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
                  <StatusPill status={row.status} />
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
    <Drawer open={!!selectedDeployment} onclose={closeDetails}>
      <div class="deployment-drawer">
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
          <StatusPill status={selectedDeployment.status} />
        </div>
        <div class="meta-row">
          <span class="meta-key">Model</span>
          <span class="meta-value">{selectedDeployment.baseModel}</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Training run</span>
          <span
            class="meta-value training-run-id-value"
            title={selectedDeployment.run?.run_id || ""}
          >
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

      {#if selectedDeployment.run}
        <section class="details-section">
          <button
            class="section-toggle"
            onclick={() => (relatedRunExpanded = !relatedRunExpanded)}
          >
            <span>Related training run</span>
            <ChevronDown size={13} class={relatedRunExpanded ? "rotated" : ""} />
          </button>
          {#if relatedRunExpanded}
            <button
              class="related-run-row"
              onclick={() => onOpenTrainingRun(selectedDeployment.run.run_id)}
            >
              <StatusPill status={getStatus(selectedDeployment.run)} />
              <span class="mono related-run-id" title={selectedDeployment.run.run_id}>
                {selectedDeployment.run.run_id}
              </span>
            </button>
          {/if}
        </section>
      {/if}

      {#if selectedDeploymentEvals.length}
        <section class="details-section">
          <button
            class="section-toggle"
            onclick={() => (relatedEvalsExpanded = !relatedEvalsExpanded)}
          >
            <span>Related evals</span>
            <ChevronDown size={13} class={relatedEvalsExpanded ? "rotated" : ""} />
          </button>
          {#if relatedEvalsExpanded}
            <div class="eval-list">
              {#each selectedDeploymentEvals as evalRun, evalIndex (`${evalRun.dataset}-${evalRun.evalId || "eval"}-${evalRun.createdAt || 0}-${evalIndex}`)}
                <div class="eval-row">
                  <div class="eval-left">
                    <span class="eval-chip">{evalRun.dataset}</span>
                    <span class="mono">{truncateId(evalRun.evalId)}</span>
                  </div>
                  <span class="eval-score">{(evalRun.score * 100).toFixed(1)}%</span>
                </div>
              {/each}
            </div>
          {/if}
        </section>
      {/if}
      </div>
    </Drawer>
  {/if}
</section>

<style>
  .summary-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
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

  .deployments-layout {
    border: 0;
    background: transparent;
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    min-height: 520px;
    padding: 0;
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
    gap: 8px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: transparent;
    width: 260px;
    padding: 6px 8px;
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
    font-size: 14px;
  }

  .search-input::placeholder {
    color: var(--color-foreground-tertiary, #747474);
  }

  .menu-wrap {
    position: relative;
  }

  .status-filter {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: var(--bg);
    color: var(--text);
    font: inherit;
    font-size: 14px;
    font-weight: 500;
    padding: 6px 8px;
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
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .subtle {
    color: var(--muted-strong);
    font-size: 0.7rem;
    margin-top: 0.1rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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

  .open-link-cell {
    width: 145px;
  }

  .open-modal-link {
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
    white-space: nowrap;
    background: transparent;
  }

  .open-modal-link:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }

  .open-modal-link-disabled {
    opacity: 0.5;
    pointer-events: none;
  }

  .deployment-drawer {
    width: min(420px, calc(100vw - 24px));
    height: 100%;
  }

  .details-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .details-eyebrow {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
    margin-bottom: 4px;
  }

  h2 {
    color: var(--text-bright);
    font-size: 16px;
    font-weight: 500;
    font-family: var(--font-mono);
    line-height: 24px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .details-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .close-panel {
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: transparent;
    color: var(--muted);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
    cursor: pointer;
  }

  .close-panel:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }

  .details-meta {
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
  }

  .meta-row {
    display: grid;
    grid-template-columns: 100px minmax(0, 1fr);
    gap: 8px;
    align-items: baseline;
    padding: 4px 0;
  }

  .meta-key {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
  }

  .meta-value {
    color: var(--text);
    font-size: 14px;
    line-height: 20px;
    overflow-wrap: anywhere;
  }

  .training-run-id-value {
    overflow-wrap: normal;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .endpoint-value {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
  }

  .details-section {
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
  }

  .section-toggle {
    width: 100%;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-bright);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 8px;
    font: inherit;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
  }

  .related-run-row {
    width: 100%;
    margin-top: 8px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: transparent;
    color: var(--text);
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    text-align: left;
  }

  .related-run-row:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }

  .mono {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
  }

  .related-run-id {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .eval-list {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .eval-row {
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 6px 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: rgba(255, 255, 255, 0.03);
  }

  .eval-left {
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 6px;
    overflow: hidden;
  }

  .eval-chip {
    border-radius: 4px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    background: transparent;
    color: var(--muted);
    padding: 2px 6px;
    font-size: 11px;
    line-height: 14px;
    white-space: nowrap;
  }

  .eval-score {
    color: var(--yellow);
    font-size: 12px;
    line-height: 16px;
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
