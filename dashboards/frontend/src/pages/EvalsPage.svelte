<script>
  import { ChevronDown, Filter, Search } from "lucide-svelte";
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import StatusPill from "../components/StatusPill.svelte";

  let {
    allEvals,
    evalCompletedTotal,
    evalPendingTotal,
    evalFailedTotal,
    loading,
    error,
    evalConfigGroups,
  } = $props();

  let search = $state("");
  let statusFilter = $state("all");
  let statusMenuOpen = $state(false);
  let expandedConfigIds = $state(new Set());
  let expandedInitialized = $state(false);

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function includesText(value, query) {
    return safeText(value).toLowerCase().includes(query);
  }

  function groupSubtitle(group) {
    const parts = [group.evalConfigId];
    if (group.meta.judge || group.meta.evalFn) {
      parts.push(group.meta.judge || group.meta.evalFn);
    }
    parts.push(`${group.deploymentCount} deployment${group.deploymentCount === 1 ? "" : "s"}`);
    return parts.join(" • ");
  }

  function evalBaseModel(run, group) {
    const config = run.eval.config || {};
    return (
      config.deployment?.served_model_name ||
      config.deployment?.model_name ||
      config.model?.model_name ||
      group.meta.model ||
      "—"
    );
  }

  function evalDeploymentName(run) {
    const config = run.eval.config || {};
    const deployment = config.deployment || {};
    if (deployment.app_name) return deployment.app_name;
    if (deployment.deployment_name) return deployment.deployment_name;
    if (deployment.name) return deployment.name;
    if (deployment.deployment_id) return deployment.deployment_id;
    if (deployment.url) {
      const value = safeText(deployment.url);
      try {
        return new URL(value).host;
      } catch {
        return value;
      }
    }
    return "—";
  }

  function toggleGroup(evalConfigId) {
    const next = new Set(expandedConfigIds);
    if (next.has(evalConfigId)) next.delete(evalConfigId);
    else next.add(evalConfigId);
    expandedConfigIds = next;
  }

  $effect(() => {
    if (expandedInitialized || !evalConfigGroups.length) return;
    expandedConfigIds = new Set(evalConfigGroups.map((group) => group.evalConfigId));
    expandedInitialized = true;
  });

  let filteredGroups = $derived.by(() => {
    const query = search.trim().toLowerCase();
    return evalConfigGroups
      .map((group) => {
        let runs = group.runs;
        if (statusFilter !== "all") {
          const wanted =
            statusFilter === "completed"
              ? "Completed"
              : statusFilter === "pending"
                ? "Pending"
                : "Failed";
          runs = runs.filter((run) => run.status === wanted);
        }
        if (query) {
          const groupMatches =
            includesText(group.evalConfigId, query) ||
            includesText(group.meta.dataset, query) ||
            includesText(group.meta.model, query) ||
            includesText(group.meta.judge, query) ||
            includesText(group.meta.evalFn, query);
          if (!groupMatches) {
            runs = runs.filter((run) => {
              const config = run.eval.config || {};
              return (
                includesText(run.eval.eval_id, query) ||
                includesText(config.deployment?.model_name, query) ||
                includesText(config.deployment?.served_model_name, query) ||
                includesText(config.deployment?.url, query)
              );
            });
          }
        }
        if (!runs.length) return null;
        return {
          ...group,
          visibleRuns: runs,
        };
      })
      .filter(Boolean);
  });
</script>

<svelte:window onclick={() => (statusMenuOpen = false)} />

<section class="summary-row">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{allEvals.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Completed runs</span>
    <strong>{evalCompletedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending runs</span>
    <strong>{evalPendingTotal}</strong>
  </article>
</section>

<section class="runs-surface">
  <div class="filters-row">
    <label class="search-wrap" aria-label="Search eval runs">
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
        <ChevronDown
          size={12}
          style={`transform: ${statusMenuOpen ? "rotate(180deg)" : "rotate(0deg)"};`}
        />
      </button>
      {#if statusMenuOpen}
        <div class="status-menu">
          <button class="status-item" onclick={() => ((statusFilter = "all"), (statusMenuOpen = false))}>
            <span>All</span>
            <span class="status-count">{allEvals.length}</span>
          </button>
          <button class="status-item" onclick={() => ((statusFilter = "completed"), (statusMenuOpen = false))}>
            <span>Completed</span>
            <span class="status-count">{evalCompletedTotal}</span>
          </button>
          <button class="status-item" onclick={() => ((statusFilter = "pending"), (statusMenuOpen = false))}>
            <span>Pending</span>
            <span class="status-count">{evalPendingTotal}</span>
          </button>
          <button class="status-item" onclick={() => ((statusFilter = "failed"), (statusMenuOpen = false))}>
            <span>Failed</span>
            <span class="status-count">{evalFailedTotal}</span>
          </button>
        </div>
      {/if}
    </div>
  </div>

  <div class="runs-body">
    {#if loading}
      <div class="table-wrap">
        <MinimalTableSkeleton
          class="runs-table"
          columns={["Base model", "Deployment name", "Status", "Average score", "Examples"]}
          rows={6}
        />
      </div>
    {:else if error}
      <div class="empty">Failed to load: {error}</div>
    {:else if !allEvals.length}
      <div class="empty">No eval results yet.</div>
    {:else}
      <div class="eval-groups">
        {#if !filteredGroups.length}
          <div class="empty">No evals match the current filters.</div>
        {/if}
        {#each filteredGroups as group (group.evalConfigId)}
          <section class="eval-group">
            <button class="eval-group-header" onclick={() => toggleGroup(group.evalConfigId)}>
              <div class="eval-group-title-wrap">
                <div class="eval-group-title">{group.meta.dataset || group.evalConfigId}</div>
                <div class="eval-group-subtitle">{groupSubtitle(group)}</div>
              </div>
              <div class="eval-group-meta">
                {#if group.meta.model}
                  <span class="group-meta-pill">{group.meta.model}</span>
                {/if}
                {#if group.meta.split}
                  <span class="group-meta-pill">split: {group.meta.split}</span>
                {/if}
                <span class="group-meta-pill">total evals: {group.totalEvals}</span>
                <span class="group-meta-pill">avg: {group.avgAccuracy.toFixed(4)}</span>
                <ChevronDown
                  size={14}
                  style={`color: var(--muted); transform: ${expandedConfigIds.has(group.evalConfigId) ? "rotate(180deg)" : "rotate(0deg)"};`}
                />
              </div>
            </button>

            {#if expandedConfigIds.has(group.evalConfigId)}
              <div class="table-wrap">
                <MinimalTable class="runs-table">
                  <thead>
                    <tr>
                      <th>Base model</th>
                      <th>Deployment name</th>
                      <th>Status</th>
                      <th>Average score</th>
                      <th>Examples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each group.visibleRuns as run (run.eval.eval_id || `${group.evalConfigId}-${run.eval.created_at || 0}`)}
                      <tr>
                        <td>{evalBaseModel(run, group)}</td>
                        <td class="mono">{evalDeploymentName(run)}</td>
                        <td>
                          <StatusPill status={run.status} />
                        </td>
                        <td class="eval-score">
                          {run.status === "Failed" ? "—" : run.avgScore.toFixed(4)}
                        </td>
                        <td>{run.totalRows ? run.totalRows : "—"}</td>
                      </tr>
                    {/each}
                  </tbody>
                </MinimalTable>
              </div>
            {/if}
          </section>
        {/each}
      </div>
    {/if}
  </div>
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

  .runs-surface {
    border: 0;
    background: transparent;
    padding: 0;
  }

  .filters-row {
    margin-bottom: 24px;
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

  .runs-body {
    padding: 0;
  }

  .empty {
    padding: 24px;
    color: var(--muted);
    text-align: center;
    font-size: 0.84rem;
  }

  .eval-groups {
    display: flex;
    flex-direction: column;
    gap: 24px;
    padding: 0;
  }

  .eval-group {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    overflow: hidden;
  }

  .eval-group-header {
    border: 0;
    width: 100%;
    text-align: left;
    background: transparent;
    color: inherit;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.8rem;
    padding: 0.75rem 0.9rem;
  }

  .eval-group-title-wrap {
    min-width: 0;
  }

  .eval-group-title {
    color: var(--text-bright);
    font-size: 0.86rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .eval-group-subtitle {
    color: var(--muted);
    font-size: 0.74rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .eval-group-meta {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  .group-meta-pill {
    border: 0;
    border-radius: 6px;
    padding: 0.11rem 0.42rem;
    font-size: 0.68rem;
    color: var(--muted);
    background: color-mix(in srgb, var(--panel-alt) 70%, transparent);
  }

  .table-wrap {
    overflow-x: auto;
  }

  :global(table.runs-table) {
    width: 100%;
    min-width: 1080px;
  }

  .mono {
    font-family: var(--font-mono);
    color: color-mix(in srgb, var(--accent) 78%, white);
    font-size: 0.72rem;
  }

  .eval-score {
    color: var(--text-bright);
    font-weight: 600;
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

    .filters-row {
      flex-direction: column;
      align-items: stretch;
    }

    .search-wrap {
      width: 100%;
    }
  }
</style>
