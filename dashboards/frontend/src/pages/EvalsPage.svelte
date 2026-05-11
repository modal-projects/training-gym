<script>
  import { ChevronDown, Filter, Search } from "lucide-svelte";
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import StatusPill from "../components/StatusPill.svelte";

  let {
    allEvals,
    deploymentRows,
    evalCompletedTotal,
    evalPendingTotal,
    evalFailedTotal,
    loading,
    error,
    evalConfigGroups,
    onOpenTrainingRun,
    onOpenDeployment,
  } = $props();

  let search = $state("");
  let statusFilter = $state("all");
  let statusMenuOpen = $state(false);
  let datasetFilter = $state("all");
  let datasetMenuOpen = $state(false);
  let expandedConfigIds = $state(new Set());
  let expandedInitialized = $state(false);

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function includesText(value, query) {
    return safeText(value).toLowerCase().includes(query);
  }

  function nonPlaceholderText(value) {
    const text = safeText(value).trim();
    if (!text || text === "—") return "";
    return text;
  }

  function evalConfigIdFallbackMeta(evalConfigId) {
    const raw = safeText(evalConfigId).trim();
    if (!raw) return { dataset: "", evalFn: "" };
    const parts = raw.split(".");
    if (parts.length < 3) return { dataset: "", evalFn: "" };
    const [prefix, dataset, evalFn] = parts;
    if (prefix !== "EvalConfig") return { dataset: "", evalFn: "" };
    return { dataset: dataset || "", evalFn: evalFn || "" };
  }

  function groupSubtitle(group) {
    const fallback = evalConfigIdFallbackMeta(group.evalConfigId);
    const parts = [];
    const dataset = nonPlaceholderText(group.meta.dataset) || fallback.dataset;
    const fnName =
      nonPlaceholderText(group.meta.evalFn || group.meta.judge) || fallback.evalFn;
    if (dataset) parts.push(dataset);
    if (fnName) parts.push(fnName);
    parts.push(`${group.deploymentCount} deployment${group.deploymentCount === 1 ? "" : "s"}`);
    return parts.join(" • ");
  }

  function groupDataset(group) {
    const fallback = evalConfigIdFallbackMeta(group.evalConfigId);
    return nonPlaceholderText(group.meta.dataset) || fallback.dataset || "Unknown";
  }

  function deploymentIdValue(deployment) {
    return safeText(deployment?.deployment_id || deployment?.id).trim();
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

  function deploymentModelValue(deployment) {
    return safeText(
      deployment?.model_name ||
        deployment?.deployment_config?.model?.model_name ||
        deployment?.served_model_name ||
        deployment?.deployment_config?.served_model_name,
    ).trim();
  }

  let deploymentsById = $derived.by(() => {
    const byId = new Map();
    for (const row of deploymentRows || []) {
      const deployment = row?.deployment || {};
      const deploymentId = deploymentIdValue(deployment);
      const appName = safeText(deployment?.app_name).trim();
      if (deploymentId && !byId.has(deploymentId)) byId.set(deploymentId, row);
      if (appName && !byId.has(appName)) byId.set(appName, row);
    }
    return byId;
  });

  let deploymentsByUrl = $derived.by(() => {
    const byUrl = new Map();
    for (const row of deploymentRows || []) {
      const deployment = row?.deployment || {};
      const normalized = normalizePath(deployment?.url);
      if (normalized && !byUrl.has(normalized)) byUrl.set(normalized, row);
    }
    return byUrl;
  });

  let deploymentsByModel = $derived.by(() => {
    const byModel = new Map();
    for (const row of deploymentRows || []) {
      const deployment = row?.deployment || {};
      const model = deploymentModelValue(deployment).toLowerCase();
      if (!model) continue;
      if (!byModel.has(model)) byModel.set(model, []);
      byModel.get(model).push(row);
    }
    for (const rows of byModel.values()) {
      rows.sort(
        (a, b) =>
          toTimestampSeconds(b?.deployment?.created_at) -
          toTimestampSeconds(a?.deployment?.created_at),
      );
    }
    return byModel;
  });

  function findDeploymentRow(run, group) {
    const config = run.eval.config || {};
    const deploymentConfig = config.deployment || {};
    const directKeys = [
      run.eval.deployment_id,
      deploymentConfig.deployment_id,
      deploymentConfig.app_name,
    ];
    for (const key of directKeys) {
      const value = safeText(key).trim();
      if (!value) continue;
      const matched = deploymentsById.get(value);
      if (matched) return matched;
    }
    const evalUrl = normalizePath(deploymentConfig.url || deploymentConfig.endpoint);
    if (evalUrl) {
      const byUrl = deploymentsByUrl.get(evalUrl);
      if (byUrl) return byUrl;
    }
    const modelKey = safeText(
      deploymentConfig.model_name ||
        deploymentConfig.served_model_name ||
        config.model?.model_name ||
        group.meta.model,
    )
      .trim()
      .toLowerCase();
    if (modelKey) {
      const byModel = deploymentsByModel.get(modelKey);
      if (byModel?.length) return byModel[0];
    }
    return null;
  }

  function deploymentRefValue(deploymentRow) {
    const deployment = deploymentRow?.deployment || {};
    return safeText(
      deployment.deployment_id || deployment.app_name || deployment.url || deployment.modal_app_id,
    ).trim();
  }

  function linkedTrainingRunId(deploymentRow) {
    return safeText(
      deploymentRow?.run?.run_id || deploymentRow?.run?.train_result?.training_run_id,
    ).trim();
  }

  function evalBaseModel(run, group, deploymentRow = null) {
    const config = run.eval.config || {};
    if (deploymentRow?.deployment) {
      const deploymentModel = deploymentModelValue(deploymentRow.deployment);
      if (deploymentModel) return deploymentModel;
    }
    const configModel = safeText(
      config.model?.model_name ||
        config.deployment?.model_name ||
        config.deployment?.served_model_name ||
        group.meta.model,
    ).trim();
    if (configModel) return configModel;
    return "[unknown Base Model]";
  }

  function evalDeploymentName(run, deploymentRow = null) {
    if (deploymentRow?.deployment) {
      const deployment = deploymentRow.deployment;
      const deploymentName = safeText(
        deployment.deployment_id || deployment.app_name || deploymentIdValue(deployment),
      ).trim();
      if (deploymentName) return deploymentName;
    }
    const config = run.eval.config || {};
    const deployment = config.deployment || {};
    if (run.eval.deployment_id) return run.eval.deployment_id;
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
    return safeText(run.eval.eval_id).trim() || "—";
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

  let datasetOptions = $derived.by(() =>
    [...new Set(evalConfigGroups.map((group) => groupDataset(group)))].sort((a, b) =>
      a.localeCompare(b),
    ),
  );

  let datasetCounts = $derived.by(
    () =>
      evalConfigGroups.reduce((acc, group) => {
        const dataset = groupDataset(group);
        acc[dataset] = (acc[dataset] || 0) + group.runs.length;
        return acc;
      }, {}),
  );

  let filteredGroups = $derived.by(() => {
    const query = search.trim().toLowerCase();
    return evalConfigGroups
      .map((group) => {
        if (datasetFilter !== "all" && groupDataset(group) !== datasetFilter) {
          return null;
        }
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
            includesText(groupDataset(group), query) ||
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

  function switchDatasetFilter(dataset) {
    datasetFilter = dataset;
    datasetMenuOpen = false;
  }
</script>

<svelte:window
  onclick={() => {
    statusMenuOpen = false;
    datasetMenuOpen = false;
  }}
/>

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
          datasetMenuOpen = false;
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
    <div class="menu-wrap">
      <button
        class="status-filter"
        class:open={datasetMenuOpen}
        onclick={(event) => {
          event.stopPropagation();
          datasetMenuOpen = !datasetMenuOpen;
          statusMenuOpen = false;
        }}
      >
        <Filter size={12} />
        <span>Dataset</span>
        <ChevronDown
          size={12}
          style={`transform: ${datasetMenuOpen ? "rotate(180deg)" : "rotate(0deg)"};`}
        />
      </button>
      {#if datasetMenuOpen}
        <div class="status-menu dataset-menu">
          <button class="status-item" onclick={() => switchDatasetFilter("all")}>
            <span class="dataset-item-label">All datasets</span>
            <span class="status-count">{allEvals.length}</span>
          </button>
          {#each datasetOptions as dataset (dataset)}
            <button class="status-item" onclick={() => switchDatasetFilter(dataset)}>
              <span class="dataset-item-label">{dataset}</span>
              <span class="status-count">{datasetCounts[dataset] || 0}</span>
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
          class="runs-table"
          columns={["Name", "Training run", "Base model", "Status", "Average score", "Examples"]}
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
                <div class="eval-group-title">{group.evalConfigId || group.meta.dataset}</div>
                <div class="eval-group-subtitle">{groupSubtitle(group)}</div>
              </div>
              <div class="eval-group-meta">
                {#if nonPlaceholderText(group.meta.model)}
                  <span class="group-meta-pill">{group.meta.model}</span>
                {/if}
                {#if nonPlaceholderText(group.meta.split)}
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
                <MinimalTable class="runs-table evals-runs-table">
                  <colgroup>
                    <col class="col-name" />
                    <col class="col-training" />
                    <col class="col-model" />
                    <col class="col-status" />
                    <col class="col-score" />
                    <col class="col-examples" />
                  </colgroup>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Training run</th>
                      <th>Base model</th>
                      <th>Status</th>
                      <th>Average score</th>
                      <th>Examples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each group.visibleRuns as run, runIndex (run.eval.eval_id || `${group.evalConfigId}-${run.eval.created_at || 0}-${runIndex}`)}
                      {@const linkedDeployment = findDeploymentRow(run, group)}
                      {@const deploymentName = evalDeploymentName(run, linkedDeployment)}
                      {@const deploymentRef = deploymentRefValue(linkedDeployment)}
                      {@const trainingRunId = linkedTrainingRunId(linkedDeployment)}
                      {@const baseModel = evalBaseModel(run, group, linkedDeployment)}
                      <tr>
                        <td class="mono name-cell" title={deploymentName}>
                          {#if deploymentRef}
                            <button
                              class="cross-link mono truncate-text"
                              onclick={() => onOpenDeployment?.(deploymentRef)}
                            >
                              {deploymentName}
                            </button>
                          {:else}
                            <span class="truncate-text">{deploymentName}</span>
                          {/if}
                        </td>
                        <td class="training-run-cell" title={trainingRunId || "—"}>
                          {#if trainingRunId}
                            <button
                              class="cross-link mono"
                              onclick={() => onOpenTrainingRun?.(trainingRunId)}
                            >
                              {trainingRunId}
                            </button>
                          {:else}
                            —
                          {/if}
                        </td>
                        <td class="base-model-cell" title={baseModel}>
                          <span class="truncate-text">{baseModel}</span>
                        </td>
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

  .dataset-menu {
    width: min(320px, calc(100vw - 2rem));
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

  .dataset-item-label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
    min-width: 1180px;
  }

  :global(table.evals-runs-table) {
    table-layout: fixed;
  }

  :global(table.evals-runs-table col.col-name) {
    width: 27%;
  }

  :global(table.evals-runs-table col.col-training) {
    width: 19%;
  }

  :global(table.evals-runs-table col.col-model) {
    width: 20%;
  }

  :global(table.evals-runs-table col.col-status) {
    width: 14%;
  }

  :global(table.evals-runs-table col.col-score) {
    width: 12%;
  }

  :global(table.evals-runs-table col.col-examples) {
    width: 8%;
  }

  .mono {
    font-family: var(--font-mono);
    color: color-mix(in srgb, var(--accent) 78%, white);
    font-size: 0.72rem;
  }

  .name-cell .truncate-text {
    display: block;
    width: 100%;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .training-run-cell {
    max-width: 0;
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
    max-width: 100%;
  }

  .cross-link:hover {
    color: var(--text-bright);
  }

  .base-model-cell .truncate-text {
    display: block;
    width: 100%;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
