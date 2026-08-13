<script>
  import {
    Check,
    ChevronDown,
    ExternalLink,
    Filter,
    PanelRightClose,
    Search,
  } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBulkActions from "../components/FilterBulkActions.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import ResizableTable from "../components/ResizableTable.svelte";
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
  let activeStatuses = $state(new Set(["Ready", "Pending", "Inactive"]));
  let statusMenuOpen = $state(false);
  let selectedDeploymentKey = $state(null);
  let relatedRunExpanded = $state(true);
  let relatedEvalsExpanded = $state(true);
  const deploymentColumns = [
    { key: "name", label: "Name", width: 220, minWidth: 140 },
    { key: "training", label: "Training run name", width: 280, minWidth: 160 },
    { key: "status", label: "Status", width: 116, minWidth: 96 },
    { key: "model", label: "Base model", width: 220, minWidth: 140 },
    { key: "created", label: "Created", width: 116, minWidth: 96 },
    { key: "actions", label: "", ariaLabel: "Actions", width: 150, minWidth: 132 },
  ];

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
    if (row.run && getStatus(row.run) === "failed") return "Inactive";
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
  let allStatusesActive = $derived(activeStatuses.size === statusFilters.length);

  let readyDeployments = $derived(statusCounts.Ready || 0);
  let pendingDeployments = $derived(statusCounts.Pending || 0);

  let filteredRows = $derived.by(() => {
    const query = search.trim().toLowerCase();
    return enrichedRows.filter((row) => {
      if (!activeStatuses.has(row.status)) return false;
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

  function toggleStatusFilter(status) {
    const next = new Set(activeStatuses);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    activeStatuses = next;
  }

  function selectAllStatuses() {
    activeStatuses = new Set(statusFilters);
  }

  function clearStatuses() {
    activeStatuses = new Set();
  }
</script>

<svelte:window onclick={() => (statusMenuOpen = false)} />

<section class="summary-sticky grid [grid-template-columns:repeat(3,minmax(0,1fr))] gap-[14px] p-[0_24px] mb-[24px] max-[1080px]:[grid-template-columns:repeat(2,minmax(0,1fr))]">
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

<section class="[border:0] [background:transparent] grid [grid-template-columns:minmax(0,1fr)] min-h-[520px] p-[0_24px_16px] max-[900px]:pb-[24px]">
  <div class="min-w-0">
    <div class="[border-bottom:0] p-[0_0_24px] flex items-center gap-[0.4rem] max-[900px]:flex-col max-[900px]:items-stretch">
      <label class="inline-flex items-center gap-[8px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] [background:transparent] w-[260px] p-[6px_8px] max-[900px]:w-full" aria-label="Search deployments">
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
      <div class="relative">
        <button
          class="deploy-status-filter"
          class:deploy-open={statusMenuOpen}
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
            <FilterBulkActions
              allSelected={allStatusesActive}
              noneSelected={activeStatuses.size === 0}
              onSelectAll={selectAllStatuses}
              onDeselectAll={clearStatuses}
            />
            {#each statusFilters as status (status)}
              <button
                class="status-item"
                onclick={(event) => {
                  event.stopPropagation();
                  toggleStatusFilter(status);
                }}
              >
                <span class="checkmark" class:checked={activeStatuses.has(status)}>
                  {#if activeStatuses.has(status)}
                    <Check size={11} />
                  {/if}
                </span>
                <span class="item-label">{status}</span>
                <span class="status-count">{statusCounts[status] || 0}</span>
              </button>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <div class="p-0">
    {#if loading}
      <div class="table-wrap freeze-header">
        <MinimalTableSkeleton
          class="deployments-table"
          columns={["Name", "Training run name", "Status", "Base model", "Created", ""]}
          rows={6}
        />
      </div>
    {:else if error}
      <div class="page-empty">Failed to load: {error}</div>
    {:else if !allDeployments.length}
      <div class="page-empty">No deployments recorded yet.</div>
    {:else}
      <div class="table-wrap freeze-header">
        <ResizableTable class="deployments-table" columns={deploymentColumns} stickyFirstColumn>
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
                <td class="min-w-0">
                  <div class="text-(--text-bright) [font-family:var(--font-mono)] text-[0.73rem] whitespace-nowrap overflow-hidden text-ellipsis" title={deploymentName}>{deploymentName}</div>
                  {#if deploymentAppName}
                    <div class="text-(--muted-strong) text-[0.7rem] mt-[0.1rem] whitespace-nowrap overflow-hidden text-ellipsis" title={deploymentAppName}>{deploymentAppName}</div>
                  {/if}
                </td>
                <td class="min-w-0">
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
                <td class="max-w-0 overflow-hidden text-ellipsis whitespace-nowrap" title={row.baseModel}>
                  {row.baseModel}
                </td>
                <td class="created-cell">
                  <TimeAgo timestamp={row.deployment.created_at || row.run?.created_at} showJustNow falsyRepresentation="—" />
                </td>
                <td class="min-w-0">
                  {#if row.deployment.modal_app_url || row.run?.modal_app_url}
                    <a
                      class="deploy-open-modal-link ghost-hover"
                      href={row.deployment.modal_app_url || row.run?.modal_app_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onclick={(event) => event.stopPropagation()}
                    >
                      <span>Open in Modal</span>
                      <ExternalLink size={12} />
                    </a>
                  {:else}
                    <span class="deploy-open-modal-link ghost-hover open-modal-link-disabled">
                      <span>Open in Modal</span>
                      <ExternalLink size={12} />
                    </span>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </ResizableTable>
      </div>
    {/if}
    </div>
  </div>

  {#if selectedDeployment}
    <Drawer open={!!selectedDeployment} onclose={closeDetails}>
      <div class="w-[min(420px,calc(100vw_-_24px))] h-full">
      <div class="drawer-panel-header">
        <div>
          <div class="drawer-panel-eyebrow">Deployment</div>
          <h2 class="text-(--text-bright) text-[16px] font-medium [font-family:var(--font-mono)] leading-[24px] overflow-hidden text-ellipsis">{selectedDeployment.deployment.deployment_id || deploymentLabel(selectedDeployment.deployment)}</h2>
        </div>
        <div class="flex items-center gap-[8px]">
          {#if selectedDeployment.deployment.modal_app_url || selectedDeployment.run?.modal_app_url}
            <a
              class="deploy-open-modal-link ghost-hover"
              href={selectedDeployment.deployment.modal_app_url || selectedDeployment.run?.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>Open in Modal</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="drawer-panel-close ghost-hover" onclick={closeDetails} aria-label="Close deployment details">
            <PanelRightClose size={15} />
          </button>
        </div>
      </div>

      <section class="p-[16px_20px] [border-bottom:1px_solid_var(--color-c-gray-10,#2f2f2f)]">
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
            class="meta-value [overflow-wrap:normal]! overflow-hidden text-ellipsis whitespace-nowrap"
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
          <span class="meta-value [font-family:var(--font-mono)] text-[12px]! leading-[16px]!">
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
            <ChevronDown
              size={13}
              style={`transform: ${relatedRunExpanded ? "rotate(180deg)" : "rotate(0deg)"};`}
            />
          </button>
          {#if relatedRunExpanded}
            <button
              class="w-full mt-[8px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] [background:transparent] text-(--text) p-[8px_12px] flex items-center gap-[8px] cursor-pointer text-left ghost-hover"
              onclick={() => onOpenTrainingRun(selectedDeployment.run.run_id)}
            >
              <StatusPill status={getStatus(selectedDeployment.run)} />
              <span class="deploy-mono min-w-0 overflow-hidden text-ellipsis whitespace-nowrap" title={selectedDeployment.run.run_id}>
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
            <ChevronDown
              size={13}
              style={`transform: ${relatedEvalsExpanded ? "rotate(180deg)" : "rotate(0deg)"};`}
            />
          </button>
          {#if relatedEvalsExpanded}
            <div class="mt-[8px] flex flex-col gap-[4px]">
              {#each selectedDeploymentEvals as evalRun, evalIndex (`${evalRun.dataset}-${evalRun.evalId || "eval"}-${evalRun.createdAt || 0}-${evalIndex}`)}
                <div class="[border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[6px_8px] flex items-center justify-between gap-[8px] [background:rgba(255,255,255,0.03)]">
                  <div class="min-w-0 flex items-center gap-[6px] overflow-hidden">
                    <span class="rounded-[4px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] [background:transparent] text-(--muted) p-[2px_6px] text-[11px] leading-[14px] whitespace-nowrap">{evalRun.dataset}</span>
                    <span class="deploy-mono">{truncateId(evalRun.evalId)}</span>
                  </div>
                  <span class="text-(--yellow) text-[12px] leading-[16px] [font-variant-numeric:tabular-nums]">{(evalRun.score * 100).toFixed(1)}%</span>
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

