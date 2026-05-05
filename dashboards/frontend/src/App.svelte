<script>
  import "./app.css";
  import { fetchRuns, fetchEvals, fetchDeployments } from "./lib/api.js";
  import FilterBar from "./FilterBar.svelte";
  import FrameworkSection from "./FrameworkSection.svelte";
  import logoSvg from "./lib/logo.svg";

  import { fmtDate, truncateId } from "./lib/format.js";

  let allRuns = $state([]);
  let allEvals = $state([]);
  let allDeployments = $state([]);
  let loading = $state(true);
  let error = $state(null);
  let search = $state("");
  let activeFrameworks = $state(new Set());
  let activeStatuses = $state(new Set());
  let activePage = $state("training");

  const pageMeta = {
    training: {
      title: "Training",
      subtitle: "Hosted RL training on environments",
    },
    deployments: {
      title: "Deployments",
      subtitle: "Manage LoRA adapter deployments",
    },
    evals: {
      title: "Evals",
      subtitle: "Model eval runs and reports",
    },
  };

  const navSections = [
    {
      label: "Overview",
      items: [
        { key: "training", label: "Training" },
        { key: "deployments", label: "Deployments" },
        { key: "evals", label: "Evals" },
      ],
    },
  ];

  function getFramework(run) {
    return run.framework || "(untagged)";
  }

  function getStatus(run) {
    return run.train_result ? "Completed" : "Pending";
  }

  async function load() {
    loading = true;
    error = null;
    try {
      const [runs, evals, deployments] = await Promise.all([
        fetchRuns(),
        fetchEvals(),
        fetchDeployments(),
      ]);
      allRuns = runs;
      allEvals = evals;
      allDeployments = deployments;
      activeFrameworks = new Set(allRuns.map(getFramework));
      activeStatuses = new Set(allRuns.map(getStatus));
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  load();

  let frameworks = $derived(
    [...new Set(allRuns.map(getFramework))].sort(),
  );

  let statuses = $derived([...new Set(allRuns.map(getStatus))].sort());

  let fwCounts = $derived(
    allRuns.reduce((acc, r) => {
      const fw = getFramework(r);
      acc[fw] = (acc[fw] || 0) + 1;
      return acc;
    }, {}),
  );

  let statusCounts = $derived(
    allRuns.reduce((acc, r) => {
      const st = getStatus(r);
      acc[st] = (acc[st] || 0) + 1;
      return acc;
    }, {}),
  );

  let filteredRuns = $derived(
    allRuns
      .filter((r) => {
        if (!activeFrameworks.has(getFramework(r))) return false;
        if (!activeStatuses.has(getStatus(r))) return false;
        if (search) {
          const q = search.toLowerCase();
          if (
            !r.run_id?.toLowerCase().includes(q) &&
            !r.modal_app_id?.toLowerCase().includes(q) &&
            !r.config_summary?.model_name?.toLowerCase().includes(q)
          )
            return false;
        }
        return true;
      })
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );

  let groupedRuns = $derived.by(() => {
    const groups = {};
    for (const r of filteredRuns) {
      const fw = getFramework(r);
      (groups[fw] ||= []).push(r);
    }
    return groups;
  });

  let sortedFrameworkKeys = $derived(Object.keys(groupedRuns).sort());
  let completedTotal = $derived(allRuns.filter((run) => run.train_result).length);
  let pendingTotal = $derived(allRuns.length - completedTotal);
  let completedRuns = $derived(
    allRuns
      .filter((run) => run.train_result)
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );

  function evalAccuracy(ev) {
    const rows = ev.rows || [];
    if (!rows.length) return 0;
    return rows.reduce((s, r) => s + (r.score || 0), 0) / rows.length;
  }

  let sortedEvals = $derived(
    [...allEvals].sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );
  let avgAccuracy = $derived(
    allEvals.length
      ? allEvals.reduce((s, e) => s + evalAccuracy(e), 0) / allEvals.length
      : 0,
  );
  let distinctDatasets = $derived(
    new Set(allEvals.map((e) => e.config?.dataset?.name || "—")).size,
  );

  let statusText = $derived.by(() => {
    if (loading) return "loading...";
    if (error) return "error";
    if (activePage === "evals")
      return `${allEvals.length} eval${allEvals.length === 1 ? "" : "s"}`;
    if (!allRuns.length) return "0 runs";
    if (activePage === "deployments")
      return `${completedRuns.length} deployment-ready run${completedRuns.length === 1 ? "" : "s"}`;
    if (activePage !== "training")
      return `${allRuns.length} total run${allRuns.length === 1 ? "" : "s"}`;
    const shown = filteredRuns.length;
    const total = allRuns.length;
    const fwCount = sortedFrameworkKeys.length;
    const fwLabel = fwCount === 1 ? "framework" : "frameworks";
    if (shown === total)
      return `${total} run${total === 1 ? "" : "s"} across ${fwCount} ${fwLabel}`;
    return `${shown} of ${total} runs across ${fwCount} ${fwLabel}`;
  });

  function toggleFramework(fw) {
    const next = new Set(activeFrameworks);
    if (next.has(fw)) next.delete(fw);
    else next.add(fw);
    activeFrameworks = next;
  }

  function toggleAllFrameworks() {
    if (activeFrameworks.size === frameworks.length) {
      activeFrameworks = new Set();
    } else {
      activeFrameworks = new Set(frameworks);
    }
  }

  function toggleStatus(st) {
    const next = new Set(activeStatuses);
    if (next.has(st)) next.delete(st);
    else next.add(st);
    activeStatuses = next;
  }

  function setActivePage(page) {
    activePage = page;
  }
</script>

<div class="shell">
  <aside class="sidebar">
    <div class="brand">
      <img src={logoSvg} alt="Modal" class="logo" />
      <div>
        <div class="brand-title">training-gym</div>
        <div class="brand-subtitle">Distributed training and evaluation</div>
      </div>
    </div>

    {#each navSections as section (section.label)}
      <div class="nav-group">
        <div class="nav-label">{section.label}</div>
        {#each section.items as item (item.key)}
          <button
            class="nav-item"
            class:active={activePage === item.key}
            onclick={() => setActivePage(item.key)}
          >
            {item.label}
          </button>
        {/each}
      </div>
    {/each}
  </aside>

  <main class="workspace">
    <header class="workspace-header">
      <div>
        <h1>{pageMeta[activePage].title}</h1>
        <p>{pageMeta[activePage].subtitle}</p>
      </div>
      <div class="workspace-actions">
        <span class="status-text">{statusText}</span>
        <button class="btn" onclick={load}>Refresh</button>
      </div>
    </header>

    {#if activePage === "training"}
      <section class="summary-row">
        <article class="summary-card">
          <span class="summary-label">Total runs</span>
          <strong>{allRuns.length}</strong>
        </article>
        <article class="summary-card summary-green">
          <span class="summary-label">Completed</span>
          <strong>{completedTotal}</strong>
        </article>
        <article class="summary-card summary-amber">
          <span class="summary-label">Pending</span>
          <strong>{pendingTotal}</strong>
        </article>
        <article class="summary-card">
          <span class="summary-label">Frameworks</span>
          <strong>{frameworks.length}</strong>
        </article>
      </section>

      <section class="runs-surface">
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
          onToggleFramework={toggleFramework}
          onToggleAllFrameworks={toggleAllFrameworks}
          onToggleStatus={toggleStatus}
        />

        <div class="runs-body">
          {#if loading}
            <div class="empty">Loading runs...</div>
          {:else if error}
            <div class="empty">Failed to load: {error}</div>
          {:else if !allRuns.length}
            <div class="empty">
              No training runs found. The cron job populates data every 5 minutes —
              if you just deployed, wait for the first refresh.
            </div>
          {:else if !filteredRuns.length}
            <div class="empty">No runs match the current filters.</div>
          {:else}
            {#each sortedFrameworkKeys as fw (fw)}
              <FrameworkSection framework={fw} runs={groupedRuns[fw]} deployments={allDeployments} />
            {/each}
          {/if}
        </div>
      </section>
    {:else if activePage === "deployments"}
      <section class="summary-row">
        <article class="summary-card">
          <span class="summary-label">Deployment-ready</span>
          <strong>{completedRuns.length}</strong>
        </article>
        <article class="summary-card summary-amber">
          <span class="summary-label">Waiting for result</span>
          <strong>{pendingTotal}</strong>
        </article>
        <article class="summary-card">
          <span class="summary-label">Distinct models</span>
          <strong>{new Set(completedRuns.map((run) => run.config_summary?.model_name || "—")).size}</strong>
        </article>
        <article class="summary-card">
          <span class="summary-label">Frameworks</span>
          <strong>{frameworks.length}</strong>
        </article>
      </section>

      <section class="runs-surface">
        <div class="runs-body">
          {#if loading}
            <div class="empty">Loading deployments...</div>
          {:else if error}
            <div class="empty">Failed to load: {error}</div>
          {:else if !completedRuns.length}
            <div class="empty">No completed runs yet. Deployments appear after training completes.</div>
          {:else}
            <div class="deployments-table-wrap">
              <table class="deployments-table">
                <thead>
                  <tr>
                    <th>Training run</th>
                    <th>Base model</th>
                    <th>Status</th>
                    <th>Cluster</th>
                    <th>Checkpoint</th>
                    <th>Links</th>
                  </tr>
                </thead>
                <tbody>
                  {#each completedRuns as run (run.run_id)}
                    <tr>
                      <td class="mono">
                        <button class="cross-link" onclick={() => { search = run.run_id; activePage = "training"; }}>
                          {truncateId(run.run_id)}
                        </button>
                      </td>
                      <td>{run.config_summary?.model_name || "—"}</td>
                      <td><span class="deploy-status">Ready</span></td>
                      <td>{run.config_summary?.gpu_type || "—"}</td>
                      <td class="mono">{run.train_result?.checkpoint_dir || "—"}</td>
                      <td class="links-cell">
                        {#if run.modal_app_url}
                          <a class="pill-link pill-modal" href={run.modal_app_url} target="_blank" rel="noopener noreferrer">Modal</a>
                        {/if}
                        {#if run.train_result?.wandb_url}
                          <a class="pill-link pill-wandb" href={run.train_result.wandb_url} target="_blank" rel="noopener noreferrer">W&B</a>
                        {/if}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </div>
      </section>
    {:else if activePage === "evals"}
      <section class="summary-row">
        <article class="summary-card">
          <span class="summary-label">Total evals</span>
          <strong>{allEvals.length}</strong>
        </article>
        <article class="summary-card summary-green">
          <span class="summary-label">Avg accuracy</span>
          <strong>{allEvals.length ? (avgAccuracy * 100).toFixed(1) + "%" : "—"}</strong>
        </article>
        <article class="summary-card">
          <span class="summary-label">Datasets</span>
          <strong>{distinctDatasets}</strong>
        </article>
        <article class="summary-card">
          <span class="summary-label">Total examples</span>
          <strong>{allEvals.reduce((s, e) => s + (e.rows || []).length, 0)}</strong>
        </article>
      </section>

      <section class="runs-surface">
        <div class="runs-body">
          {#if loading}
            <div class="empty">Loading evals...</div>
          {:else if error}
            <div class="empty">Failed to load: {error}</div>
          {:else if !allEvals.length}
            <div class="empty">No eval results yet. Run an evaluation to see results here.</div>
          {:else}
            <div class="deployments-table-wrap">
              <table class="deployments-table">
                <thead>
                  <tr>
                    <th>Eval ID</th>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Score</th>
                    <th>Created</th>
                    <th>Links</th>
                  </tr>
                </thead>
                <tbody>
                  {#each sortedEvals as ev (ev.eval_id)}
                    {@const acc = evalAccuracy(ev)}
                    {@const total = (ev.rows || []).length}
                    {@const cfg = ev.config || {}}
                    <tr>
                      <td class="mono">{ev.eval_id}</td>
                      <td>{cfg.dataset?.name || "—"}</td>
                      <td>{cfg.deployment?.model_name || "—"}</td>
                      <td>
                        <span class="eval-accuracy" class:eval-high={acc >= 0.7} class:eval-mid={acc >= 0.4 && acc < 0.7} class:eval-low={acc < 0.4}>
                          {(acc * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td>{(acc * total).toFixed(1)}/{total}</td>
                      <td>{fmtDate(ev.created_at)}</td>
                      <td class="links-cell">
                        {#if cfg.deployment?.url}
                          <a class="pill-link pill-modal" href={cfg.deployment.url} target="_blank" rel="noopener noreferrer">Endpoint</a>
                        {/if}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </div>
      </section>
    {/if}
  </main>
</div>

<style>
  .shell {
    display: grid;
    grid-template-columns: 250px minmax(0, 1fr);
    min-height: 100vh;
  }
  .sidebar {
    border-right: 1px solid var(--border);
    background: var(--panel);
    padding: 1rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.6rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel-alt);
  }
  .logo {
    width: 1.3rem;
    height: 1.3rem;
  }
  .brand-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-bright);
    line-height: 1.2;
  }
  .brand-subtitle {
    font-size: 0.72rem;
    color: var(--muted);
  }
  .nav-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .nav-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-weight: 600;
    padding: 0.4rem 0.6rem;
  }
  .nav-item {
    border: 1px solid transparent;
    background: transparent;
    color: var(--muted);
    text-align: left;
    font: inherit;
    font-size: 0.84rem;
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    cursor: pointer;
    transition: all 120ms ease;
  }
  .nav-item:hover {
    background: var(--panel-alt);
    color: var(--text-bright);
  }
  .nav-item.active {
    color: var(--text-bright);
    border-color: var(--accent-border);
    background: var(--accent-soft);
  }
  .workspace {
    padding: 1.25rem 1.5rem 2rem;
    background:
      radial-gradient(880px 420px at 14% -20%, color-mix(in srgb, var(--accent) 10%, transparent), transparent 70%),
      var(--bg);
  }
  .workspace-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  h1 {
    font-size: 1.25rem;
    line-height: 1.1;
    color: var(--text-bright);
    margin-bottom: 0.2rem;
  }
  p {
    font-size: 0.82rem;
    color: var(--muted);
  }
  .workspace-actions {
    display: flex;
    align-items: center;
    gap: 0.7rem;
  }
  .btn {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid var(--accent-border);
    padding: 0.36rem 0.82rem;
    border-radius: 7px;
    cursor: pointer;
    font: inherit;
    font-size: 0.8rem;
    font-weight: 500;
    transition:
      border-color 0.15s ease,
      background-color 0.15s ease,
      color 0.15s ease;
  }
  .btn:hover {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
    border-color: color-mix(in srgb, var(--accent) 55%, transparent);
    color: var(--text-bright);
  }
  .status-text {
    color: var(--muted);
    font-size: 0.78rem;
    white-space: nowrap;
  }
  .summary-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .summary-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
    padding: 0.65rem 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .summary-label {
    font-size: 0.78rem;
    color: var(--muted);
  }
  .summary-card strong {
    font-size: 0.95rem;
    color: var(--text-bright);
    font-weight: 600;
  }
  .summary-green strong {
    color: var(--green);
  }
  .summary-amber strong {
    color: var(--yellow);
  }
  .runs-surface {
    border: 1px solid var(--border);
    border-radius: 11px;
    background: var(--panel);
    overflow: hidden;
  }
  .runs-body {
    padding: 0.8rem;
  }
  .deployments-table-wrap {
    overflow-x: auto;
  }
  .deployments-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 860px;
  }
  .deployments-table th {
    background: var(--panel-alt);
    color: var(--muted-strong);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    text-align: left;
    border-bottom: 1px solid var(--border);
    padding: 0.45rem 0.8rem;
  }
  .deployments-table td {
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 0.8rem;
    font-size: 0.78rem;
    color: var(--text);
    max-width: 300px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
  }
  .deployments-table tr:hover td {
    background: color-mix(in srgb, var(--text) 4%, transparent);
  }
  .mono {
    font-family: var(--font-mono);
    color: color-mix(in srgb, var(--accent) 80%, white);
  }
  .deploy-status {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.12rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.71rem;
    font-weight: 500;
    border: 1px solid var(--accent-border);
    background: var(--accent-soft);
    color: var(--green);
  }
  .deploy-status::before {
    content: "";
    width: 0.38rem;
    height: 0.38rem;
    border-radius: 9999px;
    background: currentColor;
  }
  .eval-accuracy {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.73rem;
    font-weight: 600;
  }
  .eval-high {
    border: 1px solid var(--accent-border);
    background: var(--accent-soft);
    color: var(--green);
  }
  .eval-mid {
    border: 1px solid color-mix(in srgb, var(--yellow) 45%, transparent);
    background: color-mix(in srgb, var(--yellow) 10%, transparent);
    color: var(--yellow);
  }
  .eval-low {
    border: 1px solid color-mix(in srgb, var(--red) 40%, transparent);
    background: color-mix(in srgb, var(--red) 10%, transparent);
    color: var(--red);
  }
  .links-cell {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }
  .pill-link {
    display: inline-block;
    padding: 0.14rem 0.48rem;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 500;
    text-decoration: none;
    white-space: nowrap;
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
  .cross-link {
    background: none;
    border: none;
    color: color-mix(in srgb, var(--accent) 80%, white);
    font-family: var(--font-mono);
    font-size: inherit;
    padding: 0;
    cursor: pointer;
    text-decoration: none;
  }
  .cross-link:hover {
    color: var(--accent);
    text-decoration: underline;
  }
  .empty {
    color: var(--muted);
    padding: 3rem 2rem;
    text-align: center;
    font-size: 0.85rem;
  }
  @media (max-width: 1120px) {
    .summary-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 900px) {
    .shell {
      grid-template-columns: 1fr;
    }
    .sidebar {
      border-right: 0;
      border-bottom: 1px solid var(--border);
    }
    .workspace {
      padding: 1rem;
    }
  }
</style>
