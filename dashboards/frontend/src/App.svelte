<script>
  import "./app.css";
  import { fetchRuns } from "./lib/api.js";
  import FilterBar from "./FilterBar.svelte";
  import FrameworkSection from "./FrameworkSection.svelte";
  import logoSvg from "./lib/logo.svg";

  let allRuns = $state([]);
  let loading = $state(true);
  let error = $state(null);
  let search = $state("");
  let activeFrameworks = $state(new Set());
  let activeStatuses = $state(new Set());

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
      allRuns = await fetchRuns();
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
    allRuns.filter((r) => {
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
    }),
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

  let statusText = $derived.by(() => {
    if (loading) return "loading...";
    if (error) return "error";
    if (!allRuns.length) return "0 runs";
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
</script>

  <header class="topbar">
    <img src={logoSvg} alt="Modal" class="logo" />
    <h1>training-gym</h1>
    <div class="sep"></div>
    <button class="btn" onclick={load}>Refresh</button>
    <span class="status">{statusText}</span>
  </header>

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

  <div class="main">
    {#if loading}
      <div class="empty">Loading...</div>
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
        <FrameworkSection framework={fw} runs={groupedRuns[fw]} />
      {/each}
    {/if}
  </div>

<style>
  .topbar {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .logo {
    height: 1.4rem;
    width: auto;
  }
  .topbar h1 {
    font-size: 1.25rem;
    font-weight: 700;
    white-space: nowrap;
  }
  .sep {
    width: 1px;
    height: 1.5rem;
    background: var(--border);
  }
  .btn {
    background: #1f2937;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 0.85em;
    transition: background 0.15s;
  }
  .btn:hover {
    background: #374151;
  }
  .status {
    color: var(--muted);
    font-size: 0.85em;
    margin-left: auto;
  }
  .main {
    padding: 1.5rem 2rem 3rem;
  }
  .empty {
    color: var(--muted);
    padding: 3rem;
    text-align: center;
    font-size: 0.95em;
  }
</style>
