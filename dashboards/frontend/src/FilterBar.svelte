<script>
  let {
    frameworks,
    fwCounts,
    activeFrameworks,
    allActive,
    statuses,
    statusCounts,
    activeStatuses,
    totalRuns,
    search = $bindable(),
    onToggleFramework,
    onToggleAllFrameworks,
    onToggleStatus,
  } = $props();
</script>

<nav class="filters">
  <div class="search-wrap">
    <span class="search-icon">⌕</span>
    <input
      class="search"
      type="text"
      placeholder="Search by run name, model, or environment..."
      bind:value={search}
    />
  </div>

  <div class="group">
    <span class="filter-label">Framework</span>
    <div class="pill-group">
      <button
        class="pill"
        class:active={allActive}
        onclick={onToggleAllFrameworks}
      >
        All <span class="pill-count">{totalRuns}</span>
      </button>
      {#each frameworks as fw (fw)}
        <button
          class="pill"
          class:active={activeFrameworks.has(fw)}
          onclick={() => onToggleFramework(fw)}
        >
          {fw} <span class="pill-count">{fwCounts[fw] || 0}</span>
        </button>
      {/each}
    </div>
  </div>

  <div class="group">
    <span class="filter-label">Status</span>
    <div class="pill-group">
      {#each statuses as st (st)}
        <button
          class="pill"
          class:active={activeStatuses.has(st)}
          data-status={st}
          onclick={() => onToggleStatus(st)}
        >
          {st} <span class="pill-count">{statusCounts[st] || 0}</span>
        </button>
      {/each}
    </div>
  </div>
</nav>

<style>
  .filters {
    border-bottom: 1px solid var(--border);
    padding: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
    flex-wrap: wrap;
    background: var(--panel);
  }
  .group {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .filter-label {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    white-space: nowrap;
  }
  .pill-group {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.26rem 0.62rem;
    border-radius: 9999px;
    border: 1px solid var(--border);
    background: var(--panel-alt);
    color: var(--muted);
    font: inherit;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.12s ease;
    user-select: none;
  }
  .pill:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }
  .pill.active {
    background: var(--accent-soft);
    border-color: var(--accent-border);
    color: var(--accent);
  }
  .pill :global(.pill-count) {
    font-size: 0.78em;
    opacity: 0.7;
  }
  .pill[data-status="Completed"].active {
    border-color: var(--accent-border);
    color: var(--green);
    background: var(--accent-soft);
  }
  .pill[data-status="Pending"].active {
    border-color: color-mix(in srgb, var(--yellow) 40%, transparent);
    color: var(--yellow);
    background: color-mix(in srgb, var(--yellow) 12%, transparent);
  }
  .search-wrap {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0 0.65rem;
    min-width: min(100%, 360px);
    margin-right: 0.4rem;
  }
  .search-icon {
    color: var(--muted-strong);
    font-size: 0.86rem;
  }
  .search {
    background: transparent;
    border: 0;
    color: var(--text);
    padding: 0.45rem 0;
    font: inherit;
    font-size: 0.8rem;
    width: 100%;
    outline: none;
  }
  .search::placeholder {
    color: var(--muted-strong);
  }
  @media (max-width: 980px) {
    .filters {
      align-items: stretch;
    }
    .group {
      flex-wrap: wrap;
    }
    .search-wrap {
      width: 100%;
      margin-right: 0;
      min-width: 0;
    }
  }
</style>
