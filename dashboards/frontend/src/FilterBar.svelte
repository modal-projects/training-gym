<script>
  let {
    frameworks,
    fwCounts,
    activeFrameworks,
    allActive,
    states,
    stateCounts,
    activeStates,
    totalRuns,
    search = $bindable(),
    onToggleFramework,
    onToggleAllFrameworks,
    onToggleState,
  } = $props();
</script>

<nav class="filters">
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

  <div class="filter-sep"></div>
  <span class="filter-label">State</span>
  <div class="pill-group">
    {#each states as st (st)}
      <button
        class="pill"
        class:active={activeStates.has(st)}
        data-state={st}
        onclick={() => onToggleState(st)}
      >
        {st} <span class="pill-count">{stateCounts[st] || 0}</span>
      </button>
    {/each}
  </div>

  <div class="filter-sep"></div>
  <input
    class="search"
    type="text"
    placeholder="Search by name or app id..."
    bind:value={search}
  />
</nav>

<style>
  .filters {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    position: sticky;
    top: 60px;
    z-index: 9;
  }
  .filter-label {
    color: var(--muted);
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    white-space: nowrap;
  }
  .pill-group {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border-radius: 9999px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font: inherit;
    font-size: 0.85em;
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }
  .pill:hover {
    border-color: var(--accent);
    color: var(--text);
  }
  .pill.active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
  }
  .pill :global(.pill-count) {
    font-size: 0.8em;
    opacity: 0.7;
  }
  .pill[data-state="Running"].active,
  .pill[data-state="Deployed"].active {
    border-color: var(--green);
    color: var(--green);
    background: rgba(74, 222, 128, 0.1);
  }
  .pill[data-state="Stopped"].active {
    border-color: var(--muted);
  }
  .pill[data-state="Ephemeral"].active {
    border-color: var(--yellow);
    color: var(--yellow);
    background: rgba(251, 191, 36, 0.1);
  }
  .filter-sep {
    width: 1px;
    height: 1.2rem;
    background: var(--border);
    margin: 0 0.25rem;
  }
  .search {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 0.35rem 0.7rem;
    font: inherit;
    font-size: 0.85em;
    width: 14rem;
    outline: none;
  }
  .search:focus {
    border-color: var(--accent);
  }
  .search::placeholder {
    color: var(--muted);
  }
</style>
