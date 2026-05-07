<script>
  import { Check, ChevronDown, Filter } from "lucide-svelte";

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

  let openMenu = $state(null);

  function toggleMenu(menu) {
    openMenu = openMenu === menu ? null : menu;
  }
</script>

<svelte:window onclick={() => (openMenu = null)} />

<nav class="filters">
  <div class="menu-wrap">
    <button
      class="filter-button"
      class:open={openMenu === "status"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("status");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Status</span>
      <span class="chevron" class:rotated={openMenu === "status"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "status"}
      <div class="menu">
        {#each statuses as st (st)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleStatus(st);
            }}
          >
            <span class="checkmark" class:checked={activeStatuses.has(st)}>
              {#if activeStatuses.has(st)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{st}</span>
            <span class="item-count">{statusCounts[st] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="menu-wrap">
    <button
      class="filter-button"
      class:open={openMenu === "frameworks"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("frameworks");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Frameworks</span>
      <span class="chevron" class:rotated={openMenu === "frameworks"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "frameworks"}
      <div class="menu">
        <button
          class="menu-item"
          onclick={(event) => {
            event.stopPropagation();
            onToggleAllFrameworks();
          }}
        >
          <span class="checkmark" class:checked={allActive}>
            {#if allActive}
              <Check size={11} />
            {/if}
          </span>
          <span class="item-label">All</span>
          <span class="item-count">{totalRuns}</span>
        </button>
        {#each frameworks as fw (fw)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleFramework(fw);
            }}
          >
            <span class="checkmark" class:checked={activeFrameworks.has(fw)}>
              {#if activeFrameworks.has(fw)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{fw}</span>
            <span class="item-count">{fwCounts[fw] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
</nav>

<style>
  .filters {
    padding: 0;
    display: flex;
    align-items: center;
    gap: 0.42rem;
    position: relative;
  }

  .menu-wrap {
    position: relative;
  }

  .filter-button {
    display: inline-flex;
    align-items: center;
    gap: 0.34rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: color-mix(in srgb, var(--panel-alt) 88%, black);
    color: var(--text);
    font: inherit;
    font-size: 0.74rem;
    padding: 0.24rem 0.52rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .filter-button:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .filter-button.open {
    border-color: color-mix(in srgb, var(--accent) 42%, transparent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    color: var(--text-bright);
  }

  .button-icon {
    color: var(--muted);
  }

  .filter-button.open .button-icon {
    color: var(--accent);
  }

  .chevron {
    color: var(--muted);
    transition: transform 0.14s ease;
  }

  .chevron.rotated {
    transform: rotate(180deg);
  }

  .menu {
    position: absolute;
    top: calc(100% + 0.3rem);
    left: 0;
    z-index: 20;
    width: max-content;
    min-width: 210px;
    max-height: 280px;
    overflow: auto;
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    background: color-mix(in srgb, var(--panel) 94%, black);
    box-shadow: 0 12px 26px color-mix(in srgb, black 55%, transparent);
    padding: 0.25rem;
  }

  .menu-item {
    width: 100%;
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.42rem;
    border: 0;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 0.73rem;
    text-align: left;
    padding: 0.3rem 0.38rem;
    border-radius: 7px;
    cursor: pointer;
  }

  .menu-item:hover {
    background: color-mix(in srgb, var(--text-bright) 6%, transparent);
    color: var(--text-bright);
  }

  .checkmark {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    background: color-mix(in srgb, var(--panel-alt) 90%, black);
  }

  .checkmark.checked {
    border-color: color-mix(in srgb, var(--accent) 40%, transparent);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
  }

  .item-label {
    color: inherit;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .item-count {
    color: var(--muted);
    font-size: 0.67rem;
    font-variant-numeric: tabular-nums;
  }

  @media (max-width: 980px) {
    .menu {
      min-width: 180px;
    }
  }
</style>
