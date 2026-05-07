<script>
  let { navItems, activePage, onNavigate } = $props();
</script>

<aside class="sidebar">
  <nav class="nav-list">
    {#each navItems as item (item.key)}
      <a
        href={item.path}
        class="nav-item"
        class:active={activePage === item.key}
        onclick={(event) => {
          event.preventDefault();
          onNavigate(item.key);
        }}
      >
        <span class="nav-icon">
          <item.Icon size={14} strokeWidth={2.1} />
        </span>
        <span>{item.label}</span>
      </a>
    {/each}
  </nav>
</aside>

<style>
  .sidebar {
    border-right: 1px solid var(--border);
    background: var(--bg-depth);
    padding: 0.9rem 0.7rem;
  }

  .nav-list {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .nav-item {
    border: 1px solid transparent;
    border-radius: 8px;
    background: transparent;
    color: var(--muted);
    text-decoration: none;
    text-align: left;
    font: inherit;
    font-size: 0.84rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.56rem;
    cursor: pointer;
    transition: all 0.16s ease;
  }

  .nav-item:hover {
    background: var(--panel);
    color: var(--text-bright);
  }

  .nav-item.active {
    border-radius: 10px;
    background: var(--Primary---Background, rgba(127, 238, 100, 0.1));
    border-color: transparent;
    color: var(--text-bright);
  }

  .nav-icon {
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    color: color-mix(in srgb, var(--muted) 88%, white);
  }

  @media (max-width: 900px) {
    .sidebar {
      border-right: 0;
      border-bottom: 1px solid var(--border);
    }
  }
</style>
