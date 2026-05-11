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
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    background: var(--bg-depth);
    padding: 4px 20px 0;
  }

  .nav-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 12px;
  }

  .nav-item {
    border: 0;
    border-radius: 10px;
    background: transparent;
    color: var(--text);
    text-decoration: none;
    text-align: left;
    font: inherit;
    font-size: 14px;
    font-weight: 500;
    line-height: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    cursor: pointer;
    transition: all 0.16s ease;
  }

  .nav-item:hover {
    background: var(--panel);
    color: var(--text-bright);
  }

  .nav-item.active {
    border-radius: 10px;
    background: rgba(127, 238, 100, 0.1);
    color: var(--text-bright);
  }

  .nav-icon {
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    opacity: 0.6;
  }

  @media (max-width: 900px) {
    .sidebar {
      border-right: 0;
      border-bottom: 1px solid var(--border);
    }
  }
</style>
