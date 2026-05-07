<script>
  let { logoSrc, navItems, activePage, onNavigate } = $props();
</script>

<aside class="sidebar">
  <div class="brand">
    <img src={logoSrc} alt="Modal" class="logo" />
    <span class="brand-title">Modal Training Gym</span>
  </div>

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
    padding: 1rem 0.7rem;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding: 0.25rem 0.35rem;
  }

  .logo {
    width: 1.2rem;
    height: 1.2rem;
  }

  .brand-title {
    color: var(--text-bright);
    font-size: 0.92rem;
    font-weight: 600;
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
    background: color-mix(in srgb, var(--accent) 16%, transparent);
    border-color: color-mix(in srgb, var(--accent) 42%, transparent);
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
