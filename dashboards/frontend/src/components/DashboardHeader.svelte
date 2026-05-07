<script>
  let { title, statusText, onRefresh } = $props();
</script>

<header class="workspace-header">
  <h1>{title}</h1>
  <div class="workspace-actions">
    <span class="status-text">{statusText}</span>
    <button class="btn" onclick={onRefresh}>Refresh</button>
  </div>
</header>

<style>
  .workspace-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  h1 {
    color: var(--text-bright);
    font-size: 1.85rem;
    font-weight: 600;
    letter-spacing: -0.02em;
  }

  .workspace-actions {
    display: flex;
    align-items: center;
    gap: 0.7rem;
  }

  .status-text {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: lowercase;
  }

  .btn {
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    background: var(--panel);
    font: inherit;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.36rem 0.72rem;
    cursor: pointer;
  }

  .btn:hover {
    border-color: var(--border-strong);
    color: var(--text-bright);
  }
</style>
