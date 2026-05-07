<script>
  import { Book } from "lucide-svelte";

  let { title, statusText, docsUrl, onRefresh } = $props();
</script>

<div class="topbar">
  <a class="docs-button" href={docsUrl}>
    <Book size={14} strokeWidth={2.1} />
    <span>Docs</span>
  </a>
</div>

<header class="workspace-header">
  <h1>{title}</h1>
  <div class="workspace-actions">
    <span class="status-text">{statusText}</span>
    <button class="btn" onclick={onRefresh}>Refresh</button>
  </div>
</header>

<style>
  .topbar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.65rem;
  }

  .docs-button {
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--muted);
    background: var(--panel);
    text-decoration: none;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.34rem 0.66rem;
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
  }

  .docs-button:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

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
