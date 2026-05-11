<script>
  import { RefreshCw } from "lucide-svelte";

  let { title, statusText, onRefresh } = $props();
</script>

<header class="workspace-header">
  <h1>{title}</h1>
  <div class="workspace-actions">
    {#if statusText}
      <span class="status-text">{statusText}</span>
    {/if}
    <button class="btn" onclick={onRefresh}>
      <RefreshCw size={16} strokeWidth={2.1} />
      <span>Refresh</span>
    </button>
  </div>
</header>

<style>
  .workspace-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  h1 {
    color: var(--text);
    font-size: 24px;
    font-weight: 500;
    line-height: 36px;
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
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    color: var(--text);
    background: var(--bg);
    font: inherit;
    font-size: 14px;
    font-weight: 500;
    padding: 6px 8px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  .btn:hover {
    border-color: var(--color-c-gray-50, #8b8b8b);
    color: var(--text-bright);
  }
</style>
