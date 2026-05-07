<script>
  let { class: classOverride = "", children } = $props();
</script>

<table class={`minimal-table ${classOverride}`.trim()}>
  {@render children()}
</table>

<style>
  .minimal-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--panel);
  }

  :where(.minimal-table) :global(th) {
    background: var(--panel-alt);
    padding: 0.58rem 0.88rem;
    text-align: left;
    vertical-align: middle;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
  }

  :where(.minimal-table) :global(tbody tr:hover) {
    background: color-mix(in srgb, var(--text-bright) 4%, transparent);
  }

  :where(.minimal-table) :global(tbody td) {
    border-top: 1px solid var(--border);
    padding: 0.58rem 0.88rem;
    vertical-align: middle;
  }
</style>
