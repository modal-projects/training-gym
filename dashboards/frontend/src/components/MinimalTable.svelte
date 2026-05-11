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
    border: 0;
    border-radius: 0;
    background: transparent;
  }

  :where(.minimal-table) :global(th) {
    background: transparent;
    padding: 8px 16px;
    text-align: left;
    vertical-align: middle;
    font-size: 14px;
    font-weight: 400;
    color: var(--muted);
    line-height: 20px;
  }

  :where(.minimal-table) :global(tbody tr:hover) {
    background: color-mix(in srgb, var(--text-bright) 4%, transparent);
  }

  :where(.minimal-table) :global(tbody td) {
    border-top: 1px solid var(--color-c-gray-10, #2f2f2f);
    padding: 8px 16px;
    vertical-align: middle;
    height: 40px;
  }
</style>
