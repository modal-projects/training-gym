<script>
  let {
    allSelected = false,
    noneSelected = false,
    onSelectAll = () => {},
    onDeselectAll = () => {},
    selectLabel = "Select all",
    deselectLabel = "Deselect all",
  } = $props();
</script>

<div class="filter-actions">
  <button
    class="filter-action"
    disabled={allSelected}
    onclick={(event) => {
      event.stopPropagation();
      onSelectAll();
    }}
  >
    {selectLabel}
  </button>
  <button
    class="filter-action"
    disabled={noneSelected}
    onclick={(event) => {
      event.stopPropagation();
      onDeselectAll();
    }}
  >
    {deselectLabel}
  </button>
</div>
