<script>
  let { class: classOverride = "", children, ...restProps } = $props();
</script>

<table {...restProps} class={`minimal-table ${classOverride}`.trim()}>
  {@render children()}
</table>
