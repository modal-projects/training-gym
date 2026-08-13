<script>
  let { width = "100%", height = 16, light = false } = $props();
</script>

<span
  class="skeleton-pulse"
  class:light
  style:width
  style:height={`${height}px`}
></span>
