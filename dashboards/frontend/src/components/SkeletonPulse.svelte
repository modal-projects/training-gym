<script>
  let { width = "100%", height = 16, light = false } = $props();
</script>

<span
  class="skeleton-pulse"
  class:light
  style:width
  style:height={`${height}px`}
></span>

<style>
  .skeleton-pulse {
    display: inline-block;
    border-radius: 6px;
    background: #2f2f2f;
    animation: skeleton-pulse 1.2s ease-in-out infinite;
  }

  .skeleton-pulse.light {
    background: #242424;
  }

  @keyframes skeleton-pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.45;
    }
  }
</style>
