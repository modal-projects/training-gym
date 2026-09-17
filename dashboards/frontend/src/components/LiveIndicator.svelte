<script>
  let { widthPx = 6, variant = "live" } = $props();

  const colorByVariant = {
    live: "var(--color-surface-accent, var(--accent))",
    inactive: "var(--color-icon-secondary, var(--muted))",
    warning: "var(--color-surface-warning, var(--yellow))",
    stopped: "var(--color-surface-danger, var(--red))",
  };

  let color = $derived(colorByVariant[variant] ?? colorByVariant.live);
  let animated = $derived(variant !== "inactive" && variant !== "stopped");
</script>

<div
  class={["live-dot", animated && "animated"]}
  style:--color={color}
  style:width="{widthPx}px"
  style:height="{widthPx}px"
></div>

<style>
  .live-dot {
    position: relative;
    flex: 0 0 auto;
    border-radius: 9999px;
    background-color: var(--color);
  }

  .live-dot.animated {
    animation: live-pulse 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
  }

  .live-dot::before,
  .live-dot::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 9999px;
    background-color: var(--color);
  }

  .live-dot.animated::before {
    animation: live-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
  }

  .live-dot.animated::after {
    animation: live-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite 0.4s;
  }

  @keyframes live-pulse {
    0%,
    100% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.1);
    }
  }

  @keyframes live-ring {
    0% {
      transform: scale(1);
      opacity: 0.6;
    }
    100% {
      transform: scale(2);
      opacity: 0;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .live-dot,
    .live-dot::before,
    .live-dot::after {
      animation: none;
    }
  }
</style>
