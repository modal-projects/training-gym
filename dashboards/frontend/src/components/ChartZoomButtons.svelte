<script>
  // Explicit per-chart zoom in / zoom out pair. The host decides what a step
  // means (usually `zoomFractions(ZOOM_STEP)` fed through its brush handler).
  import ZoomInIcon from "lucide-svelte/icons/zoom-in";
  import ZoomOutIcon from "lucide-svelte/icons/zoom-out";

  let { onZoomIn, onZoomOut, canZoomIn = true, canZoomOut = true, size = 12 } = $props();

  // Keep a click on the buttons from starting a brush drag on a hosting plot.
  const swallow = (event) => event.stopPropagation();
</script>

<div class="chart-zoom-buttons" role="group" aria-label="Zoom" onpointerdown={swallow}>
  <button
    type="button"
    disabled={!canZoomIn}
    onclick={onZoomIn}
    title="Zoom in"
    aria-label="Zoom in"
  >
    <ZoomInIcon {size} />
  </button>
  <button
    type="button"
    disabled={!canZoomOut}
    onclick={onZoomOut}
    title="Zoom out"
    aria-label="Zoom out"
  >
    <ZoomOutIcon {size} />
  </button>
</div>
