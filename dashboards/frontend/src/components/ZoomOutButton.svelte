<script>
  // Port of the Modal dashboard's `ZoomOutButton.svelte`: triple the window
  // around its centre. `maxDuration` caps the window (the run's lifetime).
  import ZoomOutIcon from "lucide-svelte/icons/zoom-out";

  import { useGlobalHotkey } from "../lib/hotkey.svelte.js";
  import { getTimeRangeParams, zoomOutRange } from "../lib/timeRange.js";

  let { timeRange, setTimeRange, maxDuration, now } = $props();

  let timeRangeParams = $derived(getTimeRangeParams(timeRange));
  let duration = $derived(timeRangeParams.duration);

  let zoomOutParams = $derived.by(() => {
    const { start, end, live } = timeRangeParams;
    if (duration === undefined || end === undefined || start === undefined) return undefined;
    return zoomOutRange({ start, end, live }, { now, maxDuration });
  });

  let maxDurationReached = $derived(duration !== undefined && duration >= maxDuration);

  function zoomOut() {
    if (!zoomOutParams || maxDurationReached) return;
    setTimeRange({ ...zoomOutParams });
  }

  useGlobalHotkey("z", zoomOut);
</script>

<button
  type="button"
  class="btn"
  disabled={!zoomOutParams || maxDurationReached}
  onclick={zoomOut}
  title={maxDurationReached ? "Showing the entire run" : "Zoom out (Z)"}
  aria-label="Zoom out"
>
  <ZoomOutIcon size={16} class="icon" />
</button>
