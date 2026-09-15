<script>
  // Port of the Modal dashboard's `ChartControls.svelte`: shift the window
  // left/right by one duration and toggle live tailing. `timeRange` is the
  // resolved `{ start, end, live }` and `setTimeRange({ start, end, live })`
  // commits a new window. `minStart` bounds how far left the window can go.
  import ChevronsLeftIcon from "lucide-svelte/icons/chevrons-left";
  import ChevronsRightIcon from "lucide-svelte/icons/chevrons-right";
  import PauseIcon from "lucide-svelte/icons/pause";
  import PlayIcon from "lucide-svelte/icons/play";

  import { useGlobalHotkey } from "../lib/hotkey.svelte.js";
  import {
    getTimeRangeParams,
    shiftRangeLeft,
    shiftRangeRight,
  } from "../lib/timeRange.js";

  let { timeRange: timeRangeProp, setTimeRange, minStart = -Infinity, now } = $props();

  let timeRangeParams = $derived(getTimeRangeParams(timeRangeProp));
  let live = $derived(timeRangeParams.live);

  // Narrowed view of the current time range: defined only when start, end,
  // and duration are all known.
  let timeRange = $derived(
    timeRangeParams.start !== undefined &&
      timeRangeParams.end !== undefined &&
      timeRangeParams.duration !== undefined
      ? {
          start: timeRangeParams.start,
          end: timeRangeParams.end,
          duration: timeRangeParams.duration,
        }
      : undefined,
  );

  let canShiftLeft = $derived(
    timeRange !== undefined && timeRange.duration > 0 && timeRange.start > minStart,
  );

  let canShiftRight = $derived(
    timeRange !== undefined && !live && timeRange.end < now,
  );

  let canToggleLive = $derived(timeRange !== undefined && timeRange.duration > 0);

  function shiftLeft() {
    if (!timeRange || !canShiftLeft) return;
    setTimeRange(shiftRangeLeft(timeRange));
  }

  function shiftRight() {
    if (!timeRange || !canShiftRight) return;
    setTimeRange(shiftRangeRight(timeRange, { now }));
  }

  function toggleLive() {
    if (!timeRange || !canToggleLive) return;
    const { start, end, duration } = timeRange;
    if (live) {
      setTimeRange({ start, end, live: false });
    } else {
      setTimeRange({ start: now - duration, end: now, live: true });
    }
  }

  useGlobalHotkey("a", shiftLeft);
  useGlobalHotkey("t", toggleLive);
  useGlobalHotkey("d", shiftRight);
</script>

<div class="btn-group">
  <button
    type="button"
    class="btn"
    disabled={!canShiftLeft}
    onclick={shiftLeft}
    title="Shift ← (A)"
    aria-label="Shift left"
  >
    <ChevronsLeftIcon size={16} class="icon" />
  </button>
  <button
    type="button"
    class="btn"
    disabled={!canToggleLive}
    onclick={toggleLive}
    title="Toggle Pause/Play (T)"
    aria-label={live ? "Pause" : "Play"}
    aria-pressed={live}
  >
    {#if live}
      <PauseIcon size={16} class="icon" />
    {:else}
      <PlayIcon size={16} class="icon" />
    {/if}
  </button>
  <button
    type="button"
    class="btn"
    disabled={!canShiftRight}
    onclick={shiftRight}
    title="Shift → (D)"
    aria-label="Shift right"
  >
    <ChevronsRightIcon size={16} class="icon" />
  </button>
</div>
