<script>
  import { formatTimeTick, timeTicks } from "../lib/timeAxis.js";

  let {
    // Epoch seconds at the left and right edges of the plot.
    start = null,
    end = null,
    // Horizontal position of a timestamp as a fraction of the plot width;
    // linear by default, overridable for compressed or column-based axes.
    fractionAt = null,
    // Approximate pixel width per label; controls how many ticks are drawn.
    labelWidth = 80,
  } = $props();

  let width = $state(0);

  let ticks = $derived.by(() => {
    if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return [];
    const count = Math.max(2, Math.floor((width || 480) / labelWidth));
    const place = fractionAt ?? ((t) => (t - start) / (end - start));
    const minGap = width ? (labelWidth * 0.7) / width : 0;
    const out = [];
    for (const t of timeTicks(start, end, count)) {
      const f = place(t);
      if (!Number.isFinite(f) || f < 0 || f > 1) continue;
      if (out.length && f - out[out.length - 1].f < minGap) continue;
      out.push({ t, f, label: formatTimeTick(t) });
    }
    return out;
  });

  function anchor(f) {
    if (!width) return "";
    if (f * width < labelWidth / 2) return "anchor-start";
    if ((1 - f) * width < labelWidth / 2) return "anchor-end";
    return "";
  }
</script>

<div class="time-axis" bind:clientWidth={width} aria-hidden="true">
  {#each ticks as tick (tick.t)}
    <div class="time-tick {anchor(tick.f)}" style:left={`${tick.f * 100}%`}>
      <span class="time-tick-mark"></span>
      <span class="time-tick-label">{tick.label}</span>
    </div>
  {/each}
</div>

<style>
  .time-axis {
    position: relative;
    height: 20px;
    border-top: 1px solid var(--border, #2a2a2a);
    user-select: none;
  }

  .time-tick {
    position: absolute;
    top: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    transform: translateX(-50%);
  }

  .time-tick.anchor-start {
    align-items: flex-start;
    transform: none;
  }

  .time-tick.anchor-end {
    align-items: flex-end;
    transform: translateX(-100%);
  }

  .time-tick-mark {
    width: 1px;
    height: 4px;
    background: var(--border-strong, #4a4a4a);
  }

  .time-tick-label {
    margin-top: 2px;
    font-size: 10px;
    line-height: 1.2;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
</style>
