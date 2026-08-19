<script>
  import { toEpochSeconds } from "../lib/format.js";

  // A run's learning curve in miniature: step-line of dev scores over time.
  // `series` is [[ts, score], ...] straight from the API.
  let {
    series = [],
    width = 116,
    height = 30,
    color = "var(--green, #4ade80)",
  } = $props();

  let points = $derived.by(() => {
    const rows = (Array.isArray(series) ? series : [])
      .map((pair) => ({
        t: toEpochSeconds(pair?.[0]),
        y: Number(pair?.[1]),
      }))
      .filter((p) => p.t && Number.isFinite(p.y))
      .sort((a, b) => a.t - b.t);
    return rows;
  });

  const PAD = 3;

  let coords = $derived.by(() => {
    if (!points.length) return [];
    const t0 = points[0].t;
    const span = Math.max(points[points.length - 1].t - t0, 1);
    const yMax = Math.max(1, ...points.map((p) => p.y));
    return points.map((p) => ({
      x: PAD + ((p.t - t0) / span) * (width - PAD * 2),
      y: PAD + (1 - p.y / yMax) * (height - PAD * 2),
    }));
  });

  // Step-after path: a score holds until the next experiment changes it.
  let linePath = $derived.by(() => {
    if (coords.length < 2) return "";
    let d = `M ${coords[0].x.toFixed(1)} ${coords[0].y.toFixed(1)}`;
    for (let i = 1; i < coords.length; i++) {
      d += ` H ${coords[i].x.toFixed(1)} V ${coords[i].y.toFixed(1)}`;
    }
    return d;
  });

  let areaPath = $derived.by(() => {
    if (coords.length < 2) return "";
    const floor = height - PAD;
    return `${linePath} V ${floor} H ${coords[0].x.toFixed(1)} Z`;
  });

  let last = $derived(coords.length ? coords[coords.length - 1] : null);
</script>

{#if coords.length === 0}
  <span class="spark-empty">—</span>
{:else}
  <svg {width} {height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" class="block">
    {#if areaPath}
      <path d={areaPath} fill={color} opacity="0.09" />
      <path d={linePath} fill="none" stroke={color} stroke-width="1.4" opacity="0.9" />
    {/if}
    <circle cx={last.x} cy={last.y} r="2" fill={color} />
  </svg>
{/if}

<style>
  .spark-empty {
    color: var(--muted, #9ca3af);
    opacity: 0.5;
    font-size: 12px;
  }
</style>
