<script>
  // Spread-over-time: is the advantage distribution widening or collapsing as
  // training proceeds? Each step's distribution is reduced to scalar spread
  // measures and plotted vs step, so "increasing?" is a single glance at the
  // slope rather than something you eyeball off overlapping bands.
  //
  //   • std   — standard deviation of the step's advantages (stats.std)
  //   • IQR   — p75 − p25, the robust middle-50% width
  //
  // A least-squares trend line is fitted to std and its net change over the run
  // is reported with a direction arrow.

  let { steps = [] } = $props();

  const W = 640;
  const H = 200;
  const PAD = 6;

  function num(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(3) : "—";
  }

  let model = $derived.by(() => {
    const pts = (steps || [])
      .filter((s) => s && s.stats)
      .map((s) => {
        const st = s.stats;
        const q = st.quantiles || {};
        return {
          x: num(s.rollout_id),
          std: num(st.std),
          iqr: Math.max(num(q.p75, num(st.max)) - num(q.p25, num(st.min)), 0),
        };
      });
    if (pts.length < 2) return null;

    const xs = pts.map((p) => p.x);
    const xMin = Math.min(...xs);
    const xSpan = Math.max(...xs) - xMin || 1;
    const yHi = Math.max(...pts.map((p) => Math.max(p.std, p.iqr))) || 1;
    const ySpan = yHi || 1;

    const sx = (x) => PAD + ((x - xMin) / xSpan) * (W - 2 * PAD);
    const sy = (v) => H - PAD - (v / ySpan) * (H - 2 * PAD);

    const line = (key) =>
      pts
        .map((p, i) => `${i ? "L" : "M"} ${sx(p.x).toFixed(1)} ${sy(p[key]).toFixed(1)}`)
        .join(" ");

    // Least-squares fit of std vs step index for the trend line + net change.
    const n = pts.length;
    const mx = pts.reduce((a, p) => a + p.x, 0) / n;
    const my = pts.reduce((a, p) => a + p.std, 0) / n;
    let sxy = 0;
    let sxx = 0;
    for (const p of pts) {
      sxy += (p.x - mx) * (p.std - my);
      sxx += (p.x - mx) ** 2;
    }
    const slope = sxx ? sxy / sxx : 0;
    const intercept = my - slope * mx;
    const fit = (x) => intercept + slope * x;
    const fitStart = fit(xMin);
    const fitEnd = fit(Math.max(...xs));
    const delta = fitEnd - fitStart;
    const pct = fitStart > 1e-9 ? (delta / fitStart) * 100 : null;
    const dir = delta > ySpan * 0.02 ? "up" : delta < -ySpan * 0.02 ? "down" : "flat";

    return {
      std: line("std"),
      iqr: line("iqr"),
      trend: `M ${sx(xMin).toFixed(1)} ${sy(fitStart).toFixed(1)} L ${sx(Math.max(...xs)).toFixed(1)} ${sy(fitEnd).toFixed(1)}`,
      dots: pts.map((p) => ({ cx: sx(p.x).toFixed(1), cy: sy(p.std).toFixed(1) })),
      yHi,
      firstX: xMin,
      lastX: Math.max(...xs),
      latestStd: pts[pts.length - 1].std,
      delta,
      pct,
      dir,
    };
  });

  const ARROW = { up: "↑", down: "↓", flat: "→" };
</script>

{#if model}
  <div class="flex flex-wrap items-center justify-between gap-[8px] mb-[8px] text-[11px] text-(--muted)">
    <span class="inline-flex flex-wrap gap-[12px]">
      <span class="chart-legend-item"><span class="sw std"></span>std</span>
      <span class="chart-legend-item"><span class="sw iqr"></span>IQR (p25–p75)</span>
      <span class="chart-legend-item"><span class="sw trend"></span>trend</span>
    </span>
    <span class="spread-trend trend-{model.dir}">
      std {ARROW[model.dir]}
      {model.delta >= 0 ? "+" : ""}{fmt(model.delta)}
      {#if model.pct != null}
        ({model.pct >= 0 ? "+" : ""}{model.pct.toFixed(0)}%)
      {/if}
      over run
    </span>
  </div>
  <svg class="w-full h-[200px] block" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">
    <path d={model.iqr} fill="none" stroke="var(--muted)" stroke-width="1.25" stroke-opacity="0.7" />
    <path
      d={model.trend}
      fill="none"
      stroke="var(--accent)"
      stroke-width="1"
      stroke-opacity="0.55"
      stroke-dasharray="5 4"
    />
    <path d={model.std} fill="none" stroke="var(--accent)" stroke-width="1.75" />
    {#each model.dots as d}
      <circle cx={d.cx} cy={d.cy} r="2" fill="var(--accent)" />
    {/each}
  </svg>
  <div class="fan-meta">
    <span>0</span>
    <span>latest std {fmt(model.latestStd)}</span>
    <span>max {fmt(model.yHi)}</span>
  </div>
  <div class="fan-axis">
    <span>step {model.firstX}</span>
    <span class="fan-axis-label">training step</span>
    <span>step {model.lastX}</span>
  </div>
{:else}
  <div class="plot-empty">Advantage distribution needs ≥2 steps of data.</div>
{/if}
