<script>
  import { toEpochSeconds, fmtDate } from "../lib/format.js";
  import { fmtScore } from "../lib/learning.js";

  // The run's whole story on one time axis: the dev-score trajectory
  // (step-line, drawn on load), a mark for every research-log entry
  // (click to jump to it), and amber bands where GPU jobs burned.
  let {
    entries = [], // learning-log entries with _ts / _i already attached
    gpuLog = [], // scores.gpu_log rows {ts, seconds, n_gpus}
    launchedAt = null,
    runState = "",
    onJump = () => {},
  } = $props();

  let width = $state(720);
  let hover = $state(null); // {x, entry}

  const H = 168;
  const PAD_L = 10;
  const PAD_R = 10;
  const SCORE_TOP = 16;
  const SCORE_H = 84;
  const MARK_Y = SCORE_TOP + SCORE_H + 18;
  const GPU_Y = MARK_Y + 16;
  const GPU_H = 7;
  const AXIS_Y = GPU_Y + GPU_H + 14;

  let live = $derived(String(runState).toLowerCase() === "running");

  let stamped = $derived(
    entries.filter((e) => e._ts).sort((a, b) => a._ts - b._ts),
  );

  let gpuJobs = $derived(
    (Array.isArray(gpuLog) ? gpuLog : [])
      .map((r) => {
        const end = toEpochSeconds(r?.ts);
        const secs = Number(r?.seconds) || 0;
        return end ? { start: end - secs, end, gpus: Number(r?.n_gpus) || 1 } : null;
      })
      .filter(Boolean),
  );

  let t0 = $derived.by(() => {
    const launch = toEpochSeconds(launchedAt);
    if (launch) return launch;
    return stamped.length ? stamped[0]._ts : null;
  });

  let tEnd = $derived.by(() => {
    let end = t0 ?? 0;
    if (stamped.length) end = Math.max(end, stamped[stamped.length - 1]._ts);
    for (const j of gpuJobs) end = Math.max(end, j.end);
    if (live) end = Math.max(end, Date.now() / 1000);
    return end;
  });

  let span = $derived(Math.max((tEnd ?? 0) - (t0 ?? 0), 600));

  function x(t) {
    return PAD_L + ((t - t0) / span) * (width - PAD_L - PAD_R);
  }

  let scored = $derived(
    stamped.filter(
      (e) => typeof e.dev_score === "number" && Number.isFinite(e.dev_score),
    ),
  );

  let yMax = $derived(Math.max(1, ...scored.map((e) => e.dev_score)));

  function yScore(s) {
    return SCORE_TOP + (1 - s / yMax) * SCORE_H;
  }

  let scorePath = $derived.by(() => {
    if (scored.length === 0) return "";
    let d = `M ${x(scored[0]._ts).toFixed(1)} ${yScore(scored[0].dev_score).toFixed(1)}`;
    for (let i = 1; i < scored.length; i++) {
      d += ` H ${x(scored[i]._ts).toFixed(1)} V ${yScore(scored[i].dev_score).toFixed(1)}`;
    }
    // hold the last score to the right edge of the run
    d += ` H ${x(tEnd).toFixed(1)}`;
    return d;
  });

  let areaPath = $derived.by(() => {
    if (!scorePath) return "";
    const floor = SCORE_TOP + SCORE_H;
    return `${scorePath} V ${floor} H ${x(scored[0]._ts).toFixed(1)} Z`;
  });

  let best = $derived.by(() => {
    let b = null;
    for (const e of scored) if (!b || e.dev_score > b.dev_score) b = e;
    return b;
  });

  // Hour ticks: pick a step that yields ~5-8 labels.
  let ticks = $derived.by(() => {
    const hours = span / 3600;
    const step = [1, 2, 4, 6, 12, 24, 48].find((s) => hours / s <= 8) ?? 96;
    const out = [];
    for (let h = 0; h <= hours + 0.001; h += step) {
      out.push({ x: x(t0 + h * 3600), label: h === 0 ? "0h" : `+${h}h` });
    }
    return out;
  });

  const KIND_COLOR = {
    checkpoint: "var(--green, #4ade80)",
    submission: "var(--blue, #60a5fa)",
    note: "rgba(255,255,255,0.38)",
  };

  function kindOf(e) {
    const k = String(e.kind || "note").toLowerCase();
    return KIND_COLOR[k] ? k : "note";
  }

  function markLabel(e) {
    const score =
      typeof e.dev_score === "number" ? ` · dev ${fmtScore(e.dev_score)}` : "";
    return `${kindOf(e)}${e.tag ? ` ${e.tag}` : ""}${score}`;
  }

  function showTip(e) {
    hover = { px: Math.min(Math.max(x(e._ts), 70), width - 70), entry: e };
  }
</script>

<div class="chron" bind:clientWidth={width}>
  <svg width="100%" height={H} viewBox={`0 0 ${width} ${H}`} preserveAspectRatio="none" role="img" aria-label="Run timeline: score trajectory, experiments, GPU jobs">
    <!-- score gridlines -->
    {#each [0, 0.5, 1] as g (g)}
      <line x1={PAD_L} x2={width - PAD_R} y1={yScore(g * yMax)} y2={yScore(g * yMax)} class="chron-grid" />
      <text x={width - PAD_R} y={yScore(g * yMax) - 3} class="chron-gridlabel" text-anchor="end">{fmtScore(g * yMax)}</text>
    {/each}

    <!-- gpu job bands -->
    {#each gpuJobs as job, i (i)}
      <rect
        x={x(Math.max(job.start, t0))}
        y={GPU_Y}
        width={Math.max(x(job.end) - x(Math.max(job.start, t0)), 1.5)}
        height={GPU_H}
        rx="1.5"
        class="chron-gpu"
      >
        <title>{job.gpus} gpu · {Math.round((job.end - job.start) / 60)}m</title>
      </rect>
    {/each}

    <!-- score trajectory -->
    {#if areaPath}
      <path d={areaPath} class="chron-area" />
      <path d={scorePath} class="chron-score" pathLength="1" />
    {/if}
    {#if best}
      <circle cx={x(best._ts)} cy={yScore(best.dev_score)} r="3" class="chron-best" />
      <text x={x(best._ts)} y={yScore(best.dev_score) - 7} class="chron-bestlabel" text-anchor="middle">{fmtScore(best.dev_score)}</text>
    {/if}

    <!-- experiment marks -->
    {#each stamped as e (e._i)}
      {@const mx = x(e._ts)}
      {@const kind = kindOf(e)}
      <g
        class="chron-mark"
        role="button"
        tabindex="0"
        aria-label={markLabel(e)}
        onclick={() => onJump(e._i)}
        onkeydown={(ev) => (ev.key === "Enter" || ev.key === " ") && onJump(e._i)}
        onmouseenter={() => showTip(e)}
        onmouseleave={() => (hover = null)}
        onfocus={() => showTip(e)}
        onblur={() => (hover = null)}
      >
        <rect x={mx - 7} y={MARK_Y - 8} width="14" height="16" fill="transparent" />
        {#if kind === "checkpoint"}
          <rect x={mx - 3.4} y={MARK_Y - 3.4} width="6.8" height="6.8" transform={`rotate(45 ${mx} ${MARK_Y})`} fill={KIND_COLOR.checkpoint} />
        {:else if kind === "submission"}
          <rect x={mx - 3.4} y={MARK_Y - 3.4} width="6.8" height="6.8" transform={`rotate(45 ${mx} ${MARK_Y})`} fill="none" stroke={KIND_COLOR.submission} stroke-width="1.5" />
        {:else}
          <circle cx={mx} cy={MARK_Y} r="2" fill={KIND_COLOR.note} />
        {/if}
      </g>
    {/each}

    <!-- live cursor -->
    {#if live}
      <line x1={x(tEnd)} x2={x(tEnd)} y1={SCORE_TOP - 6} y2={AXIS_Y - 8} class="chron-now" />
      <circle cx={x(tEnd)} cy={SCORE_TOP - 6} r="2.5" class="chron-nowdot" />
    {/if}

    <!-- time axis -->
    <line x1={PAD_L} x2={width - PAD_R} y1={AXIS_Y} y2={AXIS_Y} class="chron-axis" />
    {#each ticks as tick (tick.label)}
      <line x1={tick.x} x2={tick.x} y1={AXIS_Y} y2={AXIS_Y + 3.5} class="chron-axis" />
      <text x={tick.x} y={AXIS_Y + 13} class="chron-ticklabel" text-anchor="middle">{tick.label}</text>
    {/each}
  </svg>

  {#if hover}
    <div class="chron-tip" style={`left:${hover.px}px`}>
      <span class="chron-tip-kind" style={`color:${KIND_COLOR[kindOf(hover.entry)]}`}>{kindOf(hover.entry)}</span>
      {#if hover.entry.tag}<span class="chron-tip-tag">{hover.entry.tag}</span>{/if}
      {#if typeof hover.entry.dev_score === "number"}<span class="chron-tip-score">{fmtScore(hover.entry.dev_score)}</span>{/if}
      <div class="chron-tip-what">{String(hover.entry.what || "").slice(0, 120)}</div>
      <div class="chron-tip-time">{hover.entry._ts ? fmtDate(hover.entry._ts) : ""}</div>
    </div>
  {/if}
</div>

<style>
  .chron {
    position: relative;
    width: 100%;
  }
  .chron-grid {
    stroke: rgba(255, 255, 255, 0.06);
    stroke-width: 1;
  }
  .chron-gridlabel,
  .chron-ticklabel {
    font-family: var(--font-mono);
    font-size: 9px;
    fill: var(--muted, #9ca3af);
    opacity: 0.7;
  }
  .chron-axis {
    stroke: rgba(255, 255, 255, 0.14);
    stroke-width: 1;
  }
  .chron-area {
    fill: var(--green, #4ade80);
    opacity: 0.07;
  }
  .chron-score {
    fill: none;
    stroke: var(--green, #4ade80);
    stroke-width: 1.6;
    stroke-dasharray: 1;
    stroke-dashoffset: 0;
    animation: chron-draw 1.1s ease-out;
  }
  @keyframes chron-draw {
    from {
      stroke-dashoffset: 1;
    }
    to {
      stroke-dashoffset: 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .chron-score {
      animation: none;
    }
  }
  .chron-best {
    fill: var(--green, #4ade80);
  }
  .chron-bestlabel {
    font-family: var(--font-mono);
    font-size: 9.5px;
    fill: var(--green, #4ade80);
  }
  .chron-gpu {
    fill: var(--yellow, #fbbf24);
    opacity: 0.65;
  }
  .chron-mark {
    cursor: pointer;
  }
  .chron-mark:hover rect,
  .chron-mark:hover circle {
    opacity: 0.8;
  }
  .chron-mark:focus-visible {
    outline: 1.5px solid var(--green, #4ade80);
    outline-offset: 2px;
  }
  .chron-now {
    stroke: var(--red, #f87171);
    stroke-width: 1;
    stroke-dasharray: 2 3;
    opacity: 0.8;
  }
  .chron-nowdot {
    fill: var(--red, #f87171);
  }
  .chron-tip {
    position: absolute;
    top: -8px;
    transform: translate(-50%, -100%);
    max-width: 320px;
    min-width: 140px;
    background: color-mix(in srgb, black 88%, var(--panel-alt, #1f1f1f));
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 6px;
    padding: 7px 10px;
    pointer-events: none;
    z-index: 5;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
  }
  .chron-tip-kind {
    font-family: var(--font-mono);
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .chron-tip-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-bright, #e5e5e5);
    margin-left: 6px;
  }
  .chron-tip-score {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--green, #4ade80);
    margin-left: 6px;
  }
  .chron-tip-what {
    font-size: 11.5px;
    line-height: 15px;
    color: var(--text, #c9c9c9);
    margin-top: 3px;
  }
  .chron-tip-time {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--muted, #9ca3af);
    margin-top: 3px;
  }
</style>
