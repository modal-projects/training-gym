<script>
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import Sparkline from "../components/Sparkline.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import { toEpochSeconds } from "../lib/format.js";
  import {
    fmtGpuHours,
    fmtScore,
    fmtSeconds,
    learningActionCount,
    runPillStatus,
  } from "../lib/learning.js";

  let {
    runs = [],
    loading = false,
    error = null,
    search = $bindable(""),
    onOpenDetail = () => {},
  } = $props();

  let filteredRuns = $derived(
    runs.filter((run) => {
      if (!search) return true;
      const q = search.toLowerCase();
      return ["run_id", "task", "scaffold", "track", "state", "best_tag"].some(
        (key) => String(run[key] || "").toLowerCase().includes(q),
      );
    }),
  );

  let runningTotal = $derived(
    runs.filter((run) => runPillStatus(run.state) === "running").length,
  );
  let experimentsTotal = $derived(
    runs.reduce((sum, run) => sum + (run.learning_log_entries || 0), 0),
  );
  let taskCount = $derived(new Set(runs.map((r) => r.task).filter(Boolean)).size);

  function detailPath(runId) {
    return `/learning/${encodeURIComponent(runId)}`;
  }

  function selectRun(runId, event) {
    if (event && (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0))
      return;
    event?.preventDefault();
    onOpenDetail(runId);
  }

  // ── Progress strip ──────────────────────────────────────────────────────
  // The page's one question: is the system improving run over run? Every
  // scored run is a dot (launch time vs best dev score, colored by
  // scaffold); the green step is the record so far. Dots open the run.
  let stripW = $state(900);
  let stripHover = $state(null); // {px, run}

  const STRIP_H = 118;
  const S_PAD_L = 10;
  const S_PAD_R = 10;
  const S_TOP = 8;
  const S_PLOT_H = 78;
  const S_AXIS_Y = S_TOP + S_PLOT_H + 12;

  const SCAFFOLD_PALETTE = ["#2dd4bf", "#f472b6", "#fb923c", "#60a5fa", "#a78bfa", "#facc15"];

  let scaffoldList = $derived(
    [...new Set(runs.map((r) => r.scaffold).filter(Boolean))].sort(),
  );

  function scaffoldColor(name) {
    const i = scaffoldList.indexOf(name);
    return SCAFFOLD_PALETTE[i >= 0 ? i % SCAFFOLD_PALETTE.length : 0];
  }

  let scoredRuns = $derived(
    runs
      .map((run) => ({
        run,
        t: toEpochSeconds(run.launched_at),
        score: run.best_dev_score,
      }))
      .filter(
        (p) => p.t && typeof p.score === "number" && Number.isFinite(p.score),
      )
      .sort((a, b) => a.t - b.t),
  );

  let sT0 = $derived(scoredRuns.length ? scoredRuns[0].t : 0);
  let sSpan = $derived(
    scoredRuns.length
      ? Math.max(scoredRuns[scoredRuns.length - 1].t - sT0, 3600)
      : 3600,
  );
  let sYMax = $derived(Math.max(1, ...scoredRuns.map((p) => p.score)));

  function sx(t) {
    return S_PAD_L + ((t - sT0) / sSpan) * (stripW - S_PAD_L - S_PAD_R);
  }

  function sy(score) {
    return S_TOP + (1 - score / sYMax) * S_PLOT_H;
  }

  // Record-so-far frontier: a step that only ever rises.
  let frontierPath = $derived.by(() => {
    if (scoredRuns.length < 2) return "";
    let record = scoredRuns[0].score;
    let d = `M ${sx(scoredRuns[0].t).toFixed(1)} ${sy(record).toFixed(1)}`;
    for (let i = 1; i < scoredRuns.length; i++) {
      const p = scoredRuns[i];
      d += ` H ${sx(p.t).toFixed(1)}`;
      if (p.score > record) {
        record = p.score;
        d += ` V ${sy(record).toFixed(1)}`;
      }
    }
    return d;
  });

  let dateTicks = $derived.by(() => {
    if (!scoredRuns.length) return [];
    const n = Math.min(5, Math.max(2, Math.round(stripW / 260)));
    const out = [];
    for (let i = 0; i <= n; i++) {
      const t = sT0 + (sSpan * i) / n;
      out.push({
        x: sx(t),
        anchor: i === 0 ? "start" : i === n ? "end" : "middle",
        label: new Date(t * 1000).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
        }),
      });
    }
    return out;
  });

  function showStripTip(p) {
    stripHover = {
      px: Math.min(Math.max(sx(p.t), 80), stripW - 80),
      run: p.run,
      score: p.score,
    };
  }
</script>

<section class="progress-strip">
  <div class="flex items-center justify-between gap-[12px] flex-wrap">
    <h3 class="strip-eyebrow">Progress · best dev score per run</h3>
    {#if scaffoldList.length}
      <div class="strip-legend" aria-hidden="true">
        {#each scaffoldList as s (s)}
          <span title={s}><i style={`background:${scaffoldColor(s)}`}></i>{s.length > 18 ? s.slice(0, 16) + "…" : s}</span>
        {/each}
        <span><i class="strip-legend-frontier"></i>record</span>
      </div>
    {/if}
  </div>

  {#if scoredRuns.length >= 2}
    <div class="strip-chart" bind:clientWidth={stripW}>
      <svg width="100%" height={STRIP_H} viewBox={`0 0 ${stripW} ${STRIP_H}`} preserveAspectRatio="none" role="img" aria-label="Best dev score of every run over time">
        {#each [0, 0.5, 1] as g (g)}
          <line x1={S_PAD_L} x2={stripW - S_PAD_R} y1={sy(g * sYMax)} y2={sy(g * sYMax)} class="strip-grid" />
          <text x={stripW - S_PAD_R} y={sy(g * sYMax) - 3} class="strip-gridlabel" text-anchor="end">{fmtScore(g * sYMax)}</text>
        {/each}

        {#if frontierPath}
          <path d={frontierPath} class="strip-frontier" />
        {/if}

        {#each scoredRuns as p (p.run.run_id)}
          <g
            class="strip-dot"
            role="button"
            tabindex="0"
            aria-label={`${p.run.run_id} · dev ${fmtScore(p.score)}`}
            onclick={() => onOpenDetail(p.run.run_id)}
            onkeydown={(ev) => (ev.key === "Enter" || ev.key === " ") && onOpenDetail(p.run.run_id)}
            onmouseenter={() => showStripTip(p)}
            onmouseleave={() => (stripHover = null)}
            onfocus={() => showStripTip(p)}
            onblur={() => (stripHover = null)}
          >
            <circle cx={sx(p.t)} cy={sy(p.score)} r="8" fill="transparent" />
            <circle cx={sx(p.t)} cy={sy(p.score)} r="3.4" fill={scaffoldColor(p.run.scaffold)} class="strip-dot-core" />
          </g>
        {/each}

        <line x1={S_PAD_L} x2={stripW - S_PAD_R} y1={S_AXIS_Y} y2={S_AXIS_Y} class="strip-axis" />
        {#each dateTicks as tick (tick.x)}
          <line x1={tick.x} x2={tick.x} y1={S_AXIS_Y} y2={S_AXIS_Y + 3.5} class="strip-axis" />
          <text x={tick.x} y={S_AXIS_Y + 13} class="strip-gridlabel" text-anchor={tick.anchor}>{tick.label}</text>
        {/each}
      </svg>

      {#if stripHover}
        <div class="strip-tip" style={`left:${stripHover.px}px`}>
          <span class="strip-tip-score">{fmtScore(stripHover.score)}</span>
          {#if stripHover.run.best_tag}<span class="strip-tip-tag">{stripHover.run.best_tag}</span>{/if}
          <div class="strip-tip-id">{stripHover.run.run_id}</div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="page-empty">Not enough scored runs to chart progress yet.</div>
  {/if}

  <div class="strip-context">
    {runs.length} runs · {taskCount} task{taskCount === 1 ? "" : "s"} ·
    {scaffoldList.length} scaffold{scaffoldList.length === 1 ? "" : "s"} ·
    {experimentsTotal} experiments logged ·
    <span class:strip-live={runningTotal > 0}>{runningTotal} live now</span>
  </div>
</section>

<section class="[border:0] [background:transparent] flex flex-col gap-[16px] p-[0_24px_16px] max-[900px]:p-[0_16px_24px] min-w-0">
  <input
    class="search-input max-w-[420px]"
    type="search"
    placeholder="Search runs (id, task, scaffold, track)…"
    bind:value={search}
    aria-label="Search learning runs"
  />

  {#if loading && !runs.length}
    <div class="table-wrap freeze-header">
      <MinimalTableSkeleton
        columns={["Run", "Status", "Learning curve", "Best dev", "Task", "Scaffold", "Track", "Log entries", "Learning actions", "GPU hours", "Started", "Duration"]}
        rows={8}
      />
    </div>
  {:else if error && !runs.length}
    <div class="page-empty">Failed to load: {error}</div>
  {:else if !runs.length}
    <div class="page-empty">
      No learning-agent runs found. Runs appear here after the observatory
      ingests them (observatory/cli.py ingest / watch).
    </div>
  {:else if !filteredRuns.length}
    <div class="page-empty">No runs match the current search.</div>
  {:else}
    <div class="table-wrap freeze-header">
      <MinimalTable>
        <thead>
          <tr>
            <th>Run</th>
            <th>Status</th>
            <th>Learning curve</th>
            <th>Best dev</th>
            <th>Task</th>
            <th>Scaffold</th>
            <th>Track</th>
            <th>Log entries</th>
            <th>Learning actions</th>
            <th>GPU hours</th>
            <th>Started</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>
          {#each filteredRuns as run (run.run_id)}
            <tr class="run-row">
              <td class="min-w-0 row-open-cell">
                <a
                  href={detailPath(run.run_id)}
                  class="cell-open-button"
                  title={run.run_id}
                  onclick={(event) => selectRun(run.run_id, event)}
                >
                  <div class="block text-(--text-bright) [font-family:var(--font-mono)] text-[13px] leading-[20px] overflow-hidden text-ellipsis whitespace-nowrap">
                    {run.run_id}
                  </div>
                </a>
              </td>
              <td><StatusPill status={runPillStatus(run.state)} label={run.state || null} /></td>
              <td>
                <a href={detailPath(run.run_id)} class="inline-block" onclick={(event) => selectRun(run.run_id, event)} aria-label={`Learning curve for ${run.run_id}`}>
                  <Sparkline series={run.score_series} />
                </a>
              </td>
              <td class="[font-variant-numeric:tabular-nums]">
                {fmtScore(run.best_dev_score)}
                {#if run.best_tag}
                  <span class="text-(--muted) text-[11px]"> ({run.best_tag})</span>
                {/if}
              </td>
              <td>{run.task || "—"}</td>
              <td class="[font-family:var(--font-mono)] text-[12px]">
                <span class="scaffold-chip"><i style={`background:${scaffoldColor(run.scaffold)}`}></i>{run.scaffold || "—"}</span>
              </td>
              <td>{run.track || "—"}</td>
              <td class="[font-variant-numeric:tabular-nums]" class:text-(--muted)={!run.learning_log_entries}>{run.learning_log_entries ?? "—"}</td>
              <td class="[font-variant-numeric:tabular-nums]">{learningActionCount(run) ?? "—"}</td>
              <td class="[font-variant-numeric:tabular-nums]">{fmtGpuHours(run.gpu_hours)}</td>
              <td class="whitespace-nowrap">
                <TimeAgo timestamp={run.launched_at} showJustNow falsyRepresentation="—" />
              </td>
              <td class="whitespace-nowrap [font-variant-numeric:tabular-nums]">{fmtSeconds(run.duration_s)}</td>
            </tr>
          {/each}
        </tbody>
      </MinimalTable>
    </div>
  {/if}
</section>

<style>
  .progress-strip {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 2px 24px 20px;
  }
  .strip-eyebrow {
    margin: 0;
    font-family: var(--font-mono);
    font-size: 10.5px;
    line-height: 16px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
    color: color-mix(in srgb, var(--text, #c9c9c9) 74%, white);
  }
  .strip-legend {
    display: flex;
    gap: 14px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.04em;
    color: var(--muted, #9ca3af);
    align-items: center;
    flex-wrap: wrap;
  }
  .strip-legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .strip-legend i {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 999px;
  }
  .strip-legend-frontier {
    width: 12px !important;
    height: 2px !important;
    border-radius: 0 !important;
    background: var(--green, #4ade80);
    opacity: 0.7;
  }
  .strip-chart {
    position: relative;
    width: 100%;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    padding: 8px 10px 2px;
  }
  .strip-grid {
    stroke: rgba(255, 255, 255, 0.06);
    stroke-width: 1;
  }
  .strip-gridlabel {
    font-family: var(--font-mono);
    font-size: 9px;
    fill: var(--muted, #9ca3af);
    opacity: 0.7;
  }
  .strip-axis {
    stroke: rgba(255, 255, 255, 0.14);
    stroke-width: 1;
  }
  .strip-frontier {
    fill: none;
    stroke: var(--green, #4ade80);
    stroke-width: 1.3;
    opacity: 0.55;
  }
  .strip-dot {
    cursor: pointer;
  }
  .strip-dot:hover .strip-dot-core {
    r: 4.5;
  }
  .strip-dot:focus-visible {
    outline: 1.5px solid var(--green, #4ade80);
    outline-offset: 2px;
  }
  .strip-tip {
    position: absolute;
    top: -6px;
    transform: translate(-50%, -100%);
    background: color-mix(in srgb, black 88%, var(--panel-alt, #1f1f1f));
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 6px;
    padding: 6px 10px;
    pointer-events: none;
    z-index: 5;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
    white-space: nowrap;
  }
  .strip-tip-score {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--green, #4ade80);
  }
  .strip-tip-tag {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-bright, #e5e5e5);
    margin-left: 6px;
  }
  .strip-tip-id {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--muted, #9ca3af);
    margin-top: 2px;
  }
  .strip-context {
    font-family: var(--font-mono);
    font-size: 10.5px;
    letter-spacing: 0.05em;
    color: var(--muted, #9ca3af);
  }
  .strip-live {
    color: var(--green, #4ade80);
  }
  .scaffold-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .scaffold-chip i {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 999px;
    flex: 0 0 auto;
  }
  @media (max-width: 900px) {
    .progress-strip {
      padding: 2px 16px 16px;
    }
  }
</style>
