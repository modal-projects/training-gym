<script>
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import Sparkline from "../components/Sparkline.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
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
  let finishedTotal = $derived(
    runs.filter((run) => runPillStatus(run.state) === "completed").length,
  );
  let bestRun = $derived.by(() => {
    let b = null;
    for (const run of runs) {
      const s = run.best_dev_score;
      if (typeof s === "number" && Number.isFinite(s) && (!b || s > b.best_dev_score)) {
        b = run;
      }
    }
    return b;
  });
  let experimentsTotal = $derived(
    runs.reduce((sum, run) => sum + (run.learning_log_entries || 0), 0),
  );

  function detailPath(runId) {
    return `/learning/${encodeURIComponent(runId)}`;
  }

  function selectRun(runId, event) {
    if (event && (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0))
      return;
    event?.preventDefault();
    onOpenDetail(runId);
  }
</script>

<section class="stat-band">
  <div class="stat">
    <span class="stat-eyebrow">runs</span>
    <strong>{runs.length}</strong>
  </div>
  <div class="stat">
    <span class="stat-eyebrow">live now</span>
    <strong class:stat-live={runningTotal > 0}>{runningTotal}</strong>
  </div>
  <div class="stat">
    <span class="stat-eyebrow">finished</span>
    <strong>{finishedTotal}</strong>
  </div>
  <div class="stat">
    <span class="stat-eyebrow">experiments logged</span>
    <strong>{experimentsTotal}</strong>
  </div>
  <div class="stat stat-wide">
    <span class="stat-eyebrow">best dev score</span>
    <span class="stat-best">
      <strong class="stat-best-num">{bestRun ? fmtScore(bestRun.best_dev_score) : "—"}</strong>
      {#if bestRun}
        <a
          href={detailPath(bestRun.run_id)}
          class="stat-best-run"
          onclick={(event) => selectRun(bestRun.run_id, event)}
        >{bestRun.best_tag ? `${bestRun.best_tag} · ` : ""}{bestRun.run_id}</a>
      {/if}
    </span>
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
              <td class="[font-family:var(--font-mono)] text-[12px]">{run.scaffold || "—"}</td>
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
  .stat-band {
    display: flex;
    align-items: stretch;
    gap: 0;
    padding: 4px 24px 22px;
    flex-wrap: wrap;
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 2px 28px 2px 0;
    margin-right: 28px;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
  }
  .stat:last-child {
    border-right: 0;
    margin-right: 0;
  }
  .stat-eyebrow {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted, #9ca3af);
  }
  .stat strong {
    font-family: var(--font-display);
    font-feature-settings: "ss01" on;
    font-size: 26px;
    line-height: 30px;
    font-weight: 600;
    color: var(--text-bright, #e5e5e5);
    font-variant-numeric: tabular-nums;
  }
  .stat-live {
    color: var(--green, #4ade80) !important;
  }
  .stat-wide {
    min-width: 0;
  }
  .stat-best {
    display: flex;
    align-items: baseline;
    gap: 10px;
    min-width: 0;
  }
  .stat-best-num {
    color: var(--green, #4ade80) !important;
  }
  .stat-best-run {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted, #9ca3af);
    text-decoration: none;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 340px;
  }
  .stat-best-run:hover {
    color: var(--text-bright, #e5e5e5);
  }
  @media (max-width: 900px) {
    .stat-band {
      padding: 4px 16px 18px;
      row-gap: 14px;
    }
    .stat {
      padding-right: 18px;
      margin-right: 18px;
    }
  }
</style>
