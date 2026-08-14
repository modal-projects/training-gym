<script>
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
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
  let bestScore = $derived.by(() => {
    const scores = runs
      .map((run) => run.best_dev_score)
      .filter((v) => typeof v === "number" && Number.isFinite(v));
    return scores.length ? Math.max(...scores) : null;
  });

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

<section class="summary-sticky grid grid-cols-4 gap-[14px] p-[0_24px] mb-[24px] max-[900px]:grid-cols-2">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{runs.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Live runs</span>
    <strong>{runningTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Finished runs</span>
    <strong>{finishedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Best dev score</span>
    <strong>{fmtScore(bestScore)}</strong>
  </article>
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
        columns={["Run", "Status", "Task", "Scaffold", "Track", "Best dev", "Log entries", "Learning actions", "GPU hours", "Started", "Duration"]}
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
            <th>Task</th>
            <th>Scaffold</th>
            <th>Track</th>
            <th>Best dev</th>
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
              <td>{run.task || "—"}</td>
              <td class="[font-family:var(--font-mono)] text-[12px]">{run.scaffold || "—"}</td>
              <td>{run.track || "—"}</td>
              <td class="[font-variant-numeric:tabular-nums]">
                {fmtScore(run.best_dev_score)}
                {#if run.best_tag}
                  <span class="text-(--muted) text-[11px]"> ({run.best_tag})</span>
                {/if}
              </td>
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
