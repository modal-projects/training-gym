<script>
  import MinimalTable from "../components/MinimalTable.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";

  let {
    allEvals,
    avgAccuracy,
    evalConfigCount,
    loading,
    error,
    evalConfigGroups,
  } = $props();
</script>

<section class="summary-row">
  <article class="summary-card">
    <span class="summary-label">Total evals</span>
    <strong>{allEvals.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Avg score</span>
    <strong>{allEvals.length ? avgAccuracy.toFixed(4) : "—"}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Eval configs</span>
    <strong>{evalConfigCount}</strong>
  </article>
</section>

<section class="runs-surface">
  <div class="runs-body">
    {#if loading}
      <div class="table-wrap">
        <MinimalTableSkeleton
          class="runs-table"
          columns={["Rank", "Eval run", "Avg score", "Delta vs best", "Examples", "Created", "Links"]}
          rows={6}
        />
      </div>
    {:else if error}
      <div class="empty">Failed to load: {error}</div>
    {:else if !allEvals.length}
      <div class="empty">No eval results yet.</div>
    {:else}
      <div class="eval-groups">
        {#each evalConfigGroups as group (group.key)}
          <section class="eval-group">
            <header class="eval-group-header">
              <div class="eval-group-title-wrap">
                <div class="eval-group-title">{group.meta.dataset}</div>
                <div class="eval-group-subtitle">{group.meta.model}</div>
              </div>
              <div class="eval-group-meta">
                {#if group.meta.split}
                  <span class="group-meta-pill">split: {group.meta.split}</span>
                {/if}
                {#if group.meta.judge}
                  <span class="group-meta-pill">judge: {group.meta.judge}</span>
                {/if}
                <span class="group-meta-pill">total evals: {group.totalEvals}</span>
                <span class="group-meta-pill">avg accuracy: {group.avgAccuracy.toFixed(4)}</span>
              </div>
            </header>

            <div class="table-wrap">
              <MinimalTable class="runs-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Eval run</th>
                    <th>Avg score</th>
                    <th>Delta vs best</th>
                    <th>Examples</th>
                    <th>Created</th>
                    <th>Links</th>
                  </tr>
                </thead>
                <tbody>
                  {#each group.runs as run (run.eval.eval_id)}
                    {@const cfg = run.eval.config || {}}
                    <tr>
                      <td>
                        <span class="rank-pill">#{run.rank}</span>
                      </td>
                      <td class="mono">{run.eval.eval_id}</td>
                      <td class="eval-score">
                        {run.avgScore.toFixed(4)}
                      </td>
                      <td class:delta-best={run.deltaFromBest === 0} class:delta-lower={run.deltaFromBest < 0}>
                        {run.deltaFromBest === 0 ? "best" : `${run.deltaFromBest > 0 ? "+" : ""}${run.deltaFromBest.toFixed(4)}`}
                      </td>
                      <td>{run.totalRows}</td>
                      <td><TimeAgo timestamp={run.eval.created_at} showJustNow /></td>
                      <td class="links-cell">
                        {#if cfg.deployment?.url}
                          <a class="pill-link" href={cfg.deployment.url} target="_blank" rel="noopener noreferrer">Endpoint</a>
                        {/if}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </MinimalTable>
            </div>
          </section>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  .summary-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-bottom: 24px;
  }

  .summary-card {
    border: 0;
    border-radius: 10px;
    background: color-mix(in srgb, #ffffff 7%, transparent);
    padding: 0.8rem 0.95rem;
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
  }

  .summary-label {
    color: var(--muted);
    font-size: 0.74rem;
    text-transform: none;
  }

  .summary-card strong {
    color: var(--text-bright);
    font-size: 1.36rem;
    font-weight: 600;
  }

  .runs-surface {
    border: 0;
    background: transparent;
  }

  .runs-body {
    padding: 0;
  }

  .empty {
    padding: 2.5rem 2rem;
    color: var(--muted);
    text-align: center;
    font-size: 0.84rem;
  }

  .eval-groups {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    padding: 0;
  }

  .eval-group {
    background: transparent;
  }

  .eval-group-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.8rem;
    padding: 0 0.2rem 0.45rem;
  }

  .eval-group-title-wrap {
    min-width: 0;
  }

  .eval-group-title {
    color: var(--text-bright);
    font-size: 0.86rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .eval-group-subtitle {
    color: var(--muted);
    font-size: 0.74rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .eval-group-meta {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  .group-meta-pill {
    border: 0;
    border-radius: 6px;
    padding: 0.11rem 0.42rem;
    font-size: 0.68rem;
    color: var(--muted);
    background: color-mix(in srgb, var(--panel-alt) 70%, transparent);
  }

  .table-wrap {
    overflow-x: auto;
  }

  :global(table.runs-table) {
    width: 100%;
    min-width: 1080px;
  }

  .rank-pill {
    display: inline-block;
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    border-radius: 9999px;
    padding: 0.08rem 0.48rem;
    color: var(--accent);
    font-size: 0.68rem;
    font-weight: 600;
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }

  .mono {
    font-family: var(--font-mono);
    color: color-mix(in srgb, var(--accent) 78%, white);
    font-size: 0.72rem;
  }

  .eval-score {
    color: var(--text-bright);
    font-weight: 600;
  }

  .delta-best {
    color: var(--green);
    font-weight: 600;
  }

  .delta-lower {
    color: var(--yellow);
  }

  .links-cell {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .pill-link {
    display: inline-block;
    border-radius: 6px;
    padding: 0.14rem 0.46rem;
    text-decoration: none;
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    font-size: 0.68rem;
    font-weight: 500;
  }

  .pill-link:hover {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
  }

  @media (max-width: 1080px) {
    .summary-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 900px) {
    .summary-row {
      grid-template-columns: 1fr;
    }
  }
</style>
