<script>
  import RunRow from "./RunRow.svelte";

  let { framework, runs, deployments = [] } = $props();
  let collapsed = $state(false);

  let completedCount = $derived(
    runs.filter((r) => r.train_result != null).length,
  );
</script>

<section class="framework-section" class:collapsed>
  <header
    class="framework-header"
    onclick={() => (collapsed = !collapsed)}
    role="button"
    tabindex="0"
    onkeydown={(e) => e.key === "Enter" && (collapsed = !collapsed)}
  >
    <div class="framework-left">
      <span class="collapse-arrow">&#9662;</span>
      <span class="framework-dot"></span>
      <span class="framework-name">{framework}</span>
      <span class="framework-count"
        >{runs.length} run{runs.length === 1 ? "" : "s"}</span
      >
    </div>
    <div class="framework-stats">
      {#if completedCount > 0}
        <span class="stat-completed">{completedCount} completed</span>
      {/if}
      {#if completedCount < runs.length}
        <span class="stat-pending">{runs.length - completedCount} pending</span>
      {/if}
    </div>
  </header>

  {#if !collapsed}
    <div class="fw-body">
      <table>
        <thead>
          <tr>
            <th>Training Run</th>
            <th>Training Run ID</th>
            <th>Model</th>
            <th>Cluster</th>
            <th>Config</th>
            <th>TrainResult</th>
            <th>Deployment</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody>
          {#each runs as run (run.run_id)}
            <RunRow {run} {deployments} />
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</section>

<style>
  .framework-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.7rem;
    overflow: hidden;
  }
  .framework-header {
    padding: 0.55rem 0.85rem;
    background: var(--panel-alt);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
  }
  .framework-header:hover {
    background: var(--surface);
  }
  .framework-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .collapse-arrow {
    color: var(--muted-strong);
    font-size: 0.7em;
    transition: transform 0.2s;
  }
  .collapsed .collapse-arrow {
    transform: rotate(-90deg);
  }
  .framework-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 9999px;
    background: var(--accent);
    box-shadow: 0 0 0.5rem color-mix(in srgb, var(--accent) 48%, transparent);
  }
  .framework-name {
    font-weight: 600;
    color: var(--text-bright);
    text-transform: lowercase;
    font-size: 0.86rem;
  }
  .framework-count {
    color: var(--muted-strong);
    font-size: 0.76rem;
  }
  .framework-stats {
    display: flex;
    gap: 0.7rem;
    font-size: 0.75rem;
  }
  .stat-completed {
    color: var(--green);
  }
  .stat-pending {
    color: var(--yellow);
  }
  .fw-body {
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    min-width: 1260px;
  }
  th {
    padding: 0.45rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-weight: 500;
    color: var(--muted-strong);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: color-mix(in srgb, var(--panel-alt) 88%, black);
  }
</style>
