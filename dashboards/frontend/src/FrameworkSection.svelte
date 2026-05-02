<script>
  import RunRow from "./RunRow.svelte";

  let { framework, runs } = $props();
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
      <span class="collapse-arrow">&#9660;</span>
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
            <th>Run ID</th>
            <th>Model</th>
            <th>Cluster</th>
            <th>Config</th>
            <th>Result</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {#each runs as run (run.run_id)}
            <RunRow {run} />
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
    border-radius: 8px;
    margin-bottom: 1rem;
    overflow: hidden;
  }
  .framework-header {
    padding: 0.65rem 1rem;
    background: #1a1f2a;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
  }
  .framework-header:hover {
    background: #1e2430;
  }
  .framework-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .collapse-arrow {
    color: var(--muted);
    font-size: 0.7em;
    transition: transform 0.2s;
  }
  .collapsed .collapse-arrow {
    transform: rotate(-90deg);
  }
  .framework-name {
    font-weight: 600;
    color: var(--accent);
  }
  .framework-count {
    color: var(--muted);
    font-size: 0.85em;
  }
  .framework-stats {
    display: flex;
    gap: 0.75rem;
    font-size: 0.8em;
  }
  .stat-completed {
    color: var(--green);
  }
  .stat-pending {
    color: var(--yellow);
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th {
    padding: 0.55rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-weight: 500;
    color: var(--muted);
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
</style>
