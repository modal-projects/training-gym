<script>
  import { fmtDate, fmtDuration, extraTags } from "./lib/format.js";

  let { run } = $props();
  let extras = $derived(extraTags(run.tags));
  let isHarbor = $derived(
    run.tags?._modal_framework === "harbor" ||
      run.metadata?.framework === "harbor",
  );
</script>

<tr>
  <td class="app-id">{run.app_id}</td>
  <td>
    <div class="run-name">{run.name || "—"}</div>
    {#if run.description}
      <div class="run-desc">{run.description}</div>
    {/if}
  </td>
  <td>
    <span class="state-badge state-{run.state}">{run.state}</span>
  </td>
  <td>{fmtDate(run.created_at)}</td>
  <td class="duration">{fmtDuration(run.created_at, run.stopped_at)}</td>
  <td>
    {#if isHarbor}
      <a
        class="traj-link"
        href="#/harbor/{encodeURIComponent(run.name)}"
      >
        Trajectories
      </a>
    {/if}
    {#if extras.length}
      {#each extras as [key, value]}
        <span class="tag"><strong>{key}</strong>={value}</span>
      {/each}
    {:else if !isHarbor}
      —
    {/if}
  </td>
</tr>

<style>
  td {
    padding: 0.55rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  tr:last-child td {
    border-bottom: none;
  }
  .app-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85em;
    color: var(--accent);
  }
  .run-name {
    font-weight: 500;
  }
  .run-desc {
    color: var(--muted);
    font-size: 0.9em;
    margin-top: 0.15rem;
  }
  .state-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.8em;
    font-weight: 500;
  }
  :global(.state-Running),
  :global(.state-Deployed) {
    background: rgba(74, 222, 128, 0.12);
    color: var(--green);
  }
  :global(.state-Stopped),
  :global(.state-Stopping) {
    background: rgba(156, 163, 175, 0.12);
    color: var(--muted);
  }
  :global(.state-Ephemeral) {
    background: rgba(251, 191, 36, 0.12);
    color: var(--yellow);
  }
  :global(.state-Initializing) {
    background: rgba(125, 211, 252, 0.12);
    color: var(--accent);
  }
  :global(.state-Errored) {
    background: rgba(248, 113, 113, 0.12);
    color: var(--red);
  }
  .duration {
    color: var(--muted);
    font-size: 0.85em;
  }
  .tag {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    margin: 0.1rem 0.2rem 0.1rem 0;
    background: #1f2937;
    border-radius: 4px;
    font-size: 0.8em;
    color: var(--muted);
  }
  .tag strong {
    color: var(--text);
    font-weight: 500;
  }
  .traj-link {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin-right: 0.4rem;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
    color: var(--accent);
    background: var(--accent-dim);
    text-decoration: none;
    cursor: pointer;
  }
  .traj-link:hover {
    background: rgba(125, 211, 252, 0.2);
  }
</style>
