<script>
  import RunRow from "./RunRow.svelte";

  let { framework, runs, deployments = [] } = $props();
  let collapsed = $state(false);

  let completedCount = $derived(
    runs.filter((r) => r.train_result != null).length,
  );
  let stoppedCount = $derived(
    runs.filter((r) => r.status === "stopped").length,
  );
  let failedCount = $derived(
    runs.filter((r) => r.status === "failed").length,
  );
  let runningCount = $derived(runs.length - completedCount - stoppedCount - failedCount);
</script>

<section class="framework-section" class:collapsed>
  <header
    class="p-[0.55rem_0.85rem] bg-(--panel-alt) [border-bottom:1px_solid_var(--border)] flex justify-between items-center cursor-pointer [user-select:none] hover:bg-(--surface)"
    onclick={() => (collapsed = !collapsed)}
    role="button"
    tabindex="0"
    onkeydown={(e) => e.key === "Enter" && (collapsed = !collapsed)}
  >
    <div class="flex items-center gap-[0.5rem]">
      <span class="collapse-arrow">&#9662;</span>
      <span class="w-[0.5rem] h-[0.5rem] rounded-[9999px] bg-(--accent) [box-shadow:0_0_0.5rem_color-mix(in_srgb,var(--accent)_48%,transparent)]"></span>
      <span class="font-[600] text-(--text-bright) lowercase text-[0.86rem]">{framework}</span>
      <span class="text-(--muted-strong) text-[0.76rem]"
        >{runs.length} run{runs.length === 1 ? "" : "s"}</span
      >
    </div>
    <div class="flex gap-[0.7rem] text-[0.75rem]">
      {#if completedCount > 0}
        <span class="text-(--green)">{completedCount} completed</span>
      {/if}
      {#if runningCount > 0}
        <span class="text-(--yellow)">{runningCount} running</span>
      {/if}
      {#if stoppedCount > 0}
        <span class="text-[#fb923c]">{stoppedCount} stopped</span>
      {/if}
      {#if failedCount > 0}
        <span class="text-[#f87171]">{failedCount} failed</span>
      {/if}
    </div>
  </header>

  {#if !collapsed}
    <div class="overflow-x-auto">
      <table class="w-full border-collapse min-w-[1260px]">
        <thead>
          <tr>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Training Run</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Training Run ID</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Model</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Cluster</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Config</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">TrainResult</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Deployment</th>
            <th class="p-[0.45rem_0.8rem] text-left [border-bottom:1px_solid_var(--border)] font-medium text-(--muted-strong) text-[0.7rem] uppercase tracking-[0.08em] bg-[color-mix(in_srgb,var(--panel-alt)_88%,black)]">Links</th>
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
