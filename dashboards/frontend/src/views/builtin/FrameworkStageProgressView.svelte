<script>
  import { FrameworkStageProgress } from "$host/components";
  let { run = null } = $props();
  let progress = $derived.by(() => {
    const value = run?.framework_progress;
    if (!value || Number(value.total) <= 0) return null;
    return { current: Number(value.current) || 0, total: Number(value.total), unit: value.unit || "step" };
  });
</script>

{#if progress}
  <FrameworkStageProgress {progress} progressLabel={`${progress.unit} ${progress.current} / ${progress.total}`} stageLabel={run?.display_stage || ""} />
{/if}
