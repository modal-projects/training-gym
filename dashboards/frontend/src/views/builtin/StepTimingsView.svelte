<script>
  import { onMount } from "svelte";
  import { StepTimings } from "$host/components";
  import { fetchRun } from "$host/data";
  let { runId } = $props();
  let run = $state(null);
  onMount(async () => { run = await fetchRun(runId); });
</script>

{#if run?.step_times || run?.substep_times}
  <StepTimings stepTimes={run.step_times} substepTimes={run.substep_times} layout="timeline" downloadName={`step_substep_times_${runId}.json`} />
{/if}
