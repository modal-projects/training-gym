<script>
  import { onMount } from "svelte";
  import StepTimings from "../../components/StepTimings.svelte";
  import { fetchRun } from "../../lib/api.js";
  let { runId } = $props();
  let run = $state(null);
  onMount(async () => { run = await fetchRun(runId); });
</script>

{#if run?.step_times || run?.substep_times}
  <StepTimings stepTimes={run.step_times} substepTimes={run.substep_times} layout="timeline" downloadName={`step_substep_times_${runId}.json`} />
{/if}
