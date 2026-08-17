<script>
  import { onMount } from "svelte";
  import { AdvantageSpreadChart } from "$host/components";
  import { fetchRunAdvantages } from "$host/data";
  let { runId } = $props();
  let steps = $state([]);
  onMount(async () => { steps = await fetchRunAdvantages(runId); });
</script>

<AdvantageSpreadChart {steps} />
