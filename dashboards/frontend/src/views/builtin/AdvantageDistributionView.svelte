<script>
  import { onMount } from "svelte";
  import AdvantageSpreadChart from "../../components/AdvantageSpreadChart.svelte";
  import { fetchRunAdvantages } from "../../lib/api.js";
  let { runId } = $props();
  let steps = $state([]);
  onMount(async () => { steps = await fetchRunAdvantages(runId); });
</script>

<AdvantageSpreadChart {steps} />
