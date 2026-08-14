<script>
  import { onMount } from "svelte";
  import ComparativeBarChart from "../../components/ComparativeBarChart.svelte";
  import { fetchRunRollouts, fetchRollout } from "../../lib/api.js";
  let { runId } = $props();
  let series = $state([]);
  onMount(async () => {
    const rows = await fetchRunRollouts(runId);
    if (!rows.length) return;
    const first = await fetchRollout(runId, rows[0].rollout_id);
    const last = await fetchRollout(runId, rows.at(-1).rollout_id);
    const score = (payload) => (payload?.samples || []).map((sample) => Number(sample.score ?? sample.reward ?? 0));
    series = [{ name: "first", values: score(first) }, { name: "latest", values: score(last) }];
  });
</script>

{#if series.length}<ComparativeBarChart categories={series[0].values.map((_, index) => String(index + 1))} {series} />{/if}
