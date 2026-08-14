<script>
  import { onMount } from "svelte";
  import LineChart from "../../components/LineChart.svelte";
  import { fetchRunRollouts } from "../../lib/api.js";
  let { runId } = $props();
  let rows = $state([]);
  onMount(async () => { rows = await fetchRunRollouts(runId); });
  let tags = $derived(Array.from(new Set(rows.flatMap((row) => Object.keys(row.tag_stats || {})))).sort());
</script>

{#each tags as tag (tag)}
  <div class="rollout-chart">
    <div class="rollout-chart-title">{tag}</div>
    <LineChart
      title={tag}
      data={rows.filter((row) => row.tag_stats?.[tag]).map((row) => ({ x: row.rollout_id, y: row.tag_stats[tag].mean }))}
      height={150}
    />
  </div>
{/each}
