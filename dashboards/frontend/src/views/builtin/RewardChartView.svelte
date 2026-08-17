<script>
  import { onMount } from "svelte";
  import { LineChart } from "$host/components";
  import { fetchRunRollouts } from "$host/data";
  let { runId } = $props();
  let rows = $state([]);
  onMount(() => {
    let stopped = false;
    const load = async () => {
      const next = await fetchRunRollouts(runId);
      if (!stopped) rows = next;
    };
    void load();
    const timer = window.setInterval(load, 5000);
    return () => { stopped = true; window.clearInterval(timer); };
  });
  let data = $derived(rows.map((row) => ({ x: row.rollout_id, y: row.mean })));
</script>

<div class="rollout-chart">
  <div class="rollout-chart-title">Reward over time</div>
  <LineChart title="Reward over time" {data} height={180} />
</div>
