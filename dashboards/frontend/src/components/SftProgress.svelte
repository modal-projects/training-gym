<script>
  import LineChart from "./LineChart.svelte";
  import { fetchRunSteps } from "../lib/api.js";
  import { lossPoints, lossEmptyMessage } from "../lib/trainingType.js";

  let { run } = $props();
  let steps = $state([]);
  let loading = $state(true);
  let error = $state("");
  const number = (value) => Number.isFinite(value) ? value.toFixed(4) : "—";
  let points = $derived(lossPoints(steps));
  let latest = $derived(steps.at(-1));

  $effect(() => {
    const id = run.training_run_id;
    const running = run.status === "running";
    const controller = new AbortController();
    let busy = false;
    async function refresh() {
      if (busy) return;
      busy = true;
      try {
        const data = await fetchRunSteps(id, { signal: controller.signal });
        if (controller.signal.aborted) return;
        steps = data;
        error = "";
      } catch (err) {
        if (!controller.signal.aborted) error = `Reporting unavailable: ${err.message}. Showing any previously loaded data.`;
      } finally {
        busy = false;
        if (!controller.signal.aborted) loading = false;
      }
    }
    void refresh();
    const timer = running ? setInterval(refresh, 5000) : null;
    return () => { controller.abort(); if (timer) clearInterval(timer); };
  });
</script>

<section class="sft-progress" aria-label="SFT training progress">
  {#if error}<p role="status">{error}</p>{/if}
  {#if points.length}
    <LineChart title="Training loss" data={points} height={180} formatX={(row) => `Completed step ${row.x}`} formatY={number} ariaLabel="Assistant-token training loss by completed optimizer step" />
    <div class="stats">
      <span>Latest loss <strong>{number(latest?.loss)}</strong></span>
      <span>Gradient norm <strong>{number(latest?.grad_norm)}</strong></span>
      <span>Learning rate <strong>{Number.isFinite(latest?.learning_rate) ? latest.learning_rate.toExponential(2) : "—"}</strong></span>
      <span>{steps.length} recorded steps</span>
    </div>
  {:else if loading}
    <p>Loading training loss…</p>
  {:else if !error}
    <p>{lossEmptyMessage(run)}</p>
  {/if}
</section>

<style>
  .sft-progress { min-width: 0; margin-bottom: 24px; }
  p { color: var(--muted); font-size: 12px; line-height: 1.6; }
  .stats { display: flex; align-items: center; flex-wrap: wrap; gap: 12px 20px; margin: 12px 0; font-size: 12px; color: var(--muted); }
  strong { color: var(--text-bright); }
</style>
