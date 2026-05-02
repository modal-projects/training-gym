<script>
  import { fetchWandbMetrics } from "./lib/api.js";

  let { trainingRunId, keys = "train/loss,train/reward" } = $props();
  let metrics = $state(null);
  let error = $state(null);
  let loading = $state(true);

  $effect(() => {
    if (!trainingRunId) return;
    loading = true;
    fetchWandbMetrics(trainingRunId, keys, 100)
      .then((data) => {
        metrics = data;
        loading = false;
      })
      .catch((e) => {
        error = e.message;
        loading = false;
      });
  });

  function sparkline(points, key, width = 120, height = 32) {
    const vals = points
      .map((p) => p[key])
      .filter((v) => v != null && !isNaN(v));
    if (vals.length < 2) return "";
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    const step = width / (vals.length - 1);
    const coords = vals.map(
      (v, i) => `${i * step},${height - ((v - min) / range) * height}`
    );
    return `M${coords.join("L")}`;
  }

  function lastValue(points, key) {
    for (let i = points.length - 1; i >= 0; i--) {
      const v = points[i][key];
      if (v != null && !isNaN(v)) return v.toFixed(4);
    }
    return "—";
  }

  let keyList = $derived(keys.split(",").map((k) => k.trim()).filter(Boolean));
</script>

{#if loading}
  <div class="metrics-loading">Loading metrics…</div>
{:else if error || !metrics?.metrics?.length}
  <div class="metrics-empty">No W&B metrics</div>
{:else}
  <div class="metrics-row">
    {#each keyList as key}
      <div class="metric-card">
        <div class="metric-label">{key.split("/").pop()}</div>
        <div class="metric-chart">
          <svg viewBox="0 0 120 32" preserveAspectRatio="none">
            <path
              d={sparkline(metrics.metrics, key)}
              fill="none"
              stroke="var(--accent)"
              stroke-width="1.5"
            />
          </svg>
        </div>
        <div class="metric-value">{lastValue(metrics.metrics, key)}</div>
      </div>
    {/each}
    {#if metrics.wandb_url}
      <a class="wandb-link" href={metrics.wandb_url} target="_blank" rel="noopener">
        W&B
      </a>
    {/if}
  </div>
{/if}

<style>
  .metrics-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.25rem 0;
  }
  .metric-card {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.8em;
  }
  .metric-label {
    color: var(--muted);
    font-weight: 500;
    min-width: 3em;
  }
  .metric-chart {
    width: 80px;
    height: 24px;
  }
  .metric-chart svg {
    width: 100%;
    height: 100%;
  }
  .metric-value {
    color: var(--text);
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.85em;
    min-width: 4em;
    text-align: right;
  }
  .metrics-loading,
  .metrics-empty {
    font-size: 0.8em;
    color: var(--muted);
    padding: 0.2rem 0;
  }
  .wandb-link {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.1);
    text-decoration: none;
  }
  .wandb-link:hover {
    background: rgba(251, 191, 36, 0.2);
  }
</style>
