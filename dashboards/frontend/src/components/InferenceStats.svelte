<script>
  let { inference = null } = $props();

  function formatCount(value) {
    return Number.isFinite(value) ? Math.round(value).toLocaleString() : "—";
  }

  function formatPercent(value) {
    return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  }

  let stats = $derived.by(() => {
    if (!inference) return [];
    const items = [];
    if (Number.isFinite(inference.tokens_in)) {
      items.push({ label: "tokens in", value: formatCount(inference.tokens_in) });
    }
    if (Number.isFinite(inference.tokens_out)) {
      items.push({ label: "tokens out", value: formatCount(inference.tokens_out) });
    }
    if (Number.isFinite(inference.cached_tokens)) {
      items.push({
        label: "cached tokens",
        value: formatCount(inference.cached_tokens),
        suffix: `(${formatPercent(inference.cache_hit_rate)})`,
      });
    }
    if (Number.isFinite(inference.new_tokens)) {
      items.push({ label: "new tokens", value: formatCount(inference.new_tokens) });
    }
    return items;
  });
</script>

{#if stats.length}
  <div
    class="grid grid-cols-[repeat(auto-fit,minmax(110px,1fr))] max-[900px]:grid-cols-1 gap-[6px] mb-[10px]"
    aria-label="Inference statistics"
  >
    {#each stats as stat (stat.label)}
      <div class="rounded-[4px] [border:1px_solid_var(--border)] p-[6px_8px]">
        <div class="text-[10px] uppercase tracking-[0.05em] text-(--muted)">{stat.label}</div>
        <div class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
          {stat.value}
          {#if stat.suffix}<span class="text-(--muted)">{stat.suffix}</span>{/if}
        </div>
      </div>
    {/each}
  </div>
{/if}
