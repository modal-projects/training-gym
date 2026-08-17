<script>
  let { data = null, title = "JSON" } = $props();
  let expanded = $state(false);
  const serialized = $derived(JSON.stringify(data, null, 2));
</script>

<section class="rounded-[8px] border border-(--color-c-gray-10) bg-(--color-c-gray-5) p-[12px]">
  <button
    type="button"
    class="text-(--text-bright) text-[12px] font-medium"
    onclick={() => (expanded = !expanded)}
  >
    {expanded ? "Hide" : "Show"} {title}
  </button>
  {#if expanded}
    <pre class="mt-[8px] max-h-[420px] overflow-auto whitespace-pre-wrap text-[12px] text-(--text)">{serialized}</pre>
  {/if}
</section>
