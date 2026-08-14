<script>
  let { component, props = {} } = $props();
  let failed = $state(null);

  function recover() {
    failed = null;
  }
</script>

{#if failed}
  <div class="detail-empty" role="alert">
    <div>View failed: {failed?.message || failed}</div>
    <button type="button" class="log-button mt-[8px]" onclick={recover}>Retry</button>
  </div>
{:else if component}
  <svelte:boundary
    onerror={(error) => {
      failed = error;
    }}
  >
    {@const View = component}
    <View {...props} />
  </svelte:boundary>
{:else}
  <div class="detail-empty" role="alert">View is unavailable.</div>
{/if}
