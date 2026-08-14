<script>
  import { onMount } from "svelte";
  import { compileAuthoredView } from "../lib/views/compile.js";

  let { source = "", props = {} } = $props();
  let component = $state(null);
  let error = $state("");
  let loading = $state(true);

  onMount(() => {
    let disposed = false;
    loading = true;
    error = "";
    compileAuthoredView(source)
      .then((compiled) => {
        if (!disposed) component = compiled;
      })
      .catch((reason) => {
        if (!disposed) error = reason?.message || String(reason);
      })
      .finally(() => {
        if (!disposed) loading = false;
      });
    return () => {
      disposed = true;
    };
  });
</script>

{#if loading}
  <div class="detail-empty">Compiling authored view…</div>
{:else if error}
  <div class="detail-empty" role="alert">
    <div>Authored view failed: {error}</div>
  </div>
{:else if component}
  {@const View = component}
  <View {...props} />
{:else}
  <div class="detail-empty">Authored view is unavailable.</div>
{/if}
