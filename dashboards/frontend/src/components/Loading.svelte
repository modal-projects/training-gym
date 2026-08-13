<script>
  import { Loader } from "lucide-svelte";

  let { class: className = "", size = 14 } = $props();
</script>

<Loader
  class={className}
  {size}
  style="animation: loading-spin 1s linear infinite;"
/>
