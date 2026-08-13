<script>
  import { RefreshCw } from "lucide-svelte";

  let { title, statusText, refreshing = false, onRefresh } = $props();

  // Spin while a fetch is in flight, with a short tail after it finishes so a
  // fast refresh still completes a smooth rotation instead of jerking to a stop.
  let spinning = $state(false);
  $effect(() => {
    if (refreshing) {
      spinning = true;
      return;
    }
    if (!spinning) return;
    const t = setTimeout(() => (spinning = false), 500);
    return () => clearTimeout(t);
  });
</script>

<header class="flex justify-between items-center gap-[12px] p-[16px_24px_0] mb-[24px] max-[900px]:p-[16px_16px_0] max-[900px]:mb-[16px]">
  <h1 class="text-(--text) text-[24px] font-medium leading-[36px] min-w-0 max-[900px]:text-[20px] max-[900px]:leading-[28px]">{title}</h1>
  <div class="flex items-center gap-[0.7rem] flex-[0_0_auto]">
    {#if statusText}
      <span class="text-(--muted) text-[0.76rem] lowercase max-[520px]:hidden">{statusText}</span>
    {/if}
    <button class="[border:1px_solid_var(--border-strong)] rounded-[6px] text-(--text) bg-(--bg) [font:inherit] text-[14px] font-medium p-[6px_8px] cursor-pointer inline-flex items-center gap-[8px] min-h-[36px] hover:[border-color:var(--color-c-gray-50,#8b8b8b)] hover:text-(--text-bright)" onclick={onRefresh} aria-label="Refresh">
      <span class="refresh-icon" class:spinning>
        <RefreshCw size={16} strokeWidth={2.1} />
      </span>
      <span class="max-[520px]:hidden">Refresh</span>
    </button>
  </div>
</header>
