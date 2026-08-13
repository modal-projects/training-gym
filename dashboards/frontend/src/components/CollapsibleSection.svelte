<script>
  import { ChevronUp } from "lucide-svelte";
  import { slide } from "svelte/transition";

  // Port of Modal's CollapsibleDrawerSection: a top-bordered panel with a
  // clickable header (chevron rotates when open) and a slide-animated body.
  // `title` and `body` are snippets so callers control the header content.
  let { isOpen = $bindable(true), title, body } = $props();
</script>

<div class="[border-top:1px_solid_var(--color-c-gray-10,#2f2f2f)]">
  <button
    class="flex w-full items-center justify-between gap-[16px] p-[12px_0] [border:0] [background:none] text-inherit [font:inherit] cursor-pointer text-left"
    onclick={() => (isOpen = !isOpen)}
    aria-expanded={isOpen}
  >
    <div class="flex-1 min-w-0">{@render title()}</div>
    <div class="collapsible-chevron" class:collapsible-open={isOpen}>
      <ChevronUp size={18} />
    </div>
  </button>

  {#if isOpen}
    <div class="pb-[16px]" transition:slide={{ duration: 200 }}>
      {@render body()}
    </div>
  {/if}
</div>
