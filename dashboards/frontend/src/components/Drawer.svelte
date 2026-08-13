<script>
  import { PanelRightClose } from "lucide-svelte";

  let {
    open = false,
    showCloseButton = false,
    onclose = () => {},
    width = "min(420px, calc(100vw - 24px))",
    minWidth,
  } = $props();

  $effect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  });
</script>

<svelte:window
  onkeydown={(event) => {
    if (open && event.key === "Escape") onclose();
  }}
/>

{#if open}
  <div class="fixed inset-0 z-[40] flex justify-end [background:rgba(0,0,0,0.35)]">
    <button
      type="button"
      class="absolute inset-0 [border:0] [background:transparent] cursor-default"
      onclick={onclose}
      aria-label="Close drawer"
    ></button>
    <div
      class="relative z-[1] bg-(--color-c-gray-2,#1c1c1c) [border-left:1px_solid_var(--color-c-gray-10,#2f2f2f)] h-full max-w-[100vw] [box-shadow:0_0_32px_6px_rgba(0,0,0,0.4)] max-[540px]:w-full! max-[540px]:min-w-0! max-[540px]:[border-left:0]"
      style:width={width}
      style:min-width={minWidth ? `${minWidth}px` : undefined}
      role="dialog"
      aria-modal="true"
    >
      <div class="h-full overflow-y-auto overflow-x-hidden overscroll-contain">
        {#if showCloseButton}
          <button
            type="button"
            class="[border:1px_solid_var(--border)] rounded-[6px] [background:transparent] text-(--muted) cursor-pointer inline-flex items-center justify-center p-[0.2rem] m-[0.8rem_0.8rem_0_0] float-right ghost-hover min-h-[36px] min-w-[36px]"
            onclick={onclose}
            aria-label="Close drawer"
          >
            <PanelRightClose size={20} />
          </button>
        {/if}
        <!-- svelte-ignore slot_element_deprecated -->
        <slot />
      </div>
    </div>
  </div>
{/if}
