<script>
  import { PanelRightClose } from "lucide-svelte";

  let {
    open = false,
    showCloseButton = false,
    onclose = () => {},
    width = "min(420px, calc(100vw - 24px))",
    minWidth,
  } = $props();

</script>

<svelte:window
  onkeydown={(event) => {
    if (open && event.key === "Escape") onclose();
  }}
/>

{#if open}
  <div class="drawer-shell">
    <button class="drawer-overlay" onclick={onclose} aria-label="Close drawer"></button>
    <div
      class="drawer"
      style:width={width}
      style:min-width={minWidth ? `${minWidth}px` : undefined}
      role="dialog"
      aria-modal="true"
    >
      <div class="drawer-body">
        {#if showCloseButton}
          <button class="drawer-x" onclick={onclose} aria-label="Close drawer">
            <PanelRightClose size={20} />
          </button>
        {/if}
        <!-- svelte-ignore slot_element_deprecated -->
        <slot />
      </div>
    </div>
  </div>
{/if}

<style>
  .drawer-shell {
    position: fixed;
    inset: 0;
    z-index: 40;
    display: flex;
    justify-content: flex-end;
    background: transparent;
  }

  .drawer {
    position: relative;
    z-index: 1;
    background: var(--color-c-gray-2, #1c1c1c);
    border-left: 1px solid var(--color-c-gray-10, #2f2f2f);
    height: 100%;
    max-width: 100vw;
    box-shadow: 0 0 32px 6px rgba(0, 0, 0, 0.4);
  }

  .drawer-overlay {
    position: absolute;
    inset: 0;
    border: 0;
    background: transparent;
    cursor: default;
  }

  .drawer-body {
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
  }

  .drawer-x {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.2rem;
    margin: 0.8rem 0.8rem 0 0;
    float: right;
  }

  .drawer-x:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  @media (max-width: 540px) {
    .drawer {
      width: 100% !important;
      min-width: 0 !important;
      border-left: 0;
    }
  }
</style>
