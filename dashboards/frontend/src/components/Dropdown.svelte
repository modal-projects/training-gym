<script module>
  export const DROPDOWN_CONTEXT = Symbol("dropdown-context");
</script>

<script>
  // Port of the Modal dashboard's `Dropdown.svelte`, without the Melt UI
  // positioning layer: the menu is anchored below the trigger with plain CSS.
  import { setContext, untrack } from "svelte";

  let {
    align = "left",
    open = $bindable(false),
    closeOnInnerClick = true,
    button,
    children,
    // Needs to be passed if you want to support keyboard navigation of dropdown items.
    elements,
    contentMaxHeightPx = 420,
  } = $props();

  let rootEl = $state(null);
  let menuElement = $state(null);

  let context = $state({
    focusedIndex: 0,
    closeMenu,
    lastNavType: null,
  });
  setContext(DROPDOWN_CONTEXT, context);

  function closeMenu() {
    open = false;
    context.lastNavType = null;
  }

  function toggle() {
    open = !open;
    if (!open) context.lastNavType = null;
  }

  const handleClick = (e) => {
    if (!open) return;
    const target = e.target;
    if (!(target instanceof Element)) return;
    if (menuElement && menuElement.contains(target)) {
      if (target.closest("button, a") !== null && closeOnInnerClick) closeMenu();
      return;
    }
    if (rootEl && rootEl.contains(target)) return;
    closeMenu();
  };

  $effect(() => {
    if (!open) return;
    const timeoutId = setTimeout(() => {
      document.body.addEventListener("click", handleClick);
    });
    return () => {
      clearTimeout(timeoutId);
      document.body.removeEventListener("click", handleClick);
    };
  });

  $effect(() => {
    if (!open) return;
    if (context.lastNavType === "mouse") return;
    const listener = () => {
      context.lastNavType = "mouse";
    };
    window.addEventListener("mousemove", listener);
    return () => window.removeEventListener("mousemove", listener);
  });

  $effect(() => {
    if (!open) return;
    const count = elements?.length ?? 0;
    const keyHandler = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeMenu();
      } else if (count && e.key === "ArrowDown") {
        e.preventDefault();
        context.focusedIndex = (context.focusedIndex + 1) % count;
        context.lastNavType = "key";
      } else if (count && e.key === "ArrowUp") {
        e.preventDefault();
        context.focusedIndex = (context.focusedIndex - 1 + count) % count;
        context.lastNavType = "key";
      }
    };
    window.addEventListener("keydown", keyHandler);
    return () => window.removeEventListener("keydown", keyHandler);
  });

  let elementCount = $derived(elements?.length ?? 0);
  $effect(() => {
    elementCount;
    untrack(() => {
      context.focusedIndex = 0;
    });
  });
</script>

<div class="dropdown-root" bind:this={rootEl}>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <span class="dropdown-trigger" onclick={toggle} aria-haspopup="menu" aria-expanded={open}>
    {@render button?.()}
  </span>

  {#if open}
    <div
      class={["dropdown-menu", `dropdown-menu-${align}`]}
      role="menu"
      bind:this={menuElement}
    >
      <div class="menu-items">
        <div class="dropdown-scroll" style:max-height={`${contentMaxHeightPx}px`}>
          {@render children?.({ closeMenu })}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .dropdown-root {
    position: relative;
    display: inline-flex;
    min-width: 0;
  }

  .dropdown-trigger {
    display: flex;
    min-width: 0;
  }

  .dropdown-menu {
    position: absolute;
    top: calc(100% + 6px);
    z-index: 40;
    display: flex;
    flex-direction: column;
    min-width: 220px;
    outline: none;
  }

  .dropdown-menu-left {
    left: 0;
  }

  .dropdown-menu-right {
    right: 0;
  }

  .dropdown-menu-full {
    left: 0;
    right: 0;
  }

  .dropdown-scroll {
    min-height: 0;
    flex: 1;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 6px;
    scrollbar-width: none;
  }
</style>
