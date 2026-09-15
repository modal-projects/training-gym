<script>
  // Port of the Modal dashboard's `DropdownItem.svelte` (button variant).
  import Check from "lucide-svelte/icons/check";
  import { getContext } from "svelte";

  import { DROPDOWN_CONTEXT } from "./Dropdown.svelte";

  let {
    selected = false,
    subtitle = "",
    disabled = false,
    icon,
    trailing,
    children,
    onclick,
    // Needs to be passed if you want to support keyboard navigation of dropdown items.
    itemIndex,
  } = $props();

  let dropdownContext = getContext(DROPDOWN_CONTEXT);

  const onmouseover = () => {
    if (itemIndex === undefined) return;
    if (dropdownContext.lastNavType === "key") return;
    dropdownContext.focusedIndex = itemIndex;
  };

  let isFocused = $derived(
    itemIndex !== undefined && dropdownContext.focusedIndex === itemIndex,
  );

  let actionEl = $state(null);

  const scrollIntoView = (block) => {
    actionEl?.scrollIntoView({ block, behavior: "auto" });
  };

  $effect(() => {
    if (!isFocused) return;
    scrollIntoView("nearest");
    const keyHandler = (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      e.stopImmediatePropagation();
      actionEl?.click();
      dropdownContext.closeMenu?.();
    };
    window.addEventListener("keydown", keyHandler, true);
    return () => window.removeEventListener("keydown", keyHandler, true);
  });

  $effect(() => {
    if (!selected) return;
    if (itemIndex === undefined) return;
    dropdownContext.focusedIndex = itemIndex;
    scrollIntoView("center");
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_mouse_events_have_key_events -->
<div {onmouseover}>
  <button
    type="button"
    role="menuitem"
    bind:this={actionEl}
    class={[
      "menu-item",
      disabled && "disabled",
      selected && "selected",
      isFocused && "menu-item-focused",
      itemIndex === undefined && "menu-item-without-index",
    ]}
    {disabled}
    onclick={() => {
      if (disabled) return;
      onclick?.();
    }}
  >
    {@render icon?.()}
    <div class="menu-item-body">
      <span class="label">{@render children?.()}</span>
      {#if subtitle}
        <p class="subtitle">{subtitle}</p>
      {/if}
    </div>
    {#if selected || trailing}
      <div class="menu-item-trailing">
        {@render trailing?.()}
        {#if selected}
          <Check size={18} class="text-(--accent)" />
        {/if}
      </div>
    {/if}
  </button>
</div>

<style>
  .menu-item-body {
    min-width: 0;
    overflow: hidden;
  }

  .menu-item-trailing {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
  }
</style>
