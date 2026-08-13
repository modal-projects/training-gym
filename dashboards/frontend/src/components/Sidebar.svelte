<script>
  let { navItems, activePage, onNavigate } = $props();
</script>

<aside
  class="[border-right:1px_solid_rgba(255,255,255,0.1)] bg-(--bg-depth) p-[4px_20px_0] max-[900px]:[border-right:0] max-[900px]:[border-bottom:1px_solid_var(--border)] max-[900px]:p-[0_8px]"
  aria-label="Primary"
>
  <nav
    class="flex flex-col gap-[4px] pt-[12px] max-[900px]:flex-row max-[900px]:gap-[4px] max-[900px]:pt-[6px] max-[900px]:pb-[6px] max-[900px]:overflow-x-auto max-[900px]:[-webkit-overflow-scrolling:touch]"
    aria-label="Sections"
  >
    {#each navItems as item (item.key)}
      <a
        href={item.path}
        class="nav-item max-[900px]:flex-[1_1_0] max-[900px]:justify-center max-[900px]:whitespace-nowrap max-[900px]:p-[10px_8px] max-[900px]:min-h-[44px] max-[900px]:text-[13px]"
        class:sidebar-active={activePage === item.key}
        aria-current={activePage === item.key ? "page" : undefined}
        onclick={(event) => {
          event.preventDefault();
          onNavigate(item.key);
        }}
      >
        <span class="inline-flex items-center flex-[0_0_auto] opacity-[0.6]">
          <item.Icon size={14} strokeWidth={2.1} />
        </span>
        <span>{item.label}</span>
      </a>
    {/each}
  </nav>
</aside>
