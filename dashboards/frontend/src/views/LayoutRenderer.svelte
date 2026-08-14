<script>
  import { onMount } from "svelte";
  import { builtinViews, resolveLayout, resolveView } from "./registry.js";
  import ViewBoundary from "./ViewBoundary.svelte";
  import JsonView from "./builtin/JsonView.svelte";
  import Tabs from "../components/Tabs.svelte";

  let {
    layoutId = "training-run.default",
    context = {},
    views = [],
    layouts = [],
    viewProps = {},
    safe = false,
  } = $props();
  let remoteViews = $state([]);
  let remoteLayouts = $state([]);

  onMount(async () => {
    function readTab() {
      return new URLSearchParams(window.location.search).get("tab");
    }
    function syncTab() {
      const requested = readTab();
      if (layout?.tabs?.some((tab) => tab.id === requested)) activeTab = requested;
    }
    syncTab();
    window.addEventListener("popstate", syncTab);
    remoteViews = views;
    remoteLayouts = layouts;
    if (safe) {
      return () => window.removeEventListener("popstate", syncTab);
    }
    try {
      const [viewResponse, layoutResponse] = await Promise.all([
        fetch("/api/ui/views"),
        fetch("/api/ui/layouts"),
      ]);
      if (viewResponse.ok) remoteViews = await viewResponse.json();
      if (layoutResponse.ok) remoteLayouts = await layoutResponse.json();
    } catch {
      // Builtins remain usable when the metadata volume is unavailable.
    }
    return () => window.removeEventListener("popstate", syncTab);
  });

  let layout = $derived(resolveLayout(remoteLayouts, layoutId, context));
  let activeTab = $state(null);
  let selectedTab = $derived(activeTab || layout?.tabs?.[0]?.id);

  function selectTab(id) {
    activeTab = id;
    const url = new URL(window.location.href);
    url.searchParams.set("tab", id);
    window.history.pushState({}, "", url);
  }

  function viewFor(entry) {
    const doc = resolveView(remoteViews, entry.view, context);
    if (!doc) return null;
    if (doc.code) return JsonView;
    return builtinViews[doc.component] || JsonView;
  }

  function propsFor(entry) {
    return {
      ...viewProps,
      config: { ...(resolveView(remoteViews, entry.view, context)?.config || {}), ...(entry.config || {}) },
      // Layout capabilities are the only shared channel between views.
      // Views may use navigation and run context values, but never another
      // view's private state.
      context: { ...context, layout: { selectTab, navigate: (path) => window.history.pushState({}, "", path) } },
      data: null,
      navigate: (path) => window.history.pushState({}, "", path),
    };
  }
</script>

  {#if layout}
    <div class="view-layout">
      {#each layout.tabs?.filter((tab) => tab.id === selectedTab) || [] as tab (tab.id)}
        {#if tab.slots?.header}
          <div class="view-slot view-slot-header">
            {#each tab.slots.header.filter((entry) => !entry.hidden) as entry, index (entry.id ?? index)}
              {#if viewFor(entry)}
                <ViewBoundary component={viewFor(entry)} props={propsFor(entry)} />
              {/if}
            {/each}
          </div>
        {/if}
        {#if layout.tabs?.length > 1}
          <Tabs
            active={selectedTab}
            onSelect={selectTab}
            tabs={layout.tabs.map((item) => ({ value: item.id, label: item.label, count: item.count }))}
          />
        {/if}
        <div class="view-layout-grid">
          {#each Object.entries(tab.slots || {}).filter(([slot]) => slot !== "header") as [slot, entries] (slot)}
            <div class={`view-slot view-slot-${slot}`}>
              {#each entries.filter((entry) => !entry.hidden) as entry, index (entry.id ?? index)}
                {#if viewFor(entry)}
                  <ViewBoundary component={viewFor(entry)} props={propsFor(entry)} />
                {/if}
              {/each}
            </div>
          {/each}
        </div>
      {/each}
    </div>
{:else}
  <JsonView data={{ error: "Layout not found", layoutId }} />
{/if}
