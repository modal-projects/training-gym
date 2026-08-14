<script>
  import { onMount } from "svelte";
  import { builtinViews, resolveLayout, resolveView } from "./registry.js";
  import ViewBoundary from "./ViewBoundary.svelte";
  import AuthoredView from "./AuthoredView.svelte";
  import ViewEditor from "./ViewEditor.svelte";
  import LayoutEditor from "./LayoutEditor.svelte";
  import Drawer from "../components/Drawer.svelte";
  import JsonView from "./builtin/JsonView.svelte";
  import Tabs from "../components/Tabs.svelte";
  import { fetchRunRollouts } from "../lib/api.js";

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
  let editingDoc = $state(null);
  let editingLayout = $state(false);
  let jsonDoc = $state(null);
  let tabCounts = $state({});

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
    if (context.run_id) {
      try {
        const rows = await fetchRunRollouts(context.run_id);
        tabCounts = { rollouts: rows.length };
      } catch {
        tabCounts = {};
      }
    }
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
    if (doc.code) return AuthoredView;
    return builtinViews[doc.component] || JsonView;
  }

  function docFor(entry) {
    return resolveView(remoteViews, entry.view, context);
  }

  function propsFor(entry) {
    const doc = docFor(entry);
    return {
      ...viewProps,
      config: { ...(doc?.config || {}), ...(entry.config || {}) },
      // Layout capabilities are the only shared channel between views.
      // Views may use navigation and run context values, but never another
      // view's private state.
      context: { ...context, layout: { selectTab, navigate: (path) => window.history.pushState({}, "", path) } },
      data: null,
      navigate: (path) => window.history.pushState({}, "", path),
      source: doc?.code || "",
      props: {
        ...viewProps,
        config: { ...(doc?.config || {}), ...(entry.config || {}) },
        context: { ...context, layout: { selectTab, navigate: (path) => window.history.pushState({}, "", path) } },
        data: null,
        navigate: (path) => window.history.pushState({}, "", path),
      },
    };
  }

  function openEditor(entry) {
    editingDoc = docFor(entry);
  }

  function openJson(entry) {
    jsonDoc = docFor(entry);
  }

  async function resetView(entry) {
    const doc = docFor(entry);
    if (!doc || doc.scope === "builtin") return;
    const response = await fetch(`/api/ui/views/${doc.scope}/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
    if (response.ok) {
      remoteViews = remoteViews.filter((item) => !(item.scope === doc.scope && item.id === doc.id));
    }
  }

  function closeEditor() {
    editingDoc = null;
  }

  function handleSaved(doc) {
    remoteViews = [...remoteViews.filter((item) => !(item.scope === doc.scope && item.id === doc.id)), doc];
    editingDoc = doc;
  }

  function handleLayoutSaved(doc) {
    remoteLayouts = [...remoteLayouts.filter((item) => !(item.scope === doc.scope && item.id === doc.id)), doc];
    editingLayout = false;
  }
</script>

  {#if layout}
    <div class="view-layout">
      {#each layout.tabs?.filter((tab) => tab.id === selectedTab) || [] as tab (tab.id)}
        {#if tab.slots?.header}
          <div class="view-slot view-slot-header">
            {#each tab.slots.header.filter((entry) => !entry.hidden) as entry, index (entry.id ?? index)}
              {#if viewFor(entry)}
                <div class="view-instance">
                  <div class="view-instance-actions">
                    <button type="button" onclick={() => openEditor(entry)}>Edit</button>
                    <button type="button" onclick={() => resetView(entry)}>Reset</button>
                    <button type="button" onclick={() => openJson(entry)}>JSON</button>
                  </div>
                  <ViewBoundary component={viewFor(entry)} props={propsFor(entry)} />
                </div>
              {/if}
            {/each}
          </div>
        {/if}
        {#if layout.tabs?.length > 1}
          <div class="layout-toolbar">
            <Tabs
              active={selectedTab}
              onSelect={selectTab}
              tabs={layout.tabs.map((item) => ({
                value: item.id,
                label: item.label,
                count: item.count ?? tabCounts[item.id],
              }))}
            />
            <button class="layout-edit-button" type="button" onclick={() => (editingLayout = true)}>Edit layout</button>
          </div>
        {/if}
        <div class="view-layout-grid">
          {#each Object.entries(tab.slots || {}).filter(([slot]) => slot !== "header") as [slot, entries] (slot)}
            <div class={`view-slot view-slot-${slot}`}>
              {#each entries.filter((entry) => !entry.hidden) as entry, index (entry.id ?? index)}
                {#if viewFor(entry)}
                  <div class="view-instance">
                    <div class="view-instance-actions">
                      <button type="button" onclick={() => openEditor(entry)}>Edit</button>
                      <button type="button" onclick={() => resetView(entry)}>Reset</button>
                      <button type="button" onclick={() => openJson(entry)}>JSON</button>
                    </div>
                    <ViewBoundary component={viewFor(entry)} props={propsFor(entry)} />
                  </div>
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

{#if editingDoc}
  <ViewEditor
    open={true}
    doc={editingDoc}
    context={context}
    viewProps={viewProps}
    onclose={closeEditor}
    onsaved={handleSaved}
  />
{/if}

{#if editingLayout}
  <LayoutEditor
    open={true}
    layout={layout}
    onclose={() => (editingLayout = false)}
    onsaved={handleLayoutSaved}
  />
{/if}

{#if jsonDoc}
  <Drawer open={true} onclose={() => (jsonDoc = null)} width="min(620px, 100vw)">
    <div class="authoring-drawer">
      <div class="authoring-eyebrow">Resolved view document</div>
      <h2>{jsonDoc.title || jsonDoc.id}</h2>
      <pre class="view-json-output">{JSON.stringify(jsonDoc, null, 2)}</pre>
    </div>
  </Drawer>
{/if}
