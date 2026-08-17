<script>
  import Drawer from "../components/Drawer.svelte";

  let { open = false, layout, onclose = () => {}, onsaved = () => {} } = $props();
  let draft = $state(null);
  let saving = $state(false);
  let message = $state("");

  $effect(() => {
    if (layout) draft = structuredClone($state.snapshot(layout));
  });

  function entriesFor(tab, slot) {
    return draft.tabs?.find((item) => item.id === tab.id)?.slots?.[slot] || [];
  }

  function move(tabId, slot, index, delta) {
    const tab = draft.tabs.find((item) => item.id === tabId);
    const entries = tab?.slots?.[slot];
    const target = index + delta;
    if (!entries || target < 0 || target >= entries.length) return;
    [entries[index], entries[target]] = [entries[target], entries[index]];
    draft = { ...draft, tabs: [...draft.tabs] };
  }

  function toggle(tabId, slot, index) {
    const tab = draft.tabs.find((item) => item.id === tabId);
    const entries = tab?.slots?.[slot];
    if (!entries?.[index]) return;
    entries[index] = { ...entries[index], hidden: !entries[index].hidden };
    draft = { ...draft, tabs: [...draft.tabs] };
  }

  async function save() {
    saving = true;
    message = "";
    try {
      const response = await fetch(`/api/ui/layouts/user/${encodeURIComponent(draft.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...draft, scope: "user", updated_at: Math.floor(Date.now() / 1000) }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      onsaved(await response.json());
      message = "Saved";
    } catch (reason) {
      message = `Save failed: ${reason?.message || reason}`;
    } finally {
      saving = false;
    }
  }

  async function reset() {
    const response = await fetch(`/api/ui/layouts/user/${encodeURIComponent(draft.id)}`, { method: "DELETE" });
    if (response.ok || response.status === 404) onclose();
    else message = `Reset failed: HTTP ${response.status}`;
  }
</script>

<Drawer {open} onclose={onclose} width="min(620px, 100vw)">
    <div class="authoring-drawer">
      {#if draft}
    <div class="authoring-header">
      <div><div class="authoring-eyebrow">Layout editor</div><h2>{draft.title}</h2></div>
      <div class="authoring-actions">
        <button class="log-button" type="button" onclick={reset}>Reset layout</button>
        <button class="log-button primary" type="button" disabled={saving} onclick={save}>{saving ? "Saving…" : "Save layout"}</button>
      </div>
    </div>
    {#each draft.tabs || [] as tab}
      <section class="layout-editor-tab">
        <div class="authoring-section-title">{tab.label}</div>
        {#each Object.entries(tab.slots || {}) as [slot, entries]}
          <div class="layout-editor-slot">
            <div class="layout-editor-slot-title">{slot}</div>
            {#each entries as entry, index (entry.id ?? index)}
              <div class:hidden={entry.hidden} class="layout-editor-entry">
                <code>{entry.view}</code>
                <div class="layout-editor-entry-actions">
                  <button type="button" onclick={() => move(tab.id, slot, index, -1)} aria-label="Move up">↑</button>
                  <button type="button" onclick={() => move(tab.id, slot, index, 1)} aria-label="Move down">↓</button>
                  <button type="button" onclick={() => toggle(tab.id, slot, index)}>{entry.hidden ? "Show" : "Hide"}</button>
                </div>
              </div>
            {/each}
          </div>
        {/each}
      </section>
    {/each}
    {#if message}<div class="authoring-message">{message}</div>{/if}
      {/if}
  </div>
</Drawer>
