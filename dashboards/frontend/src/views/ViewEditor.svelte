<script>
  import Drawer from "../components/Drawer.svelte";
  import { builtinViews } from "./registry.js";
  import AuthoredView from "./AuthoredView.svelte";
  import exampleDoc from "./examples/trajectory-authored.json";

  let {
    open = false,
    doc,
    context = {},
    viewProps = {},
    onclose = () => {},
    onsaved = () => {},
  } = $props();

  let source = $state("");
  let previewSource = $state("");
  let dirty = $state(false);
  let saving = $state(false);
  let message = $state("");
  let previewData = $state(null);
  let timer;

  const initialSource = $derived(doc?.code || doc?.source || "");

  $effect(() => {
    if (!dirty) {
      source = initialSource;
      previewSource = initialSource;
    }
  });

  const allowedImports = [
    "$host/components",
    "$host/format",
    "$host/data",
    "$host/icons",
    "svelte",
  ];
  const themeVariables = [
    "--bg",
    "--panel",
    "--surface",
    "--border",
    "--text",
    "--text-bright",
    "--muted",
    "--accent",
    "--green",
    "--yellow",
    "--red",
    "--color-c-gray-10",
  ];

  $effect(() => {
    if (!open) return;
    if (context.run_id) {
      fetch(`/api/docs/gym/runs/${encodeURIComponent(context.run_id)}`)
        .then((response) => (response.ok ? response.json() : null))
        .then((value) => {
          previewData = value;
        })
        .catch(() => {
          previewData = viewProps.initialRun || null;
        });
    }
  });

  function updateSource(value) {
    source = value;
    dirty = true;
    clearTimeout(timer);
    timer = window.setTimeout(() => {
      previewSource = source;
    }, 350);
  }

  function loadExample() {
    source = exampleDoc.code;
    previewSource = source;
    dirty = true;
    message = "Loaded authored trajectory example";
  }

  async function save() {
    saving = true;
    message = "";
    const next = {
      ...doc,
      id: doc.id,
      scope: "user",
      code: source,
      component: null,
      forked_from: doc.scope === "builtin" ? `builtin:${doc.id}` : doc.forked_from || null,
      updated_at: Math.floor(Date.now() / 1000),
    };
    try {
      const response = await fetch(`/api/ui/views/user/${encodeURIComponent(doc.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const saved = await response.json();
      dirty = false;
      message = "Saved";
      onsaved(saved);
    } catch (reason) {
      message = `Save failed: ${reason?.message || reason}`;
    } finally {
      saving = false;
    }
  }

  async function reset() {
    message = "";
    try {
      const response = await fetch(`/api/ui/views/user/${encodeURIComponent(doc.id)}`, {
        method: "DELETE",
      });
      if (!response.ok && response.status !== 404) throw new Error(`HTTP ${response.status}`);
      onclose();
    } catch (reason) {
      message = `Reset failed: ${reason?.message || reason}`;
    }
  }

  const previewProps = $derived({
    ...viewProps,
    data: previewData,
    config: doc.config || {},
    context,
    navigate: (path) => window.history.pushState({}, "", path),
  });
  const builtinComponent = $derived(doc.scope === "builtin" ? builtinViews[doc.component] : null);
</script>

<Drawer {open} onclose={onclose} width="min(1180px, 100vw)">
  <div class="authoring-drawer">
    <div class="authoring-header">
      <div>
        <div class="authoring-eyebrow">View editor</div>
        <h2>{doc.title || doc.id}</h2>
      </div>
      <div class="authoring-actions">
        <button class="log-button" type="button" onclick={loadExample}>Load example</button>
        <button class="log-button" type="button" onclick={reset}>Reset to builtin</button>
        <button class="log-button primary" type="button" disabled={saving || !dirty} onclick={save}>{saving ? "Saving…" : "Save"}</button>
      </div>
    </div>
    <div class="authoring-grid">
      <section class="authoring-source">
        <label for="view-source">Source</label>
        <textarea id="view-source" spellcheck="false" value={source} oninput={(event) => updateSource(event.currentTarget.value)}></textarea>
      </section>
      <section class="authoring-preview">
        <div class="authoring-section-title">Live preview</div>
        {#if builtinComponent && !dirty}
          {@const Preview = builtinComponent}
          <Preview {...viewProps} data={previewData} config={doc.config || {}} context={context} />
        {:else}
          <AuthoredView source={previewSource} props={previewProps} />
        {/if}
      </section>
    </div>
    <details class="authoring-details">
      <summary>Resolved data, imports, and theme variables</summary>
      <div class="authoring-details-grid">
        <pre>{JSON.stringify(previewData, null, 2)}</pre>
        <div>
          <div class="authoring-section-title">Allowed imports</div>
          <ul>{#each allowedImports as item}<li><code>{item}</code></li>{/each}</ul>
          <div class="authoring-section-title">Theme CSS variables</div>
          <ul>{#each themeVariables as item}<li><code>{item}</code></li>{/each}</ul>
        </div>
      </div>
    </details>
    {#if message}<div class="authoring-message">{message}</div>{/if}
  </div>
</Drawer>
