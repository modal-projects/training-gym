<script>
  import DefaultTrajectoryViewer from "./TrajectoryViewer.svelte";
  import ConversationView from "./ConversationView.svelte";
  import { rolloutIndex } from "../lib/rolloutGrouping.js";

  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
  } = $props();

  let host = $state(null);
  let mode = $state("fallback");
  let loadError = $state("");
  let mounted = null;
  let loadKey = $state("");
  let generation = 0;

  let hasTrajectory = $derived(Array.isArray(trajectory) && trajectory.length > 0);

  function runId() {
    return run?.training_run_id || run?.run_id || "";
  }

  function sampleKey() {
    const coords = [
      rollout?.rollout_id ?? "",
      sample?.id ?? sample?.sample_id ?? "",
      rolloutIndex(sample) ?? "",
      sample?.sample_index ?? "",
      sample?.group_index ?? "",
    ];
    return coords.join("/");
  }

  function componentKey() {
    return JSON.stringify(run?.metadata?.dashboard_components || "");
  }

  function dispose() {
    if (typeof mounted?.unmount === "function") mounted.unmount();
    else if (typeof mounted?.$destroy === "function") mounted.$destroy();
    mounted = null;
    if (host) host.replaceChildren();
  }

  async function load() {
    const id = runId();
    const trajectoryLength = Array.isArray(trajectory) ? trajectory.length : 0;
    const rewardLength = Array.isArray(rewardEvents) ? rewardEvents.length : 0;
    const sampleCount = Array.isArray(samples) ? samples.length : 0;
    const key = `${id}:${sampleKey()}:${componentKey()}:${trajectoryLength}:${rewardLength}:${sampleCount}`;
    if (!host || !id || key === loadKey) return;
    loadKey = key;
    const current = ++generation;
    dispose();
    mode = "loading";
    loadError = "";
    try {
      const endpoint = `/api/runs/${encodeURIComponent(id)}/dashboard-components/trajectory_viewer/bundle.js`;
      const response = await fetch(endpoint, { credentials: "same-origin" });
      if (response.status === 404) {
        mode = "fallback";
        return;
      }
      if (!response.ok) throw new Error(`component request failed (${response.status})`);
      // Import the same-origin endpoint directly. Blob URLs are blocked by
      // some dashboard CSP/proxy configurations and are unnecessary because
      // the endpoint already serves a self-contained ES module.
      const module = await import(`${endpoint}?v=${encodeURIComponent(key)}`);
      if (current !== generation || !host || typeof module.mountViewer !== "function") return;
      mounted = module.mountViewer(host, { sample, samples, trajectory, rewardEvents, rollout, run });
      mode = "custom";
    } catch (error) {
      if (current !== generation) return;
      loadError = error?.message || String(error);
      mode = "fallback";
    }
  }

  $effect(() => {
    runId();
    sampleKey();
    componentKey();
    trajectory;
    rewardEvents;
    samples;
    if (host) load();
  });

  $effect(() => () => {
    generation += 1;
    dispose();
  });
</script>

{#if mode !== "custom"}
  {#if hasTrajectory}
    <DefaultTrajectoryViewer
      {sample}
      {samples}
      {trajectory}
      {rewardEvents}
      {rollout}
      {run}
    />
  {:else}
    <ConversationView
      messages={null}
      response={sample?.response || ""}
      thinking={sample?.thinking || ""}
      evalReport={sample?.metadata?.eval_report}
    />
  {/if}
  {#if mode === "loading"}
    <div class="viewer-note">Loading run-scoped trajectory viewer…</div>
  {:else if loadError}
    <div class="viewer-note viewer-error" title={loadError}>Run-scoped viewer unavailable; showing the default viewer.</div>
  {/if}
{/if}
<div class:custom-viewer={mode === "custom"} class="custom-viewer-host" bind:this={host}></div>

<style>
  .custom-viewer-host { position: absolute; width: 0; height: 0; overflow: hidden; }
  .custom-viewer-host.custom-viewer { position: static; width: 100%; height: auto; overflow: visible; }
  .viewer-note { margin: -8px 0 10px; color: var(--muted, #8b8b8b); font-size: 10px; }
  .viewer-error { color: var(--red, #f87171); }
</style>
