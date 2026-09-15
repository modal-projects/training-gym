<script>
  import { untrack } from "svelte";
  import DefaultTrajectoryViewer from "./TrajectoryViewer.svelte";

  /**
   * Renders the run-scoped trajectory viewer when one is attached to `run`,
   * otherwise the built-in `TrajectoryViewer`.
   *
   * Run-scoped components are untrusted code. They run inside a sandboxed
   * <iframe> served by `/dashboard-components/trajectory_viewer/frame.html`
   * (opaque origin, CSP without network access) and only receive props via
   * postMessage, so they can never act with the dashboard's authority.
   */
  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
    position = null,
  } = $props();

  const MARK = "trainingGymDashboardComponent";
  const READY_TIMEOUT_MS = 20000;

  let frame = $state(null);
  let frameSrc = $state("");
  let frameHeight = $state(0);
  let mode = $state("fallback"); // "fallback" | "loading" | "custom"
  let loadError = $state("");
  let frameReady = false;
  let generation = 0;
  let readyTimer = null;

  function runId() {
    return run?.training_run_id || run?.run_id || "";
  }

  function componentKey() {
    return JSON.stringify(run?.metadata?.dashboard_components || "");
  }

  function reset() {
    generation += 1;
    frameReady = false;
    clearTimeout(readyTimer);
    readyTimer = null;
    frameSrc = "";
    frameHeight = 0;
  }

  function fail(message) {
    loadError = message;
    mode = "fallback";
    reset();
  }

  function postProps() {
    if (!frameReady || !frame?.contentWindow) return;
    const props = $state.snapshot({ sample, samples, trajectory, rewardEvents, rollout, run, position });
    // The frame has an opaque origin, so "*" is the only valid target.
    frame.contentWindow.postMessage({ [MARK]: true, type: "props", props }, "*");
  }

  async function load() {
    const id = runId();
    reset();
    loadError = "";
    if (!id) {
      mode = "fallback";
      return;
    }
    const current = generation;
    mode = "loading";
    try {
      const base = `/api/runs/${encodeURIComponent(id)}/dashboard-components/trajectory_viewer`;
      const manifestResponse = await fetch(base, { credentials: "same-origin" });
      if (current !== generation) return;
      if (manifestResponse.status === 404) {
        mode = "fallback";
        return;
      }
      if (!manifestResponse.ok) throw new Error(`component lookup failed (${manifestResponse.status})`);
      const manifest = await manifestResponse.json();
      if (current !== generation) return;
      const src = `${base}/frame.html?v=${encodeURIComponent(manifest?.sha256 || "")}`;
      // Compile errors surface here as a readable status instead of a blank frame.
      const frameResponse = await fetch(src, { credentials: "same-origin" });
      if (current !== generation) return;
      if (!frameResponse.ok) {
        let detail = `component request failed (${frameResponse.status})`;
        try {
          detail = (await frameResponse.json())?.detail || detail;
        } catch {
          // Non-JSON error bodies keep the generic status message.
        }
        throw new Error(detail);
      }
      frameSrc = src;
      readyTimer = setTimeout(() => {
        if (current === generation && !frameReady) fail("run-scoped viewer did not start");
      }, READY_TIMEOUT_MS);
    } catch (error) {
      if (current !== generation) return;
      fail(error?.message || String(error));
    }
  }

  function onMessage(event) {
    if (!frame || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (!data || data[MARK] !== true) return;
    if (data.type === "ready") {
      frameReady = true;
      clearTimeout(readyTimer);
      readyTimer = null;
      postProps();
    } else if (data.type === "rendered") {
      mode = "custom";
    } else if (data.type === "resize") {
      const height = Number(data.height);
      if (Number.isFinite(height) && height >= 0) frameHeight = Math.ceil(height);
    } else if (data.type === "error") {
      fail(String(data.message || "run-scoped viewer failed"));
    }
  }

  $effect(() => {
    runId();
    componentKey();
    untrack(load);
  });

  $effect(() => {
    // Track the identity of the selection props so a new selection (or a
    // freshly fetched rollout) is re-sent; deep reads inside `postProps` are
    // untracked so `run` polling does not remount the component.
    sample;
    samples;
    trajectory;
    rewardEvents;
    rollout;
    position;
    untrack(postProps);
  });

  $effect(() => {
    window.addEventListener("message", onMessage);
    return () => {
      window.removeEventListener("message", onMessage);
      reset();
    };
  });
</script>

{#if mode !== "custom"}
  <DefaultTrajectoryViewer
    {sample}
    {samples}
    {trajectory}
    {rewardEvents}
    {rollout}
    {run}
  />
  {#if mode === "loading"}
    <div class="viewer-note">Loading run-scoped trajectory viewer…</div>
  {:else if loadError}
    <div class="viewer-note viewer-error" title={loadError}>Run-scoped viewer unavailable; showing the default viewer.</div>
  {/if}
{/if}
{#if frameSrc}
  <iframe
    bind:this={frame}
    class="custom-viewer-frame"
    class:visible={mode === "custom"}
    src={frameSrc}
    sandbox="allow-scripts"
    referrerpolicy="no-referrer"
    title="Run-scoped trajectory viewer"
    style:height={mode === "custom" ? `${frameHeight}px` : "0px"}
  ></iframe>
{/if}

<style>
  .custom-viewer-frame { display: block; width: 100%; height: 0; border: 0; overflow: hidden; color-scheme: dark; }
  .custom-viewer-frame:not(.visible) { position: absolute; width: 0; height: 0; }
  .viewer-note { margin: -8px 0 10px; color: var(--muted, #8b8b8b); font-size: 10px; }
  .viewer-error { color: var(--red, #f87171); }
</style>
