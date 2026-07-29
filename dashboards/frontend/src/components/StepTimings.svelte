<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";

  let {
    stepTimes = null,
    substepTimes = null,
    layout = "rows",
    downloadName = "step_substep_times.json",
  } = $props();

  const SUBSTEP_LABELS = {
    evaluate_rollouts: "Eval (before)",
    generate_rollouts: "Generate rollouts",
    offload_rollout: "Offload rollout",
    compute_log_probs: "Compute log probs",
    optimizer_step: "Optimizer step",
    checkpoint_save: "Checkpoint save",
    offload_train: "Offload train",
    weight_sync: "Weight sync",
    evaluate_rollouts_end: "Eval (after)",
    full_step: "Full step",
    wait_for_rollout: "Wait for rollout",
    train_models: "Train models",
    train_model: "Train model",
    forward_backward: "Forward/backward",
    training_cleanup: "Training cleanup",
    wait_for_next_rollout: "Wait for next rollout",
    custom_reward: "Custom reward",
    custom_reward_post_process: "Reward post-process",
    evaluate_rollouts_before: "Eval (before)",
    evaluate_rollouts_after: "Eval (after)",
  };

  const SUBSTEP_COLORS = {
    evaluate_rollouts: "#60a5fa",
    generate_rollouts: "#34d399",
    offload_rollout: "#a78bfa",
    compute_log_probs: "#fbbf24",
    optimizer_step: "#f87171",
    weight_sync: "#22d3ee",
    checkpoint_save: "#f472b6",
    offload_train: "#c084fc",
    evaluate_rollouts_end: "#818cf8",
    full_step: "#64748b",
    wait_for_rollout: "#38bdf8",
    train_models: "#4ade80",
    train_model: "#4ade80",
    forward_backward: "#22c55e",
    training_cleanup: "#c084fc",
    wait_for_next_rollout: "#0ea5e9",
    custom_reward: "#f59e0b",
    custom_reward_post_process: "#fbbf24",
    evaluate_rollouts_before: "#60a5fa",
    evaluate_rollouts_after: "#818cf8",
  };

  const ORDER = Object.keys(SUBSTEP_LABELS);

  // Timeline zoom bounds: 1 = fit-to-width, MAX_ZOOM = deepest magnification.
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

  const ROLE_ORDER = ["rollout", "driver", "actor", "critic", "step"];

  function parseSubstepName(name) {
    const match = name.match(/^(.*) \((.*)\)$/);
    return {
      phase: match?.[1] || name,
      role: match?.[2] || "step",
    };
  }

  function labelFor(phase) {
    return SUBSTEP_LABELS[phase] || phase.replace(/_/g, " ");
  }

  function colorFor(phase) {
    return SUBSTEP_COLORS[phase] || "#fb923c";
  }

  // Durations are float seconds; keep up to 3 decimals (trailing zeros trimmed).
  function fmtSecs(s) {
    if (s == null) return "—";
    const n = Number(s);
    if (!Number.isFinite(n)) return "—";
    const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
    if (n >= 60) {
      const m = Math.floor(n / 60);
      return `${m}m ${trim(n - m * 60)}s`;
    }
    return `${trim(n)}s`;
  }

  function downloadJson() {
    const payload = { step_times: stepTimes || {}, substep_times: substepTimes || {} };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    a.click();
    URL.revokeObjectURL(url);
  }

  let steps = $derived.by(() => {
    const stepKeys = Object.keys(stepTimes || {});
    const subKeys = Object.keys(substepTimes || {});
    const keys = Array.from(new Set([...stepKeys, ...subKeys]));
    const out = keys.map((k) => {
      const st = (stepTimes || {})[k] || null;
      const subs = (substepTimes || {})[k] || {};
      const substeps = Object.entries(subs)
        .map(([name, v]) => {
          const { phase, role } = parseSubstepName(name);
          return {
            name,
            phase,
            role,
            start: v?.start ?? null,
            duration: v?.duration_s ?? null,
          };
        })
        .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      const roleNames = Array.from(
        new Set(substeps.map((substep) => substep.role)),
      ).sort((a, b) => {
        const aIndex = ROLE_ORDER.indexOf(a);
        const bIndex = ROLE_ORDER.indexOf(b);
        return (aIndex < 0 ? ROLE_ORDER.length : aIndex)
          - (bIndex < 0 ? ROLE_ORDER.length : bIndex);
      });
      const roles = roleNames.map((role) => ({
        role,
        substeps: substeps.filter((substep) => substep.role === role),
      }));
      const phaseStarts = substeps
        .map((substep) => Number(substep.start))
        .filter(Number.isFinite);
      const phaseEnds = substeps
        .map((substep) => Number(substep.start) + Number(substep.duration))
        .filter(Number.isFinite);
      const timelineStart = phaseStarts.length
        ? Math.min(...phaseStarts)
        : st?.start ?? null;
      const timelineEnd = phaseEnds.length
        ? Math.max(...phaseEnds)
        : st?.end ?? null;
      return {
        key: k,
        n: Number.isFinite(Number(k)) ? Number(k) : k,
        timelineStart,
        timelineDuration:
          timelineStart != null && timelineEnd != null
            ? Math.max(timelineEnd - timelineStart, 0)
            : st?.duration_s ?? null,
        duration: st?.duration_s ?? null,
        substeps,
        roles,
      };
    });
    out.sort((a, b) => (Number(a.key) || 0) - (Number(b.key) || 0));
    return out;
  });

  let hasData = $derived(steps.length > 0);

  let legend = $derived.by(() => {
    const seen = new Set();
    for (const s of steps) for (const sub of s.substeps) seen.add(sub.phase);
    return Array.from(seen).sort((a, b) => {
      const aIndex = ORDER.indexOf(a);
      const bIndex = ORDER.indexOf(b);
      return (aIndex < 0 ? ORDER.length : aIndex)
        - (bIndex < 0 ? ORDER.length : bIndex);
    });
  });

  let tip = $state(null);
  let pinned = $state(false);

  // ── Timeline zoom / pan state ────────────────────────────────────────
  let zoom = $state(1);
  let viewport = $state(null);

  function stepWeight(step) {
    if (step.timelineDuration != null && step.timelineDuration > 0) {
      return step.timelineDuration;
    }
    const roleDuration = Math.max(
      0,
      ...step.roles.map((role) =>
        role.substeps.reduce((total, substep) => total + (substep.duration ?? 0), 0),
      ),
    );
    if (roleDuration > 0) return roleDuration;
    return 1;
  }

  function positionedSubsteps(step, role) {
    const duration = Number(step.timelineDuration);
    const start = Number(step.timelineStart);
    const hasWallClockBounds =
      Number.isFinite(duration) && duration > 0 && Number.isFinite(start);
    if (hasWallClockBounds) {
      return role.substeps.map((substep) => {
        const substepStart = Number(substep.start);
        const substepDuration = Number(substep.duration);
        if (!Number.isFinite(substepStart) || !Number.isFinite(substepDuration)) {
          return { ...substep, left: 0, width: 2 };
        }
        const left = Math.max(0, Math.min(100, ((substepStart - start) / duration) * 100));
        const width = Math.max(
          0.4,
          Math.min(100 - left, (substepDuration / duration) * 100),
        );
        return { ...substep, left, width };
      });
    }
    const total = role.substeps.reduce(
      (sum, substep) => sum + Math.max(Number(substep.duration) || 0, 0),
      0,
    );
    let elapsed = 0;
    return role.substeps.map((substep) => {
      const substepDuration = Math.max(Number(substep.duration) || 0, 0);
      const left = total > 0 ? (elapsed / total) * 100 : 0;
      const width = total > 0 ? (substepDuration / total) * 100 : 100;
      elapsed += substepDuration;
      return { ...substep, left, width };
    });
  }

  function setZoom(next, anchorX = null) {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
    if (clamped === zoom) return;
    if (viewport) {
      const rect = viewport.getBoundingClientRect();
      const cursorX = anchorX == null ? rect.width / 2 : anchorX - rect.left;
      const contentX = viewport.scrollLeft + cursorX;
      const scale = clamped / zoom;
      zoom = clamped;
      requestAnimationFrame(() => {
        viewport.scrollLeft = contentX * scale - cursorX;
      });
    } else {
      zoom = clamped;
    }
  }

  function handleWheel(e) {
    // Let horizontal trackpad gestures pan natively; vertical wheel zooms.
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    setZoom(zoom * Math.exp(-e.deltaY * WHEEL_SENSITIVITY), e.clientX);
  }

  // Wheel listeners are passive by default; zooming needs preventDefault.
  function wheelZoom(node) {
    node.addEventListener("wheel", handleWheel, { passive: false });
    return {
      destroy() {
        node.removeEventListener("wheel", handleWheel);
      },
    };
  }

  function isActive(step, sub) {
    return tip && tip.step === step.n && tip.name === sub.name && tip.role === sub.role;
  }

  function showTip(e, step, sub) {
    if (pinned) return;
    tip = {
      x: e.clientX,
      y: e.clientY,
      step: step.n,
      name: sub.name,
      phase: sub.phase,
      role: sub.role,
      dur: sub.duration,
    };
  }

  function moveTip(e) {
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    if (pinned) return;
    tip = null;
  }

  function pinTip(e, step, sub) {
    e.stopPropagation();
    if (pinned && isActive(step, sub)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = {
      x: e.clientX,
      y: e.clientY,
      step: step.n,
      name: sub.name,
      phase: sub.phase,
      role: sub.role,
      dur: sub.duration,
    };
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }
</script>

<svelte:window onclick={clearPin} />

{#snippet segment(step, sub)}
  <div
    class="seg"
    class:seg-null={sub.duration == null}
    class:active={pinned && isActive(step, sub)}
    style:left={`${sub.left}%`}
    style:width={`${sub.width}%`}
    style:background={sub.duration == null ? undefined : colorFor(sub.phase)}
    role="button"
    tabindex="0"
    onmouseenter={(e) => showTip(e, step, sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, step, sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, step, sub);
      }
    }}
  ></div>
{/snippet}

{#if hasData}
  <div class="step-timings">
    {#if legend.length || layout === "timeline"}
      <div class="legend-row">
        <div class="legend">
          {#each legend as phase (phase)}
            <span class="legend-item">
              <span class="swatch" style:background={colorFor(phase)}></span>
              {labelFor(phase)}
            </span>
          {/each}
        </div>
        {#if layout === "timeline"}
          <div class="tl-toolbar">
            <div class="zoom-controls">
              <button
                class="zoom-btn"
                onclick={() => setZoom(zoom / ZOOM_BTN_FACTOR)}
                disabled={zoom <= MIN_ZOOM}
                title="Zoom out"
              >
                <ZoomOut size={13} />
              </button>
              <button
                class="zoom-level"
                onclick={() => setZoom(MIN_ZOOM)}
                disabled={zoom <= MIN_ZOOM}
                title="Reset zoom to fit"
              >
                {zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1).replace(/\.0$/, "")}×
              </button>
              <button
                class="zoom-btn"
                onclick={() => setZoom(zoom * ZOOM_BTN_FACTOR)}
                disabled={zoom >= MAX_ZOOM}
                title="Zoom in"
              >
                <ZoomIn size={13} />
              </button>
            </div>
            <button
              class="dl-btn"
              onclick={downloadJson}
              title="Download step + substep times as JSON"
            >
              <Download size={13} />
              Download JSON
            </button>
          </div>
        {/if}
      </div>
    {/if}

    {#if layout === "timeline"}
      <div class="tl-viewport" bind:this={viewport} use:wheelZoom>
        <div class="tl-track" style:width={`${zoom * 100}%`}>
          {#each steps as step, index (step.key)}
            <div
              class="tl-step"
              class:step-even={index % 2 === 0}
              class:step-odd={index % 2 !== 0}
              style:flex-grow={stepWeight(step)}
            >
              <div class="tl-step-head">
                <span class="tl-step-name">Step {step.n}</span>
                <span class="tl-step-dur">
                  {fmtSecs(step.duration)}
                  {#if step.timelineDuration > step.duration * 1.05}
                    · {fmtSecs(step.timelineDuration)} span
                  {/if}
                </span>
              </div>
              {#if step.roles.length}
                <div class="role-lanes">
                  {#each step.roles as role (role.role)}
                    <div class="role-lane">
                      <span class="role-label">{role.role === "step" ? "Timing" : role.role}</span>
                      <div class="bar tl-bar">
                        {#each positionedSubsteps(step, role) as sub (sub.name)}
                          {@render segment(step, sub)}
                        {/each}
                      </div>
                    </div>
                  {/each}
                </div>
              {:else}
                <div class="bar tl-bar bar-empty"></div>
              {/if}
            </div>
          {/each}
        </div>
      </div>
      <div class="tl-hint">Scroll to zoom · shift-scroll or drag the scrollbar to pan</div>
    {:else}
      {#each steps as step, index (step.key)}
        <div class="step-row" class:step-even={index % 2 === 0} class:step-odd={index % 2 !== 0}>
          <div class="step-head">
            <span class="step-name">Step {step.n}</span>
            <span class="step-dur">
              {fmtSecs(step.duration)}
              {#if step.timelineDuration > step.duration * 1.05}
                · {fmtSecs(step.timelineDuration)} span
              {/if}
            </span>
          </div>
          {#if step.roles.length}
            <div class="role-lanes">
              {#each step.roles as role (role.role)}
                <div class="role-lane">
                  <span class="role-label">{role.role === "step" ? "Timing" : role.role}</span>
                  <div class="bar">
                    {#each positionedSubsteps(step, role) as sub (sub.name)}
                      {@render segment(step, sub)}
                    {/each}
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="bar bar-empty"></div>
          {/if}
        </div>
      {/each}
    {/if}
  </div>

  {#if tip}
    <div class="tg-tip" class:pinned style:left={`${tip.x}px`} style:top={`${tip.y}px`}>
      <span class="tg-tip-step">Step {tip.step}</span>
      <span class="tg-tip-name">{labelFor(tip.phase)} ({tip.role})</span>
      <span class="tg-tip-dur">
        {tip.dur == null ? "unknown (report dropped)" : fmtSecs(tip.dur)}
      </span>
    </div>
  {/if}
{/if}

<style>
  .step-timings {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .legend-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 4px;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    font-size: 11px;
    color: var(--muted);
  }

  .dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex-shrink: 0;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
  }

  .dl-btn:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .swatch {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .step-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
    border-radius: 5px;
  }

  .step-even {
    background: color-mix(in srgb, var(--accent, #60a5fa) 5%, transparent);
    border-top: 1px solid color-mix(in srgb, var(--accent, #60a5fa) 20%, transparent);
  }

  .step-odd {
    background: color-mix(in srgb, #a78bfa 6%, transparent);
    border-top: 1px solid color-mix(in srgb, #a78bfa 22%, transparent);
  }

  .step-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    font-size: 12px;
  }

  .step-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .step-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .bar {
    position: relative;
    flex: 1;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .bar-empty {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .seg {
    position: absolute;
    top: 0;
    min-width: 2px;
    height: 100%;
    cursor: pointer;
    border-left: 1px solid color-mix(in srgb, #000 35%, transparent);
    transition: filter 0.1s ease;
  }

  .seg:hover {
    filter: brightness(1.25);
  }

  .seg.active {
    outline: 1px solid var(--text-bright, #fff);
    outline-offset: -1px;
    filter: brightness(1.3);
  }

  /* Dropped substep: visible but doesn't distort the proportional widths. */
  .seg-null {
    background: repeating-linear-gradient(
      45deg,
      var(--color-c-gray-20, #3a3a3a),
      var(--color-c-gray-20, #3a3a3a) 3px,
      var(--color-c-gray-10, #2f2f2f) 3px,
      var(--color-c-gray-10, #2f2f2f) 6px
    );
  }

  /* ── Timeline (full-width zoomable bar across all steps) ─────────────── */
  .tl-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .zoom-controls {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    overflow: hidden;
  }

  .zoom-btn,
  .zoom-level {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 7px;
    cursor: pointer;
  }

  .zoom-level {
    min-width: 38px;
    border-left: 1px solid var(--border, #2f2f2f);
    border-right: 1px solid var(--border, #2f2f2f);
    font-variant-numeric: tabular-nums;
  }

  .zoom-btn:hover:not(:disabled),
  .zoom-level:hover:not(:disabled) {
    color: var(--text);
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .zoom-btn:disabled,
  .zoom-level:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .tl-viewport {
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
    overscroll-behavior-x: contain;
    touch-action: pan-x;
    -webkit-overflow-scrolling: touch;
  }

  .tl-track {
    display: flex;
    gap: 3px;
    min-width: 100%;
  }

  .tl-step {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex-basis: 0;
    overflow: hidden;
    padding: 4px;
    border-radius: 4px;
  }

  @media (max-width: 900px) {
    .tl-step {
      min-width: 72px;
    }
  }

  .tl-step-head {
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-size: 10px;
    line-height: 14px;
    white-space: nowrap;
    overflow: hidden;
  }

  .tl-step-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .tl-step-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .tl-bar {
    height: 16px;
  }

  .role-lanes {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .role-lane {
    position: relative;
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }

  .role-label {
    flex: 0 0 45px;
    overflow: hidden;
    color: var(--muted);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.04em;
    line-height: 14px;
    text-align: right;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .tl-step .role-lane {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
  }

  .tl-step .role-label {
    position: absolute;
    z-index: 2;
    width: auto;
    max-width: calc(100% - 8px);
    margin-left: 3px;
    color: rgba(255, 255, 255, 0.82);
    line-height: 16px;
    text-align: left;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.9);
    pointer-events: none;
  }

  .tl-hint {
    font-size: 10px;
    color: var(--muted);
    opacity: 0.7;
  }

  /* ── Pinnable tooltip ────────────────────────────────────────────────── */
  .tg-tip {
    position: fixed;
    z-index: 1000;
    transform: translate(-50%, calc(-100% - 10px));
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 6px 9px;
    border-radius: 6px;
    background: var(--color-c-gray-02, #0d0d0d);
    border: 1px solid var(--border, #3a3a3a);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    font-size: 11px;
    white-space: nowrap;
  }

  .tg-tip.pinned {
    border-color: var(--accent, #60a5fa);
  }

  .tg-tip-step {
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .tg-tip-name {
    color: var(--text-bright, #fff);
    font-weight: 600;
  }

  .tg-tip-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
</style>
