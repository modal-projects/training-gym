<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";
  import {
    buildActivityMetrics,
    buildFrameworkTimeline,
    isWait,
  } from "./frameworkActivityTrace.js";

  let {
    steps,
    stepTimes,
    rolloutStats,
    legend,
    phaseOrder,
    labelFor,
    colorFor,
    descriptionFor,
    downloadJson,
  } = $props();

  const TIME_TICKS = [0, 0.25, 0.5, 0.75, 1];
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

  let expandedGroups = $state(new Set());
  let zoom = $state(1);
  let viewport = $state(null);
  let tip = $state(null);
  let pinned = $state(false);
  let selectedFlow = $state(null);

  let metrics = $derived(buildActivityMetrics(steps, rolloutStats));
  let timeline = $derived(
    buildFrameworkTimeline(
      steps,
      stepTimes,
      expandedGroups,
      labelFor,
      phaseOrder,
    ),
  );

  function fmtSecs(value) {
    if (value == null) return "—";
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "—";
    const trim = (number) => number.toFixed(3).replace(/\.?0+$/, "");
    if (seconds >= 60) {
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m ${trim(seconds - minutes * 60)}s`;
    }
    return `${trim(seconds)}s`;
  }

  function fmtTimestamp(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "—";
    return new Date(seconds * 1000)
      .toISOString()
      .replace("T", " ")
      .replace("Z", " UTC");
  }

  function fmtClockUncertainty(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "—";
    if (seconds < 0.001) return `${Math.round(seconds * 1_000_000)}µs`;
    if (seconds < 1) {
      return `${(seconds * 1000).toFixed(1).replace(/\.0$/, "")}ms`;
    }
    return fmtSecs(seconds);
  }

  function toggleGroup(group) {
    const next = new Set(expandedGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    expandedGroups = next;
  }

  function setZoom(next, anchorX = null) {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
    if (clamped === zoom) return;
    if (!viewport) {
      zoom = clamped;
      return;
    }
    const rect = viewport.getBoundingClientRect();
    const cursorX = anchorX == null ? rect.width / 2 : anchorX - rect.left;
    const contentX = viewport.scrollLeft + cursorX;
    const scale = clamped / zoom;
    zoom = clamped;
    requestAnimationFrame(() => {
      viewport.scrollLeft = contentX * scale - cursorX;
    });
  }

  function wheelZoom(node) {
    const handleWheel = (event) => {
      if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
      event.preventDefault();
      setZoom(
        zoom * Math.exp(-event.deltaY * WHEEL_SENSITIVITY),
        event.clientX,
      );
    };
    node.addEventListener("wheel", handleWheel, { passive: false });
    return {
      destroy() {
        node.removeEventListener("wheel", handleWheel);
      },
    };
  }

  function isActive(step, substep) {
    return (
      tip &&
      tip.step === step &&
      tip.name === substep.name &&
      tip.role === substep.role
    );
  }

  function isRelated(substep) {
    if (selectedFlow == null) return true;
    const sourceMatches =
      selectedFlow.source != null &&
      substep.sourceRolloutId != null &&
      Number(substep.sourceRolloutId) === selectedFlow.source;
    const trainingMatches =
      selectedFlow.training != null &&
      substep.trainingRolloutId != null &&
      Number(substep.trainingRolloutId) === selectedFlow.training;
    return sourceMatches || trainingMatches;
  }

  function containedMeasurements(stepNumber, parent) {
    const step = steps.find((candidate) => candidate.n === stepNumber);
    const parentStart = Number(parent.start);
    const parentEnd = parentStart + Number(parent.duration);
    if (!step || !Number.isFinite(parentStart) || !Number.isFinite(parentEnd)) {
      return [];
    }
    const contained = step.substeps
      .filter((candidate) => {
        if (
          candidate.phase === parent.phase ||
          candidate.phase === "full_step"
        ) {
          return false;
        }
        const parentMatches =
          candidate.parentPhase != null &&
          candidate.parentPhase === parent.phase;
        const legacyRoleMatches =
          candidate.parentPhase == null &&
          (candidate.role === parent.role ||
            (parent.phase === "train_models" &&
              (candidate.role === "actor" || candidate.role === "critic")));
        if (!parentMatches && !legacyRoleMatches) return false;
        const start = Number(candidate.start);
        const end = start + Number(candidate.duration);
        return (
          Number.isFinite(start) &&
          Number.isFinite(end) &&
          start >= parentStart - 0.001 &&
          end <= parentEnd + 0.001
        );
      })
      .sort((left, right) => Number(left.start) - Number(right.start));
    const counts = new Map();
    for (const candidate of contained) {
      const key = `${candidate.role}:${candidate.phase}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const occurrences = new Map();
    return contained.map((candidate) => {
      const key = `${candidate.role}:${candidate.phase}`;
      const occurrence = (occurrences.get(key) || 0) + 1;
      occurrences.set(key, occurrence);
      const suffix = counts.get(key) > 1 ? ` ${occurrence}` : "";
      return {
        key: `${candidate.role}:${candidate.name}`,
        label: `${candidate.displayName || labelFor(candidate.phase)}${suffix} (${candidate.role})`,
        duration: candidate.duration,
      };
    });
  }

  function tooltipFor(event, step, substep) {
    return {
      x: event.clientX,
      y: event.clientY,
      step,
      name: substep.name,
      phase: substep.phase,
      role: substep.role,
      displayName: substep.displayName,
      duration: substep.duration,
      start: substep.start,
      description: descriptionFor(substep.phase),
      details: containedMeasurements(step, substep),
      activityRolloutId: substep.activityRolloutId,
      activityRolloutKind: substep.activityRolloutKind,
      sourceRolloutId: substep.sourceRolloutId,
      trainingRolloutId: substep.trainingRolloutId,
      clockUncertainty: substep.clockUncertainty,
      executionSequence: substep.executionSequence,
    };
  }

  function showTip(event, step, substep) {
    if (!pinned) tip = tooltipFor(event, step, substep);
  }

  function moveTip(event) {
    if (!pinned && tip) tip = { ...tip, x: event.clientX, y: event.clientY };
  }

  function hideTip() {
    if (!pinned) tip = null;
  }

  function pinTip(event, step, substep) {
    event.stopPropagation();
    if (pinned && isActive(step, substep)) {
      pinned = false;
      tip = null;
      selectedFlow = null;
      return;
    }
    pinned = true;
    tip = tooltipFor(event, step, substep);
    selectedFlow =
      substep.sourceRolloutId == null && substep.trainingRolloutId == null
        ? null
        : {
            source:
              substep.sourceRolloutId == null
                ? null
                : Number(substep.sourceRolloutId),
            training:
              substep.trainingRolloutId == null
                ? null
                : Number(substep.trainingRolloutId),
          };
  }

  function handleWindowClick(event) {
    if (
      event.target instanceof Element &&
      event.target.closest(".framework-activity-trace")
    ) {
      return;
    }
    pinned = false;
    tip = null;
    selectedFlow = null;
  }
</script>

<svelte:window onclick={handleWindowClick} />

{#snippet segment(substep, summary)}
  <div
    class="segment"
    class:wait={!summary && isWait(substep)}
    class:summary
    class:unrelated={!isRelated(substep)}
    class:active={pinned && isActive(substep.step, substep)}
    style:left={`${substep.left}%`}
    style:width={`${substep.width}%`}
    style:background={substep.summaryColor ||
      (isWait(substep) ? undefined : colorFor(substep.phase))}
    role="button"
    tabindex="0"
    onmouseenter={(event) => showTip(event, substep.step, substep)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(event) => pinTip(event, substep.step, substep)}
    onkeydown={(event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        pinTip(event, substep.step, substep);
      }
    }}
  >
    {#if substep.activityRolloutId != null}
      {#if substep.phase === "generate_rollouts"}
        <span class="segment-label">R{Number(substep.activityRolloutId) + 1}</span>
      {:else if substep.phase === "train_model"}
        <span class="segment-label">T{Number(substep.activityRolloutId) + 1}</span>
      {/if}
    {/if}
  </div>
{/snippet}

<div class="framework-activity-trace">
  <div class="trace-header">
    <div class="legend">
      {#each legend as phase (phase)}
        <span class="legend-item">
          <span
            class="swatch"
            class:wait={phase === "wait_for_rollout" ||
              phase === "wait_for_next_rollout"}
            style:background={colorFor(phase)}
          ></span>
          {labelFor(phase)}
        </span>
      {/each}
    </div>
    <div class="toolbar">
      <div class="zoom-controls">
        <button
          onclick={() => setZoom(zoom / ZOOM_FACTOR)}
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
          onclick={() => setZoom(zoom * ZOOM_FACTOR)}
          disabled={zoom >= MAX_ZOOM}
          title="Zoom in"
        >
          <ZoomIn size={13} />
        </button>
      </div>
      <button class="download" onclick={downloadJson} title="Download timing JSON">
        <Download size={13} />
        Download JSON
      </button>
    </div>
  </div>

  {#if metrics.hasFrameworkActivity}
    <div class="metrics">
      <div class="metric">
        <span class="metric-value">{metrics.rolloutCount}</span>
        <span class="metric-label">Rollouts shown</span>
      </div>
      {#if metrics.throughput != null}
        <div class="metric">
          <span class="metric-value">{metrics.throughput.toFixed(1)}</span>
          <span class="metric-label">Samples / rollout second</span>
        </div>
      {/if}
      {#if metrics.overlap != null}
        <div class="metric">
          <span class="metric-value">{(metrics.overlap * 100).toFixed(1)}%</span>
          <span class="metric-label">Rollout / train overlap</span>
        </div>
      {/if}
      <div class="metric">
        <span class="metric-value">{metrics.retries}</span>
        <span class="metric-label">Role retries</span>
      </div>
    </div>
  {/if}

  <div class="viewport" bind:this={viewport} use:wheelZoom>
    <div class="content" style:width={`${zoom * 100}%`}>
      <div class="axis">
        <div class="axis-head">Shared wall time</div>
        {#each timeline.tracks as track (track.key)}
          {#if track.isSummary}
            <button
              class="track-label group-label"
              class:expanded={track.expanded}
              disabled={!track.expandable}
              onclick={(event) => {
                event.stopPropagation();
                toggleGroup(track.group);
              }}
            >
              <span class="caret">{track.expandable ? "›" : ""}</span>
              <span>{track.label}</span>
            </button>
          {:else}
            <div class="track-label child-label">{track.label}</div>
          {/if}
        {/each}
      </div>
      <div class="timeline">
        <div class="timeline-head">
          {#each TIME_TICKS as tick (tick)}
            <span
              class="time-tick"
              class:first={tick === 0}
              class:last={tick === 1}
              style:left={`${tick * 100}%`}
            >
              +{fmtSecs(timeline.duration * tick)}
            </span>
          {/each}
        </div>
        <div class="lanes">
          {#each TIME_TICKS as tick (tick)}
            <div class="grid-line" style:left={`${tick * 100}%`}></div>
          {/each}
          {#each timeline.tracks as track (track.key)}
            <div
              class="track"
              class:summary-track={track.isSummary}
              class:child-track={!track.isSummary}
            >
              {#each track.substeps as substep (`${substep.step}:${substep.name}`)}
                {@render segment(substep, track.isSummary)}
              {/each}
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>
  <div class="hint">
    Shared span {fmtSecs(timeline.duration)} · expand groups for phase detail ·
    striped color is a known wait · blank space is inactive or not instrumented
  </div>

  {#if tip}
    <div
      class="tooltip"
      class:pinned
      style:left={`${tip.x}px`}
      style:top={`${tip.y}px`}
    >
      <span class="tooltip-step">
        {#if
          tip.sourceRolloutId != null &&
          tip.trainingRolloutId != null &&
          Number(tip.sourceRolloutId) !== Number(tip.trainingRolloutId)}
          Source R{Number(tip.sourceRolloutId) + 1}
          → Training T{Number(tip.trainingRolloutId) + 1}
        {:else if tip.activityRolloutId != null}
          {tip.activityRolloutKind === "source" ? "Source" : "Training"}
          rollout {Number(tip.activityRolloutId) + 1}
        {:else}
          Step {tip.step}
        {/if}
      </span>
      <span class="tooltip-name">
        {tip.displayName || labelFor(tip.phase)} ({tip.role})
      </span>
      <span class="tooltip-duration">{fmtSecs(tip.duration)}</span>
      <span class="tooltip-time">
        {fmtTimestamp(tip.start)} → {fmtTimestamp(
          Number(tip.start) + Number(tip.duration),
        )}
      </span>
      {#if tip.description}
        <span class="tooltip-description">{tip.description}</span>
      {/if}
      {#if tip.details.length}
        <span class="tooltip-detail-title">Contained measurements</span>
        <span class="tooltip-details">
          {#each tip.details as detail (detail.key)}
            <span class="tooltip-detail">
              <span>{detail.label}</span>
              <span>{fmtSecs(detail.duration)}</span>
            </span>
          {/each}
        </span>
      {/if}
      {#if tip.clockUncertainty != null}
        <span class="tooltip-uncertainty">
          start aligned within ±{fmtClockUncertainty(tip.clockUncertainty)}
        </span>
      {/if}
      {#if Number(tip.executionSequence) > 1}
        <span class="tooltip-retry">
          retry {Number(tip.executionSequence) - 1}
        </span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .framework-activity-trace {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .trace-header,
  .toolbar,
  .zoom-controls,
  .legend,
  .legend-item {
    display: flex;
    align-items: center;
  }

  .trace-header {
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .legend {
    flex-wrap: wrap;
    gap: 6px 14px;
    color: var(--muted);
    font-size: 11px;
  }

  .legend-item {
    gap: 5px;
  }

  .swatch {
    width: 9px;
    height: 9px;
    flex-shrink: 0;
    border-radius: 2px;
  }

  .wait {
    background: repeating-linear-gradient(
      135deg,
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 65%, #181818),
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 65%, #181818) 3px,
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 20%, #181818) 3px,
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 20%, #181818) 6px
    ) !important;
  }

  .toolbar {
    flex-shrink: 0;
    gap: 8px;
  }

  .zoom-controls {
    overflow: hidden;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
  }

  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 0;
    background: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 11px;
  }

  .zoom-controls button {
    padding: 3px 7px;
  }

  .zoom-level {
    min-width: 38px;
    border-right: 1px solid var(--border, #2f2f2f);
    border-left: 1px solid var(--border, #2f2f2f);
    font-variant-numeric: tabular-nums;
  }

  button:disabled {
    cursor: default;
    opacity: 0.4;
  }

  .download {
    gap: 5px;
    padding: 3px 8px;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
  }

  .metrics {
    display: inline-flex;
    width: fit-content;
    flex-wrap: wrap;
    overflow: hidden;
    border: 1px solid var(--color-edge-secondary, #2f2f2f);
    border-radius: 6px;
    background: var(--color-surface-primary, #181818);
  }

  .metric {
    display: flex;
    min-width: 108px;
    flex-direction: column;
    gap: 2px;
    padding: 7px 11px;
    border-right: 1px solid var(--color-edge-secondary, #2f2f2f);
  }

  .metric:last-child {
    border-right: 0;
  }

  .metric-value {
    color: var(--color-foreground-active, #fff);
    font-family: var(--font-mono);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
  }

  .metric-label {
    color: var(--color-foreground-tertiary, #747474);
    font-size: 10px;
    line-height: 14px;
  }

  .viewport {
    overflow-x: auto;
    overflow-y: hidden;
    padding: 8px 0 6px;
    border: 1px solid var(--color-edge-secondary, #2f2f2f);
    border-radius: 6px;
    background: var(--color-surface-primary, #181818);
    overscroll-behavior-x: contain;
  }

  .content {
    display: grid;
    min-width: 100%;
    grid-template-columns: 154px minmax(0, 1fr);
  }

  .axis {
    position: sticky;
    left: 0;
    z-index: 5;
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-right: 8px;
    padding-left: 10px;
    background: var(--color-surface-primary, #181818);
    box-shadow: 6px 0 8px -8px rgba(0, 0, 0, 0.9);
  }

  .axis-head {
    box-sizing: border-box;
    height: 30px;
    padding-top: 4px;
    color: var(--color-foreground-tertiary, #747474);
    font-family: var(--font-mono);
    font-size: 10px;
  }

  .track-label {
    box-sizing: border-box;
    width: 100%;
    height: 18px;
    padding: 0;
    color: var(--color-foreground-secondary, #a3a3a3);
    font-size: 11px;
    font-weight: 500;
    line-height: 18px;
    text-align: left;
  }

  .group-label {
    justify-content: flex-start;
    gap: 4px;
    color: var(--color-foreground-primary, #d1d1d1);
  }

  .group-label:disabled {
    opacity: 1;
  }

  .caret {
    width: 10px;
    flex: 0 0 10px;
    font-size: 13px;
    transition: transform 0.12s ease;
  }

  .group-label.expanded .caret {
    transform: rotate(90deg);
  }

  .child-label {
    overflow: hidden;
    height: 16px;
    padding-left: 18px;
    color: var(--color-foreground-tertiary, #747474);
    font-size: 10px;
    line-height: 16px;
    text-overflow: ellipsis;
  }

  .timeline {
    min-width: 0;
  }

  .timeline-head {
    position: relative;
    box-sizing: border-box;
    height: 30px;
    overflow: hidden;
    border-bottom: 1px solid var(--color-edge-secondary, #2f2f2f);
  }

  .time-tick {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    color: var(--color-foreground-tertiary, #747474);
    font-family: var(--font-mono);
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  .time-tick.first {
    transform: none;
  }

  .time-tick.last {
    transform: translateX(-100%);
  }

  .lanes {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 3px;
    overflow: hidden;
  }

  .grid-line {
    position: absolute;
    top: 0;
    bottom: 0;
    z-index: 0;
    border-left: 1px solid color-mix(in srgb, var(--muted) 15%, transparent);
  }

  .track {
    position: relative;
    z-index: 1;
    box-sizing: border-box;
    width: 100%;
    height: 18px;
    border: 1px solid var(--color-edge-secondary, #2f2f2f);
    border-radius: 3px;
    background: var(--color-c-gray-2, #1c1c1c);
  }

  .child-track {
    height: 16px;
    border-color: color-mix(
      in srgb,
      var(--color-edge-secondary, #2f2f2f) 70%,
      transparent
    );
  }

  .segment {
    position: absolute;
    top: 0;
    z-index: 2;
    height: 100%;
    border-left: 1px solid color-mix(in srgb, #000 35%, transparent);
    cursor: pointer;
  }

  .segment:hover,
  .segment.active {
    filter: brightness(1.25);
  }

  .segment.active {
    outline: 1px solid var(--text-bright, #fff);
    outline-offset: -1px;
  }

  .segment.summary {
    box-shadow: inset 0 0 0 1px color-mix(in srgb, #fff 24%, transparent);
    opacity: 0.82;
  }

  .segment.unrelated {
    opacity: 0.16;
  }

  .segment-label {
    position: absolute;
    inset: 0 2px;
    overflow: hidden;
    color: color-mix(in srgb, #000 72%, transparent);
    font-size: 8px;
    font-weight: 700;
    line-height: 14px;
    white-space: nowrap;
  }

  .hint {
    color: var(--muted);
    font-size: 10px;
    opacity: 0.7;
  }

  .tooltip {
    position: fixed;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 8px 10px;
    transform: translate(-50%, calc(-100% - 10px));
    border: 1px solid var(--color-edge-primary, #464646);
    border-radius: 6px;
    background: var(--color-surface-overlay-large, #1c1c1c);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    font-size: 12px;
    pointer-events: none;
    white-space: nowrap;
  }

  .tooltip.pinned {
    position: static;
    width: min(560px, 100%);
    box-sizing: border-box;
    padding: 12px 14px;
    transform: none;
    border-color: var(--color-c-dataviz-primary-7, #648fe0);
    pointer-events: auto;
    white-space: normal;
  }

  .tooltip-step,
  .tooltip-time,
  .tooltip-uncertainty {
    color: var(--color-foreground-tertiary, #747474);
    font-family: var(--font-mono);
    font-size: 10px;
  }

  .tooltip-name {
    color: var(--color-foreground-active, #fff);
    font-size: 13px;
    font-weight: 500;
  }

  .tooltip-duration {
    color: var(--color-foreground-secondary, #a3a3a3);
    font-family: var(--font-mono);
  }

  .tooltip-description {
    max-width: 300px;
    margin-top: 4px;
    color: var(--color-foreground-secondary, #a3a3a3);
    line-height: 1.35;
  }

  .tooltip-detail-title {
    margin-top: 5px;
    color: var(--color-foreground-tertiary, #747474);
    font-size: 10px;
  }

  .tooltip-details {
    display: flex;
    min-width: 230px;
    flex-direction: column;
    gap: 2px;
  }

  .tooltip-detail {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    color: var(--color-foreground-primary, #d1d1d1);
    font-family: var(--font-mono);
  }

  .tooltip-retry {
    color: var(--yellow, #fbbf24);
    font-size: 10px;
  }
</style>
