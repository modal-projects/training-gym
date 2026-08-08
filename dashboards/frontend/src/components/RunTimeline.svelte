<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";
  import {
    CATEGORIES,
    colorFor,
    fmtSecs,
    HIDDEN_PHASES,
    TOOLTIP_HIDDEN_PHASES,
    labelFor,
    PHASE_COLORS,
    TRAIN_OUTLINE_COLOR,
    runTimeline,
  } from "../lib/timing.js";

  let {
    timings = null,
    timelineKey = "",
    downloadName = "substep_timing.json",
    onOpenRollout = null,
    rolloutIds = [],
    runOrigin = null,
    asyncOverride = null,
    showOpenRollout = true,
  } = $props();

  const MIN_ZOOM = 1;
  const MAX_ZOOM = 128;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0025;

  const ROW_HEIGHT_PX = 15;
  const DETAIL_ROW_HEIGHT_PX = 22;
  const ROW_GAP_PX = 4;
  const HEADER_PX = 0;
  const GROUP_GAP_PX = 12;
  const STEP_GAP_PX = 8;
  const BAR_GAP_PX = 1;
  let showDetails = $state(false);
  let timeline = $derived(runTimeline(timings, asyncOverride));
  let intervalOrigin = $derived(runOrigin ?? timeline.runStart);
  let rowHeight = $derived(showDetails ? DETAIL_ROW_HEIGHT_PX : ROW_HEIGHT_PX);
  let groups = $derived(
    timeline.groups.map((group) => ({
      ...group,
      height: HEADER_PX + group.rows.length * (rowHeight + ROW_GAP_PX),
    })),
  );
  $effect(() => {
    timelineKey;
    pinned = false;
    tip = null;
  });
  let visibleGroups = $derived(groups);
  let trackHeight = $derived(
    visibleGroups.reduce((total, group) => total + group.height + GROUP_GAP_PX, 0),
  );

  const pct = (seconds) => (seconds / timeline.span) * 100;

  let zoom = $state(1);
  let viewport = $state(null);

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
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    setZoom(zoom * Math.exp(-e.deltaY * WHEEL_SENSITIVITY), e.clientX);
  }

  function downloadJson() {
    const blob = new Blob([JSON.stringify(timings || {}, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    a.click();
    URL.revokeObjectURL(url);
  }

  let tip = $state(null);
  let pinned = $state(false);
  let hideTimer = null;

  const isActive = (bar) => tip && tip.bar.key === bar.key;

  function clearHideTimer() {
    if (hideTimer !== null) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  function scheduleHide() {
    clearHideTimer();
    if (!pinned) hideTimer = window.setTimeout(() => (tip = null), 180);
  }

  function showTip(e, bar) {
    clearHideTimer();
    if (pinned) return;
    tip = { x: e.clientX, y: e.clientY, bar };
  }

  function moveTip(e) {
    clearHideTimer();
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    scheduleHide();
  }

  function pinTip(e, bar) {
    e.stopPropagation();
    if (pinned && isActive(bar)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = { x: e.clientX, y: e.clientY, bar };
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }

  function tipTitle(bar) {
    const name = labelFor(bar.name, bar.rolloutId);
    return bar.ordinal ? `${name} ${bar.ordinal}` : name;
  }

  function nestedChild(bar, name) {
    for (const child of bar.children || []) {
      if (child.name === name) return child;
      const nested = nestedChild(child, name);
      if (nested) return nested;
    }
    return null;
  }

  function generationStats(bar) {
    if (bar.aggregateStats?.sample_generation) {
      return bar.aggregateStats.sample_generation;
    }
    return bar.name === "generate_samples"
      ? nestedChild(bar, "sample_generation")
      : null;
  }

  function hasVisibleChildren(bar) {
    return bar.children?.some((child) => !HIDDEN_PHASES.has(child.name)) ?? false;
  }

  function isExpandedParent(bar) {
    return showDetails && bar.depth === 0 && hasVisibleChildren(bar);
  }

  function displaySpans(row) {
    return row.sortedSpans
      .filter(
        (bar) =>
          !bar.mergedGeneration &&
          !HIDDEN_PHASES.has(bar.name) &&
          (showDetails || bar.depth === 0),
      )
  }

  function visualInset(row, bar) {
    return row.insetKeys.has(bar.key) ? BAR_GAP_PX : 0;
  }

  function visibleChildren(bar) {
    return showDetails
      ? (bar.children || []).filter((child) => !HIDDEN_PHASES.has(child.name))
      : [];
  }

  function shellLeft(row, bar) {
    if (bar.insideStart == null) {
      return `calc(${pct(bar.offset)}% + ${visualInset(row, bar)}px)`;
    }
    const duration = Math.max(bar.insideEnd - bar.insideStart, 0);
    const start = duration
      ? Math.min(1, Math.max(0, (bar.start - bar.insideStart) / duration))
      : 0;
    return `${start * 100}%`;
  }

  function shellWidth(bar) {
    if (bar.insideStart == null) {
      return `${Math.max(pct(bar.duration), 0.01)}%`;
    }
    const duration = Math.max(bar.insideEnd - bar.insideStart, 0);
    const end = duration
      ? Math.min(1, Math.max(0, (bar.end - bar.insideStart) / duration))
      : 0;
    const start = duration
      ? Math.min(1, Math.max(0, (bar.start - bar.insideStart) / duration))
      : 0;
    return `${Math.max(0, end - start) * 100}%`;
  }
</script>

<svelte:window onclick={clearPin} />

<div class="run-timeline">
  {#if !groups.length}
    <div class="empty">No substep timing recorded for these rollouts yet.</div>
  {:else}
    <div class="toolbar">
      <div class="legend">
        {#each timeline.categories as key (key)}
          <span class="legend-item">
            <span
              class="swatch"
              class:waiting-swatch={key === "idle"}
              style:background={key === "idle" ? "transparent" : CATEGORIES[key].color}
            ></span>
            {CATEGORIES[key].label}
          </span>
        {/each}
      </div>
      <div class="controls">
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
        <button class="dl-btn" onclick={downloadJson} title="Download timing as JSON">
          <Download size={13} />
          Download JSON
        </button>
        <button
          class="dl-btn detail-btn"
          onclick={() => {
            showDetails = !showDetails;
          }}
          aria-pressed={showDetails}
          title={showDetails ? "Hide detailed view" : "Show detailed view"}
        >
          {showDetails ? "Hide detailed view" : "Show detailed view"}
        </button>
      </div>
    </div>

    <div class="chart">
      <div class="gutter" style:padding-top={`${rowHeight + STEP_GAP_PX}px`}>
        {#each visibleGroups as group (group.key)}
          <div style:margin-bottom={`${GROUP_GAP_PX}px`}>
            {#each group.rows as row, index (index)}
              <div
                class="gutter-row"
                class:lane={true}
                style:height={`${rowHeight}px`}
                style:margin-bottom={`${ROW_GAP_PX}px`}
                title={`${row.label} — ${row.hint || group.hint}`}
              >
                {row.label}
              </div>
            {/each}
          </div>
        {/each}
      </div>

      <div
        class="viewport"
        bind:this={viewport}
        onwheel={handleWheel}
      >
        <div class="track" style:width={`${zoom * 100}%`}>
          <div class="steps" style:height={`${rowHeight}px`}>
            {#each timeline.steps as step (step.id)}
              <div
                class="step"
                style:left={`${pct(step.offset)}%`}
                style:width={`${Math.max(pct(step.duration), 0.05)}%`}
                title={`Step ${step.id}: ${fmtSecs(step.duration)} wall clock — ${fmtSecs(step.work)} work, ${fmtSecs(step.stalled)} measured stall`}
              >
                <span class="step-text"
                  >Step {step.id} · {fmtSecs(step.duration)}</span
                >
              </div>
            {/each}
          </div>

          <div class="groups" style:height={`${trackHeight}px`}>
            {#each visibleGroups as group (group.key)}
              <div
                class="group"
                style:height={`${group.height}px`}
                style:margin-bottom={`${GROUP_GAP_PX}px`}
              >
                {#each group.rows as row, index (index)}
                  <div
                    class="row"
                    style:top={`${HEADER_PX + index * (rowHeight + ROW_GAP_PX)}px`}
                    style:height={`${rowHeight}px`}
                  >
                    {#snippet renderBar(bar, row)}
                      <div
                        class="bar-shell"
                        class:nested-shell={bar.depth > 0}
                        style:left={shellLeft(row, bar)}
                        style:width={shellWidth(bar)}
                      >
                        <button
                          class="bar"
                          class:stall={bar.kind === "stall"}
                          class:detail-stall={showDetails && bar.kind === "stall"}
                          class:sampled={bar.kind === "sampled"}
                          class:nested-bar={showDetails && bar.depth > 0}
                          class:outlined={isExpandedParent(bar)}
                          class:train-parent={
                            isExpandedParent(bar) &&
                            ["train_models", "training", "train_model"].includes(bar.name)
                          }
                          class:expanded-parent={isExpandedParent(bar)}
                          class:active={pinned && isActive(bar)}
                          aria-label={`${labelFor(bar.name, bar.rolloutId)} ${fmtSecs(bar.duration)}`}
                          style:left="0"
                          style:width="100%"
                          style:--bar-color={
                            isExpandedParent(bar) &&
                            ["train_models", "training", "train_model"].includes(bar.name)
                              ? TRAIN_OUTLINE_COLOR
                              : colorFor(bar.name)
                          }
                          style:background={bar.kind === "work" && !isExpandedParent(bar) ? colorFor(bar.name) : undefined}
                          style:border-color={
                            isExpandedParent(bar)
                              ? ["train_models", "training", "train_model"].includes(bar.name)
                                ? TRAIN_OUTLINE_COLOR
                                : colorFor(bar.name)
                              : undefined
                          }
                          onmouseenter={(e) => showTip(e, bar)}
                          onmousemove={moveTip}
                          onmouseleave={hideTip}
                          onclick={(e) => pinTip(e, bar)}
                        >
                        </button>
                        {#if visibleChildren(bar).length}
                          <div class="bar-children">
                            {#each visibleChildren(bar) as child (child.key)}
                              {@render renderBar(child, row)}
                            {/each}
                          </div>
                        {/if}
                      </div>
                    {/snippet}
                    {#each displaySpans(row).filter((bar) => bar.depth === 0) as bar (bar.key)}
                      {@render renderBar(bar, row)}
                    {/each}
                  </div>
                {/each}
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>

  {/if}
</div>

{#if tip}
  <div
    class="tg-tip"
    class:pinned
    role="tooltip"
    style:left={`${tip.x}px`}
    style:top={`${tip.y}px`}
    onmouseenter={clearHideTimer}
    onmouseleave={scheduleHide}
  >
    <div class="tg-tip-main">
      <span class="tg-tip-time">
        {tip.bar.rolloutId == null ? "" : `Step ${tip.bar.rolloutId} · `}{fmtSecs(tip.bar.duration)}
      </span>
    </div>
    <span class="tg-tip-name">{tipTitle(tip.bar)}</span>
    {#if tip.bar.count === 1}
      <span class="tg-tip-when">
        {fmtSecs(tip.bar.start - intervalOrigin)} → {fmtSecs(tip.bar.end - intervalOrigin)}
      </span>
    {/if}
    {#if generationStats(tip.bar)}
      <span class="tg-tip-stat">
        Average sample generation time: {fmtSecs(generationStats(tip.bar).average)}
      </span>
      <span class="tg-tip-stat">
        Longest sample generation time: {fmtSecs(
          generationStats(tip.bar).longest ?? generationStats(tip.bar).duration,
        )}
      </span>
    {/if}
    {#if tip.bar.children?.length}
      <div class="tg-tip-children">
        {#each tip.bar.children.filter((child) => !child.mergedGeneration && !TOOLTIP_HIDDEN_PHASES.has(child.name)) as child (child.key)}
          <span class="tg-tip-child">
            <span class="tg-tip-child-line">
              {labelFor(child.name, child.rolloutId)}
              <span class="tg-tip-child-duration"> · {fmtSecs(child.duration)}</span>
            </span>
            {#if showDetails && child.count === 1}
              <span class="tg-tip-when">
                {fmtSecs(child.start - intervalOrigin)} → {fmtSecs(child.end - intervalOrigin)}
              </span>
            {/if}
          </span>
        {/each}
      </div>
    {/if}
    {#if showOpenRollout && tip.bar.name === "generate_samples" && tip.bar.kind === "work" && tip.bar.rolloutId != null && rolloutIds.includes(Number(tip.bar.rolloutId)) && onOpenRollout}
      <button
        class="tg-tip-action"
        onclick={(e) => {
          e.stopPropagation();
          onOpenRollout(tip.bar.rolloutId);
        }}
      >
        Open in Rollouts →
      </button>
    {/if}
  </div>
{/if}

<style>
  .run-timeline {
    display: flex;
    flex-direction: column;
    gap: 12px;
    font-family: var(--font-sans);
  }

  .empty {
    color: var(--color-c-gray-45, #6e6e6e);
    font-size: 0.85rem;
    padding: 0.5rem 0;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .legend {
    flex: 1 1 100%;
    min-width: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 3px 8px;
    font-size: 9px;
    color: var(--muted);
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
  }

  .swatch {
    width: 7px;
    height: 7px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .waiting-swatch {
    height: 2px;
    border-radius: 0;
    border-top: 2px solid var(--color-c-gray-30, #6a6a6a);
  }

  .controls {
    margin-left: auto;
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
    font-family: inherit;
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

  .dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
    font-family: inherit;
  }

  .dl-btn:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .chart {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .gutter {
    flex-shrink: 0;
    width: 108px;
  }

  .gutter-row {
    font-size: 11px;
    line-height: 20px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: var(--font-sans);
  }

  .gutter-row.lane {
    color: var(--text);
    font-weight: 600;
  }

  .viewport {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 10px;
    scrollbar-width: thin;
    scrollbar-color: var(--color-c-gray-20, #464646) transparent;
    overscroll-behavior-x: contain;
    touch-action: pan-x;
  }

  .track {
    position: relative;
    min-width: 100%;
  }

  .steps {
    position: relative;
    margin-bottom: 8px;
  }

  .step {
    position: absolute;
    top: 0;
    bottom: 0;
    border-left: 1px solid var(--border-strong, #4a4a4a);
    background: var(--color-c-gray-05, #171717);
    overflow: visible;
  }

  .step-text {
    display: block;
    padding: 0 6px;
    font-size: 10px;
    line-height: 20px;
    color: var(--muted);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .groups {
    position: relative;
  }

  .group {
    position: relative;
    border-top: 1px solid var(--border, #2f2f2f);
  }

  .row {
    position: absolute;
    left: 0;
    right: 0;
    pointer-events: none;
  }

  .bar {
    position: absolute;
    top: 0;
    height: 100%;
    display: flex;
    align-items: center;
    min-width: 1px;
    padding: 0;
    border: none;
    border-radius: 1px;
    box-sizing: border-box;
    outline: 1px solid var(--panel, #1a1a1a);
    overflow: hidden;
    cursor: pointer;
    pointer-events: auto;
    background: transparent;
    font-family: inherit;
  }

  .bar-shell {
    position: absolute;
    top: 0;
    height: 100%;
    pointer-events: none;
  }

  .bar-shell > .bar {
    pointer-events: auto;
  }

  .bar-children {
    position: absolute;
    inset: 2px;
    z-index: 3;
    overflow: hidden;
    pointer-events: none;
  }

  .bar-children > .bar-shell {
    pointer-events: auto;
  }

  .bar.outlined {
    background: transparent;
    border: 1px solid var(--bar-color);
  }

  .bar.train-parent {
    border-width: 2px;
  }

  .bar.expanded-parent {
    z-index: 2;
    pointer-events: auto;
    background: color-mix(in srgb, var(--bar-color) 18%, transparent) !important;
  }

  .bar.nested-bar {
    min-width: 2px;
    top: 4px;
    height: calc(100% - 8px);
    z-index: 3;
    outline: none;
    border-radius: 0;
  }

  .bar.stall {
    background: linear-gradient(
      var(--color-c-gray-30, #6a6a6a),
      var(--color-c-gray-30, #6a6a6a)
    );
    background-size: 100% 2px;
    background-position: center;
    background-repeat: no-repeat;
  }

  .bar.detail-stall {
    opacity: 0.35;
    z-index: 1;
  }

  .bar.sampled {
    background: transparent;
    border: none;
    background: linear-gradient(var(--bar-color), var(--bar-color));
    background-size: 100% 2px;
    background-position: center;
    background-repeat: no-repeat;
  }

  .bar.active {
    outline: 2px solid var(--color-c-green-80, #6ac355);
    outline-offset: -1px;
  }

  .tg-tip {
    position: fixed;
    z-index: 1000;
    transform: translate(-50%, calc(-100% - 10px));
    pointer-events: auto;
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
    font-family: var(--font-sans);
  }

  .tg-tip::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -12px;
    height: 12px;
    pointer-events: none;
  }

  .tg-tip.pinned {
    border-color: var(--accent, #60a5fa);
    pointer-events: auto;
  }

  .tg-tip-main {
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-name {
    color: var(--color-c-gray-100);
    font-weight: 600;
  }

  .tg-tip-when {
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 9px;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-children {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-top: 3px;
    padding-top: 3px;
    border-top: 1px solid var(--border, #3a3a3a);
  }

  .tg-tip-child {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .tg-tip-child-line {
    color: var(--muted);
    font-weight: 600;
    font-family: inherit;
    font-size: inherit;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-child-duration {
    font-family: var(--font-mono);
    font-weight: 400;
  }

  .tg-tip-action {
    align-self: flex-start;
    margin-top: 4px;
    padding: 2px 0;
    border: none;
    background: none;
    color: var(--accent, #60a5fa);
    font-family: inherit;
    cursor: pointer;
  }

  .tg-tip-action:hover {
    text-decoration: underline;
  }

  .detail-btn {
    background: var(--color-c-gray-12, #262626);
    border-color: var(--color-c-gray-25, #555);
    color: var(--text-bright, #fff);
  }

</style>
