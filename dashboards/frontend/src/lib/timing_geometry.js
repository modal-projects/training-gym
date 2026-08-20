import {
  CATEGORIES,
  GROUPS,
  HIDDEN_PHASES,
  NEGLIGIBLE_WORK_S,
  SAMPLED,
  TOOLTIP_HIDDEN_PHASES,
  basePhaseName,
} from "./timing_vocabulary.js";
import {
  collect,
  isAsyncSpans,
  isApproximateSpan,
  nest,
  stepsOf,
} from "./timing_spans.js";

const MIN_NESTED_RENDER_WIDTH_PX = 3;
// A stretch of the run that no phase covers is time we didn't measure, not time
// nothing happened in: container restart after a retry, or driver work between
// phases. Rendered to scale it crowds every measured bar off the chart, so a
// long enough one is compressed into a labelled break.
const COLLAPSED_GAP_MIN_S = 2;
const COLLAPSED_GAP_MIN_FRACTION = 0.04;
const COLLAPSED_GAP_FRACTION = 0.02;
const BREAK_LABEL_GAP_PX = 4;

export function isRenderedTimingSpan(span) {
  return (
    !span.mergedGeneration && !HIDDEN_PHASES.has(basePhaseName(span.name))
  );
}

export function breakLabelLayout(
  breaks,
  labelWidthsPx,
  timelineSpan,
  pixelsPerSecond,
) {
  if (
    !Number.isFinite(timelineSpan) ||
    timelineSpan <= 0 ||
    !Number.isFinite(pixelsPerSecond) ||
    pixelsPerSecond <= 0
  ) {
    return (breaks || []).map(() => ({ hidden: false, rightAligned: false }));
  }
  const timelineWidth = timelineSpan * pixelsPerSecond;
  let previousRight = Number.NEGATIVE_INFINITY;
  return (breaks || []).map((gap, index) => {
    const labelWidth = Math.max(Number(labelWidthsPx?.[index]) || 0, 0);
    const naturalLeft = gap.offset * pixelsPerSecond + 2;
    const rightAligned = naturalLeft + labelWidth > timelineWidth;
    const left = rightAligned
      ? timelineWidth - labelWidth - 2
      : naturalLeft;
    const hidden = left < previousRight + BREAK_LABEL_GAP_PX;
    if (!hidden) previousRight = Math.max(previousRight, left + labelWidth);
    return { hidden, rightAligned: rightAligned && !hidden };
  });
}

export function mergeSyncGenerationSpans(spans) {
  const drivers = new Map(
    spans
      .filter(
        (span) =>
          span.role === "driver" &&
          basePhaseName(span.name) === "generate_rollouts" &&
          span.rolloutId != null,
      )
      .map((span) => [span.rolloutId, span]),
  );
  const rolloutGenerationWrappers = new Set(
    spans
      .filter(
        (span) =>
          span.role === "rollout" &&
          basePhaseName(span.name) === "generate_samples" &&
          span.rolloutId != null,
      )
      .map((span) => span.rolloutId),
  );
  const mergeGeneration = (span, driver, wrapper) => {
    const sampleGeneration =
      wrapper
        ? nestedChild(wrapper, "sample_generation")
        : basePhaseName(span.name) === "sample_generation"
          ? span
          : null;
    const aggregateStats = Object.fromEntries(
      Object.entries(driver.aggregateStats || {}).map(([name, aggregate]) => [
        name,
        { ...aggregate },
      ]),
    );
    const descendants = wrapper ? [...(wrapper.children || [])] : [];
    for (let index = 0; index < descendants.length; index += 1) {
      for (const child of descendants[index].children || []) {
        descendants.push(child);
      }
    }
    const accumulate = (name, incoming) => {
      const aggregate = {
        ...incoming,
        duration: incoming.duration ?? incoming.end - incoming.start,
        total: incoming.total ?? incoming.duration ?? incoming.end - incoming.start,
        count: incoming.count ?? 1,
        longest: incoming.longest ?? incoming.duration ?? incoming.end - incoming.start,
        average: 0,
        mergedGeneration: false,
      };
      aggregate.average = aggregate.count ? aggregate.total / aggregate.count : 0;
      const current = aggregateStats[name];
      if (current) {
        current.duration += aggregate.duration;
        current.total += aggregate.total;
        current.count += aggregate.count;
        current.longest = Math.max(current.longest, aggregate.longest);
        current.start = Math.min(current.start, aggregate.start);
        current.end = Math.max(current.end, aggregate.end);
        current.average = current.count ? current.total / current.count : 0;
      } else {
        aggregateStats[name] = aggregate;
      }
    };
    if (sampleGeneration) accumulate("sample_generation", sampleGeneration);
    if (!wrapper && basePhaseName(span.name) !== "sample_generation") {
      accumulate(span.name, span);
    }
    for (const descendant of descendants) {
      if (!TOOLTIP_HIDDEN_PHASES.has(basePhaseName(descendant.name))) {
        const duration = descendant.duration ?? descendant.end - descendant.start;
        accumulate(descendant.name, {
          ...descendant,
          duration,
          total: descendant.total ?? duration,
          count: descendant.count ?? 1,
          longest: descendant.longest ?? duration,
        });
      }
    }
    if (Object.keys(aggregateStats).length) {
      driver.aggregateStats = aggregateStats;
    }
    (wrapper || span).mergedGeneration = true;
    while (descendants.length) {
      const descendant = descendants.pop();
      descendant.mergedGeneration = true;
      for (const child of descendant.children || []) {
        descendants.push(child);
      }
    }
    if ((wrapper || span).parent) {
      (wrapper || span).parent.children = (wrapper || span).parent.children.filter(
        (child) => child !== (wrapper || span),
      );
    }
  };
  for (const span of spans) {
    if (span.role !== "rollout") continue;
    const driver = drivers.get(span.rolloutId);
    if (!driver) continue;
    if (basePhaseName(span.name) === "generate_samples") {
      mergeGeneration(span, driver, span);
      continue;
    }
    if (
      !SAMPLED.has(basePhaseName(span.name)) &&
      basePhaseName(span.name) !== "reward_post_process"
    ) {
      continue;
    }
    if (rolloutGenerationWrappers.has(span.rolloutId)) {
      continue;
    }
    mergeGeneration(span, driver, null);
  }
  return spans;
}

function nestedChild(span, name) {
  for (const child of span.children || []) {
    if (basePhaseName(child.name) === name) return child;
    const nested = nestedChild(child, name);
    if (nested) return nested;
  }
  return null;
}

export function nestedHitTargetsForRow(row, pixelsPerSecond) {
  const nestedBars = row.sortedSpans.filter(
    (candidate) => candidate.depth > 0 && isRenderedTimingSpan(candidate),
  );
  const pixelExpansion =
    pixelsPerSecond > 0 ? 6 / pixelsPerSecond : 0;
  const renderedStart = (candidate) => candidate.renderStart ?? candidate.start;
  const renderedEnd = (candidate) => candidate.renderEnd ?? candidate.end;
  const renderedDuration = (candidate) =>
    candidate.renderDuration ?? renderedEnd(candidate) - renderedStart(candidate);
  const widthPx = (candidate) =>
    pixelsPerSecond > 0
      ? Math.round(renderedDuration(candidate) * pixelsPerSecond)
      : 0;
  const centers = nestedBars
    .map((candidate) => ({
      candidate,
      center: (renderedStart(candidate) + renderedEnd(candidate)) / 2,
    }))
    .sort(
      (a, b) =>
        a.center - b.center ||
        String(a.candidate.key).localeCompare(String(b.candidate.key)),
    );
  const previousEdges = new Map();
  let previous = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < centers.length; ) {
    const center = centers[index].center;
    let end = index;
    while (end < centers.length && centers[end].center === center) end += 1;
    for (let current = index; current < end; current += 1) {
      previousEdges.set(centers[current].candidate.key, previous);
    }
    previous = Math.max(
      previous,
      ...centers
        .slice(index, end)
        .map(({ candidate }) => renderedEnd(candidate)),
    );
    index = end;
  }
  const nextEdges = new Map();
  let next = Number.POSITIVE_INFINITY;
  for (let index = centers.length - 1; index >= 0; ) {
    const center = centers[index].center;
    let start = index;
    while (start >= 0 && centers[start].center === center) start -= 1;
    for (let current = start + 1; current <= index; current += 1) {
      nextEdges.set(centers[current].candidate.key, next);
    }
    next = Math.min(
      next,
      ...centers
        .slice(start + 1, index + 1)
        .map(({ candidate }) => renderedStart(candidate)),
    );
    index = start;
  }

  const orderedBars = [...nestedBars].sort((a, b) => {
    return (
      widthPx(a) - widthPx(b) ||
      renderedStart(b) - renderedStart(a) ||
      String(a.key).localeCompare(String(b.key))
    );
  });
  const zIndexes = new Map(
    orderedBars.map((candidate, index) => [
      candidate.key,
      orderedBars.length - index + 3,
    ]),
  );
  const targets = new Map();
  for (const candidate of nestedBars) {
    const start = renderedStart(candidate);
    const end = renderedEnd(candidate);
    const center = (start + end) / 2;
    const leftBoundary = previousEdges.get(candidate.key);
    const rightBoundary = nextEdges.get(candidate.key);
    const leftExpansion =
      leftBoundary === Number.NEGATIVE_INFINITY
        ? pixelExpansion
        : Math.max(0, start - leftBoundary);
    const rightExpansion =
      rightBoundary === Number.POSITIVE_INFINITY
        ? pixelExpansion
        : Math.max(0, rightBoundary - end);
    const insideStart =
      candidate.insideRenderStart ?? candidate.insideStart;
    const insideEnd = candidate.insideRenderEnd ?? candidate.insideEnd;
    const leftCap =
      insideStart == null
        ? pixelExpansion
        : Math.max(0, start - insideStart);
    const rightCap =
      insideEnd == null
        ? pixelExpansion
        : Math.max(0, insideEnd - end);
    const leftPx = Math.max(
      0,
      Math.min(leftCap, leftExpansion) * pixelsPerSecond,
    );
    const rightPx = Math.max(
      0,
      Math.min(rightCap, rightExpansion) * pixelsPerSecond,
    );
    targets.set(candidate.key, {
      zIndex: zIndexes.get(candidate.key),
      left: `${leftPx}px`,
      right: `${rightPx}px`,
    });
  }
  return targets;
}

function collapseUnmeasuredGaps(spans, runStart, timelineEnd) {
  const covered = [];
  const visit = (span) => {
    if (Number.isFinite(span.renderStart) && Number.isFinite(span.renderEnd)) {
      covered.push([span.renderStart, span.renderEnd]);
    }
    for (const child of span.children || []) visit(child);
  };
  for (const span of spans) visit(span);
  covered.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const total = timelineEnd - runStart;
  const minimum = Math.max(COLLAPSED_GAP_MIN_S, total * COLLAPSED_GAP_MIN_FRACTION);
  const gaps = [];
  let cursor = runStart;
  for (const [start, end] of covered) {
    if (start - cursor >= minimum) gaps.push([cursor, start]);
    cursor = Math.max(cursor, end);
  }
  if (timelineEnd - cursor >= minimum) gaps.push([cursor, timelineEnd]);
  if (!gaps.length) return null;
  const hidden = gaps.reduce((sum, [start, end]) => sum + (end - start), 0);
  const width = Math.max((total - hidden) * COLLAPSED_GAP_FRACTION, 1e-3);
  const mapTime = (seconds) => {
    let mapped = seconds;
    for (const [start, end] of gaps) {
      if (seconds <= start) break;
      if (seconds >= end) mapped -= end - start - width;
      else mapped -= (seconds - start) * (1 - width / (end - start));
    }
    return mapped;
  };
  return {
    mapTime,
    span: Math.max(mapTime(timelineEnd) - runStart, 1e-6),
    breaks: gaps.map(([start, end]) => ({
      offset: mapTime(start) - runStart,
      duration: width,
      hidden: end - start,
    })),
  };
}

export function clipIdleSpans(spans, async) {
  const workByRow = new Map();
  for (const span of spans) {
    if (span.kind === "idle" || span.unaligned) continue;
    const row = async && span.role === "rollout" ? "generation" : "step";
    const intervals = workByRow.get(row) || [];
    intervals.push([span.start, span.end]);
    workByRow.set(row, intervals);
  }
  for (const [row, intervals] of workByRow.entries()) {
    intervals.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const merged = [];
    for (const [start, end] of intervals) {
      const previous = merged.at(-1);
      if (previous && start <= previous[1]) {
        previous[1] = Math.max(previous[1], end);
      } else {
        merged.push([start, end]);
      }
    }
    workByRow.set(row, merged);
  }
  const piecesByIndex = new Map();
  spans.forEach((span, index) => {
    if (span.kind !== "idle") piecesByIndex.set(index, [span]);
  });
  const idleSpans = spans
    .map((span, index) => ({ span, index }))
    .filter(({ span }) => span.kind === "idle")
    .sort((a, b) => a.span.start - b.span.start);
  for (const { span, index } of idleSpans) {
    const pieces = [];
    const row = async && span.role === "rollout" ? "generation" : "step";
    const ranges = workByRow.get(row) || [];
    let low = 0;
    let high = ranges.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (ranges[middle][1] <= span.start) low = middle + 1;
      else high = middle;
    }
    let cursor = span.start;
    for (let current = low; current < ranges.length; current += 1) {
      const [workStart, workEnd] = ranges[current];
      if (workStart >= span.end) break;
      if (workStart > cursor) {
        const end = Math.min(workStart, span.end);
        if (end - cursor >= NEGLIGIBLE_WORK_S) {
          pieces.push({
            ...span,
            start: cursor,
            end,
            clockStart: span.clockStart + (cursor - span.start),
            clockEnd: span.clockStart + (end - span.start),
            total: end - cursor,
            duration: end - cursor,
          });
        }
      }
      cursor = Math.max(cursor, workEnd);
      if (cursor >= span.end) break;
    }
    if (cursor < span.end) {
      const start = cursor;
      const end = span.end;
      if (end - start >= NEGLIGIBLE_WORK_S) {
        pieces.push({
          ...span,
          start,
          end,
          clockStart: span.clockStart + (start - span.start),
          clockEnd: span.clockStart + (end - span.start),
          total: end - start,
          duration: end - start,
        });
      }
    }
    piecesByIndex.set(index, pieces);
  }
  const clipped = [];
  for (let index = 0; index < spans.length; index += 1) {
    clipped.push(...(piecesByIndex.get(index) || []));
  }
  return clipped;
}

function rowsOf(spans, async) {
  if (!spans.length) return [];
  const prepareRow = (row) => {
    row.unaligned = row.spans
      .filter(isRenderedTimingSpan)
      .some(isApproximateSpan);
    row.sortedSpans = [...row.spans].sort(
      (a, b) => a.depth - b.depth || a.start - b.start || b.end - a.end,
    );
    row.insetKeys = new Set();
    const byDepth = new Map();
    for (const span of row.sortedSpans) {
      const bucket = byDepth.get(span.depth) || [];
      bucket.push(span);
      byDepth.set(span.depth, bucket);
    }
    for (const spansAtDepth of byDepth.values()) {
      const ends = [...spansAtDepth].sort((a, b) => a.end - b.end);
      let endIndex = 0;
      let previousEnd = null;
      for (const span of spansAtDepth) {
        while (endIndex < ends.length && ends[endIndex].end <= span.start) {
          previousEnd = ends[endIndex].end;
          endIndex += 1;
        }
        if (previousEnd !== null && span.start > previousEnd) {
          row.insetKeys.add(span.key);
        }
      }
    }
    return row;
  };
  const rolloutSpans = spans.filter((span) => span.role === "rollout");
  const rootOf = (span) => {
    let root = span;
    while (root.parent) root = root.parent;
    return root;
  };
  const driverSpans = spans.filter(
    (span) =>
      !async ||
      span.role !== "rollout" ||
      rootOf(span).role !== "rollout",
  );
  if (!async) {
    if (!driverSpans.length) return [];
    return [prepareRow(
      {
        key: "driver",
        label: "Train",
        role: "driver",
        hint: "Driver and trainer phases on the shared wall clock.",
        spans: driverSpans,
      },
    )];
  }

  const rows = [];
  if (driverSpans.length) {
    rows.push(
      prepareRow({
        key: "driver",
        label: "Train",
        role: "driver",
        hint: "Driver and trainer phases on the shared wall clock.",
        spans: driverSpans,
      }),
    );
  }
  const roots = rolloutSpans.filter(
    (span) => span.depth === 0 && !span.mergedGeneration,
  );
  const packed = [];
  for (const span of [...roots].sort((a, b) => a.start - b.start || b.end - a.end)) {
    const row = packed.find((candidate) => candidate.end <= span.start);
    if (row) {
      row.end = span.end;
      row.roots.push(span);
    } else {
      packed.push({ end: span.end, roots: [span] });
    }
  }
  let rolloutIndex = 0;
  for (const packedRow of packed) {
    const rootSet = new Set(packedRow.roots);
    const rowSpans = rolloutSpans.filter((span) => {
      return rootSet.has(rootOf(span));
    });
    if (!rowSpans.some(isRenderedTimingSpan)) continue;
    rows.push(prepareRow({
      key: `rollout-${rolloutIndex}`,
      label: "Rollouts",
      role: "rollout",
      hint: "Rollout engine phases packed by their actual wall-clock overlap.",
      spans: rowSpans,
    }));
    rolloutIndex += 1;
  }
  return rows;
}

export function runTimeline(
  timings,
  asyncOverride = null,
  pixelsPerSecond = 0,
) {
  const measured = collect(timings);
  if (!measured.length) {
    return {
      span: 0,
      runStart: null,
      unaligned: false,
      breaks: [],
      mapOffset: (seconds) => seconds,
      async: false,
      groups: [],
      steps: [],
      categories: [],
    };
  }
  // Avoid spreading one argument per invocation; long runs can exceed the call stack.
  const driverSpans = measured.filter((span) => span.role === "driver");
  const bounds = driverSpans.length ? driverSpans : measured;
  let runStart = Infinity;
  let driverRunEnd = -Infinity;
  for (const span of bounds) {
    runStart = Math.min(runStart, span.start);
    driverRunEnd = Math.max(driverRunEnd, span.end);
  }
  const steps = stepsOf(measured);
  const rawSpans = measured;
  const sync = rawSpans.some(
    (span) =>
      span.role === "driver" &&
      basePhaseName(span.name) === "generate_rollouts",
  );
  const async =
    asyncOverride ?? isAsyncSpans(rawSpans);
  const spans = clipIdleSpans(nest(rawSpans), async);
  if (sync && !async) mergeSyncGenerationSpans(spans);
  const inFlightSpans = spans.filter(
    (span) =>
      span.unaligned &&
      span.start >= runStart &&
      span.start <= driverRunEnd &&
      span.end > driverRunEnd,
  );
  let timelineEnd = driverRunEnd;
  for (const span of inFlightSpans) {
    timelineEnd = Math.max(timelineEnd, span.end);
  }

  const renderBounds = (span) => {
    const inFlight =
      span.unaligned &&
      span.start >= runStart &&
      span.start <= driverRunEnd &&
      span.end > driverRunEnd;
    if (inFlight) return [span.start, span.end];
    if (!span.unaligned) return [span.start, span.end];
    const start = Math.max(runStart, Math.min(driverRunEnd, span.start));
    const end = Math.max(runStart, Math.min(driverRunEnd, span.end));
    if (end > start) return [start, end];
    const minimum = Math.min(
      driverRunEnd - runStart,
      Math.max((driverRunEnd - runStart) * 0.01, 1e-3),
    );
    if (span.end <= runStart) return [runStart, runStart + minimum];
    return [driverRunEnd - minimum, driverRunEnd];
  };
  const setRenderGeometry = (span, parent = null) => {
    const inFlight =
      span.unaligned &&
      span.start >= runStart &&
      span.start <= driverRunEnd &&
      span.end > driverRunEnd;
    span.alignmentQuestionable =
      !inFlight &&
      span.unaligned &&
      (span.start < runStart || span.end > driverRunEnd);
    let [renderStart, renderEnd] = renderBounds(span);
    span.insideRenderStart = parent?.renderStart ?? null;
    span.insideRenderEnd = parent?.renderEnd ?? null;
    const measuredRenderDuration = Math.max(renderEnd - renderStart, 0);
    span.widthInflated = false;
    if (parent && pixelsPerSecond > 0) {
      const minimumDuration =
        MIN_NESTED_RENDER_WIDTH_PX / pixelsPerSecond;
      if (measuredRenderDuration < minimumDuration) {
        const center = (renderStart + renderEnd) / 2;
        renderStart = center - minimumDuration / 2;
        renderEnd = center + minimumDuration / 2;
        const parentStart = parent.renderStart;
        const parentEnd = parent.renderEnd;
        if (renderStart < parentStart) {
          renderEnd += parentStart - renderStart;
          renderStart = parentStart;
        }
        if (renderEnd > parentEnd) {
          renderStart -= renderEnd - parentEnd;
          renderEnd = parentEnd;
        }
        renderStart = Math.max(parentStart, renderStart);
        renderEnd = Math.min(parentEnd, renderEnd);
        span.widthInflated = renderEnd - renderStart > measuredRenderDuration;
      }
    }
    span.renderStart = renderStart;
    span.renderEnd = renderEnd;
    span.renderOffset = renderStart - runStart;
    span.renderDuration = Math.max(renderEnd - renderStart, 0);
  };
  for (const [index, span] of spans.entries()) {
    span.key = `${span.rolloutId}:${span.role}:${span.name}:${span.start.toFixed(3)}:${index}`;
    setRenderGeometry(span);
  }
  for (const span of spans) {
    span.offset = span.start - runStart;
    span.duration = span.end - span.start;
    span.average = span.total / span.count;
    span.inside = span.parent ? span.parent.name : null;
    span.insideKey = span.parent ? span.parent.key : null;
    span.insideStart = span.parent ? span.parent.start : null;
    span.insideEnd = span.parent ? span.parent.end : null;
  }

  function hydrateNestedChildren(parent) {
    for (const child of parent.children || []) {
      child.offset = child.start - runStart;
      child.depth = parent.depth + 1;
      child.inside = parent.name;
      child.insideKey = parent.key;
      child.insideStart = parent.start;
      child.insideEnd = parent.end;
      setRenderGeometry(child, parent);
      child.key ??= `${parent.key}:child:${child.name}`;
      hydrateNestedChildren(child);
    }
  }
  for (const span of spans) {
    hydrateNestedChildren(span);
  }

  const collapsed = collapseUnmeasuredGaps(spans, runStart, timelineEnd);
  const mapTime = collapsed ? collapsed.mapTime : (seconds) => seconds;
  if (collapsed) {
    // Nested spans appear both in `spans` and in their parent's children, and
    // mapping a time twice is not the same as mapping it once, so each span is
    // compressed exactly once. The recursion still has to run: aggregated
    // per-sample children are only reachable through it.
    const compressed = new Set();
    const compress = (span) => {
      if (compressed.has(span)) return;
      compressed.add(span);
      span.renderStart = mapTime(span.renderStart);
      span.renderEnd = mapTime(span.renderEnd);
      span.renderOffset = span.renderStart - runStart;
      span.renderDuration = Math.max(span.renderEnd - span.renderStart, 0);
      if (span.insideRenderStart != null) {
        span.insideRenderStart = mapTime(span.insideRenderStart);
      }
      if (span.insideRenderEnd != null) {
        span.insideRenderEnd = mapTime(span.insideRenderEnd);
      }
      for (const child of span.children || []) compress(child);
    };
    for (const span of spans) compress(span);
  }

  const visibleSpans = spans;
  const rows = rowsOf(visibleSpans, async);
  const groups = rows.length ? [{ ...GROUPS[0], rows }] : [];
  for (const span of spans) delete span.parent;

  return {
    runStart,
    unaligned: spans.filter(isRenderedTimingSpan).some(isApproximateSpan),
    span: collapsed ? collapsed.span : Math.max(timelineEnd - runStart, 1e-6),
    breaks: collapsed ? collapsed.breaks : [],
    // Absolute-time affordances drawn on the track (attempt boundaries) have to
    // travel through the same compression as the bars.
    mapOffset: (seconds) => mapTime(runStart + seconds) - runStart,
    async,
    groups,
    steps: steps.map((step) => ({
      ...step,
      offset: step.start - runStart,
      duration: step.end - step.start,
      renderOffset: mapTime(step.start) - runStart,
      renderDuration: Math.max(mapTime(step.end) - mapTime(step.start), 0),
    })),
    categories: [...new Set(spans.map((span) => span.category))].sort(
      (a, b) => Object.keys(CATEGORIES).indexOf(a) - Object.keys(CATEGORIES).indexOf(b),
    ),
  };
}

export function timingRunStart(timings) {
  const measured = collect(timings);
  const driverSpans = measured.filter((span) => span.role === "driver");
  const bounds = driverSpans.length ? driverSpans : measured;
  // Avoid spreading one argument per invocation; long runs can exceed the call stack.
  let start = Infinity;
  for (const span of bounds) {
    start = Math.min(start, span.start);
  }
  return bounds.length ? start : null;
}
