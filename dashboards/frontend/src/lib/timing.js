const slot = (name) => `var(--color-c-dataviz-${name})`;

export const TRAIN_OUTLINE_COLOR = slot("train-outline");

export const CATEGORIES = {
  train: { label: "Train", color: slot("primary-1") },
  generate: { label: "Rollout", color: slot("primary-3") },
  transfer: { label: "Weight sync", color: slot("primary-2") },
  checkpoint: { label: "Checkpoint", color: slot("primary-5") },
  eval: { label: "Eval", color: slot("primary-4") },
  idle: { label: "Waiting", color: "var(--color-c-gray-30)" },
};

export const PHASE_CATEGORY = {
  train_models: "train",
  training: "train",
  train_model: "train",
  compute_log_probs: "train",
  forward_backward: "train",
  optimizer_step: "train",
  generate_rollouts: "generate",
  generate_samples: "generate",
  sample_generation: "generate",
  reward: "generate",
  reward_batch: "generate",
  reward_post_process: "generate",
  weight_sync: "transfer",
  initial_weight_sync: "transfer",
  offload_train: "transfer",
  offload_rollout: "transfer",
  checkpoint_save: "checkpoint",
  evaluate_rollouts: "eval",
  evaluate_rollouts_end: "eval",
  wait_for_rollout: "idle",
  wait_for_next_rollout: "idle",
};

export const PHASE_COLORS = {
  compute_log_probs: slot("train-large"),
  forward_backward: slot("train-alt-a"),
  optimizer_step: slot("train-alt-b"),
};

export const TIMING_LABELS = {
  evaluate_rollouts: "Eval (before training)",
  evaluate_rollouts_end: "Eval (after training)",
  generate_rollouts: "Rollout generation",
  offload_rollout: "Offload generation engines",
  compute_log_probs: "Calculate log probs",
  train_models: "Train",
  training: "Train",
  train_model: "Train",
  checkpoint_save: "Save checkpoint",
  offload_train: "Offload trainer",
  weight_sync: "Weight sync",
  initial_weight_sync: "Initial weight sync",
  wait_for_rollout: "Waiting for this rollout",
  wait_for_next_rollout: "Waiting for the next rollout",
  generate_samples: "Rollout generation",
  sample_generation: "Sample generation",
  reward: "Reward",
  reward_batch: "Reward (whole batch)",
  reward_post_process: "Reward post process",
  forward_backward: "Forward/backward",
  optimizer_step: "Optimizer step",
};

const STALLS = new Set([
  "wait_for_rollout",
  "wait_for_next_rollout",
]);

const SAMPLED = new Set(["reward", "reward_batch", "sample_generation"]);
export const HIDDEN_PHASES = new Set([
  "reward",
  "reward_batch",
  "reward_post_process",
  "sample_generation",
]);
export const TOOLTIP_HIDDEN_PHASES = new Set([
  "reward",
  "reward_batch",
  "sample_generation",
]);

const NESTS_IN = {
  generate_samples: ["generate_rollouts"],
  compute_log_probs: ["train_models"],
  forward_backward: ["train_models"],
  optimizer_step: ["train_models"],
  reward: ["generate_samples"],
  reward_batch: ["generate_samples"],
  reward_post_process: ["generate_samples"],
  sample_generation: ["generate_samples"],
};

export const GROUPS = [
  {
    key: "timeline",
    label: "Timeline",
    hint: "measured phases on the shared wall clock",
  },
];

const NEGLIGIBLE_WORK_S = 0.0005;
export function labelFor(name, rolloutId = null) {
  if (
    (name === "wait_for_rollout" || name === "wait_for_next_rollout") &&
    rolloutId != null
  ) {
    const step = Number(rolloutId) + (name === "wait_for_next_rollout" ? 1 : 0);
    if (Number.isInteger(step)) {
      return `Waiting for rollout generation (step ${step})`;
    }
  }
  return TIMING_LABELS[name] || name.replace(/_/g, " ");
}

export function isLegacyTiming(timings) {
  return timings?.metadata?.legacy_derived === true;
}

export function rolloutIdForTimingKey(id) {
  if (id === "") return null;
  const parsedId = Number(id);
  return Number.isInteger(parsedId) && parsedId >= 0 ? parsedId : null;
}

export function categoryOf(name) {
  return PHASE_CATEGORY[name] || "idle";
}

export function colorFor(name) {
  return PHASE_COLORS[name] || CATEGORIES[categoryOf(name)].color;
}

function collect(timings) {
  const spans = [];
  for (const [id, lanes] of Object.entries(timings || {})) {
    const rolloutId = rolloutIdForTimingKey(id);
    for (const [role, lane] of Object.entries(lanes?.roles || {})) {
      if (lane?.lane_start_unix_s == null) continue;
      const laneStart = Number(lane?.lane_start_unix_s);
      if (!Number.isFinite(laneStart)) continue;
      for (const [name, phase] of Object.entries(lane?.phases || {})) {
        const count = Number(phase?.count) || 0;
        const total = Number(phase?.total_duration_s) || 0;
        if (!count || total < NEGLIGIBLE_WORK_S) continue;
        const where = {
          rolloutId,
          role,
          group: role === "rollout" ? "generation" : "step",
          name,
          category: categoryOf(name),
        };
        const runs =
          !SAMPLED.has(name) && Array.isArray(phase?.invocations)
            ? phase.invocations
            : [];
        if (runs.length) {
          for (const [from, to] of runs) {
            const start = laneStart + (Number(from) || 0);
            const end = laneStart + (Number(to) || 0);
            spans.push({
              ...where,
              kind: STALLS.has(name) ? "stall" : "work",
              start,
              end,
              count: 1,
              total: end - start,
              longest: end - start,
            });
          }
          continue;
        }
        spans.push({
          ...where,
          kind: STALLS.has(name)
            ? "stall"
            : SAMPLED.has(name) || count > 1
              ? "sampled"
              : "work",
          start: laneStart + (Number(phase?.first_start_s) || 0),
          end: laneStart + (Number(phase?.last_end_s) || 0),
          count,
          total,
          longest: Number(phase?.longest_duration_s) || 0,
        });
      }
    }
  }
  return spans;
}

function stepsOf(spans) {
  const byRollout = new Map();
  for (const span of spans) {
    if (span.role !== "driver" || span.rolloutId == null) continue;
    const step = byRollout.get(span.rolloutId) ?? {
      id: span.rolloutId,
      start: span.start,
      end: span.end,
      work: 0,
      stalled: 0,
    };
    step.start = Math.min(step.start, span.start);
    step.end = Math.max(step.end, span.end);
    if (span.kind === "stall") step.stalled += span.total;
    else step.work += span.total;
    byRollout.set(span.rolloutId, step);
  }
  return [...byRollout.values()].sort((a, b) => a.start - b.start);
}

function nest(spans) {
  const ordered = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const byRolloutAndName = new Map();
  for (const [index, span] of ordered.entries()) {
    span.orderIndex = index;
    const key = `${span.rolloutId}:${span.name}`;
    const bucket = byRolloutAndName.get(key) || [];
    bucket.push(span);
    byRolloutAndName.set(key, bucket);
  }
  for (const span of ordered) {
    let parent = null;
    for (const parentName of NESTS_IN[span.name] || []) {
      for (const other of byRolloutAndName.get(
        `${span.rolloutId}:${parentName}`,
      ) || []) {
        if (
          other !== span &&
          (other.group === span.group ||
            (span.name === "generate_samples" &&
              other.name === "generate_rollouts")) &&
          other.start <= span.start &&
          span.end <= other.end &&
          (!parent ||
            other.depth > parent.depth ||
            (other.depth === parent.depth &&
              other.orderIndex < parent.orderIndex))
        ) {
          parent = other;
        }
      }
    }
    span.depth = parent ? parent.depth + 1 : 0;
    span.parent = parent ?? null;
    span.children = [];
    if (parent) parent.contains = true;
    if (parent) parent.children.push(span);
  }
  for (const span of ordered) {
    const occurrences = new Map();
    for (const child of [...span.children].sort((a, b) => a.start - b.start)) {
      if (!["forward_backward", "optimizer_step"].includes(child.name)) continue;
      const occurrence = (occurrences.get(child.name) || 0) + 1;
      occurrences.set(child.name, occurrence);
      child.ordinal = occurrence;
    }
    const duration = Math.max(span.end - span.start, 0);
    const aggregates = new Map();
    for (const child of span.children) {
      if (!SAMPLED.has(child.name)) {
        continue;
      }
      const childDuration = Math.max(child.end - child.start, 0);
      const current = aggregates.get(child.name) ?? {
        name: child.name,
        kind: child.kind,
        category: child.category,
        role: child.role,
        rolloutId: child.rolloutId,
        duration: 0,
        total: 0,
        count: 0,
        longest: 0,
        start: child.start,
        end: child.end,
      };
      current.duration += childDuration;
      current.total += child.total ?? childDuration;
      current.count += child.count || 1;
      current.longest = Math.max(current.longest, child.longest || childDuration);
      current.start = Math.min(current.start, child.start);
      current.end = Math.max(current.end, child.end);
      aggregates.set(child.name, current);
    }
    const children = span.children.filter(
      (child) => !SAMPLED.has(child.name),
    );
    for (const child of aggregates.values()) {
      children.push({
        ...child,
        share: duration > 0 ? child.duration / duration : 0,
        average: child.count ? child.total / child.count : 0,
      });
    }
    span.children = children;
    delete span.orderIndex;
  }
  return ordered;
}

function mergeSyncGenerationSpans(spans) {
  const drivers = new Map(
    spans
      .filter(
        (span) =>
          span.role === "driver" &&
          span.name === "generate_rollouts" &&
          span.rolloutId != null,
      )
      .map((span) => [span.rolloutId, span]),
  );
  for (const span of spans) {
    if (span.role !== "rollout" || span.name !== "generate_samples") continue;
    const driver = drivers.get(span.rolloutId);
    if (!driver) continue;
    const sampleGeneration = nestedChild(span, "sample_generation");
    if (sampleGeneration) {
      driver.aggregateStats = {
        ...(driver.aggregateStats || {}),
        sample_generation: sampleGeneration,
      };
    }
    span.mergedGeneration = true;
    const descendants = [...(span.children || [])];
    while (descendants.length) {
      const descendant = descendants.pop();
      descendant.mergedGeneration = true;
      for (const child of descendant.children || []) {
        descendants.push(child);
      }
    }
    if (span.parent) {
      span.parent.children = span.parent.children.filter(
        (child) => child.name !== "generate_samples",
      );
    }
  }
  return spans;
}

function nestedChild(span, name) {
  for (const child of span.children || []) {
    if (child.name === name) return child;
    const nested = nestedChild(child, name);
    if (nested) return nested;
  }
  return null;
}

function clipStalls(spans, async) {
  const workByRow = new Map();
  for (const span of spans) {
    if (span.kind === "stall") continue;
    const row = async && span.role === "rollout" ? "generation" : "step";
    const intervals = workByRow.get(row) || [];
    intervals.push([span.start, span.end]);
    workByRow.set(row, intervals);
  }
  const clipped = [];
  for (const span of spans) {
    if (span.kind !== "stall") {
      clipped.push(span);
      continue;
    }
    let pieces = [[span.start, span.end]];
    for (const [workStart, workEnd] of workByRow.get(
      async && span.role === "rollout" ? "generation" : "step",
    ) || []) {
      pieces = pieces.flatMap(([start, end]) =>
        end <= workStart || start >= workEnd
          ? [[start, end]]
          : [
              ...(start < workStart ? [[start, workStart]] : []),
              ...(end > workEnd ? [[workEnd, end]] : []),
            ],
      );
    }
    for (const [start, end] of pieces) {
      if (end - start >= NEGLIGIBLE_WORK_S) {
        clipped.push({
          ...span,
          start,
          end,
          total: end - start,
          duration: end - start,
        });
      }
    }
  }
  return clipped;
}

function rowsOf(spans, async) {
  if (!spans.length) return [];
  const prepareRow = (row) => {
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
  const driverSpans = spans.filter((span) => !async || span.role !== "rollout");
  if (!async) {
    return [prepareRow(
      {
        key: "driver",
        label: "Train",
        hint: "Driver and trainer phases on the shared wall clock.",
        spans: driverSpans,
      },
    )];
  }

  const rows = [
    prepareRow({
      key: "driver",
      label: "Train",
      hint: "Driver and trainer phases on the shared wall clock.",
      spans: driverSpans,
    }),
  ];
  const rolloutSpans = spans.filter((span) => span.role === "rollout");
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
  for (const [index, packedRow] of packed.entries()) {
    const rootSet = new Set(packedRow.roots);
    rows.push(prepareRow({
      key: `rollout-${index}`,
      label: "Rollouts",
      hint: "Rollout engine phases packed by their actual wall-clock overlap.",
      spans: rolloutSpans.filter((span) => {
        let root = span;
        while (root.parent) root = root.parent;
        return rootSet.has(root);
      }),
    }));
  }
  return rows;
}

export function runTimeline(timings) {
  const measured = collect(timings);
  if (!measured.length) {
    return { span: 0, runStart: null, async: false, groups: [], steps: [], categories: [] };
  }
  // Avoid spreading one argument per invocation; long runs can exceed the call stack.
  let runStart = Infinity;
  let runEnd = -Infinity;
  for (const span of measured) {
    runStart = Math.min(runStart, span.start);
    runEnd = Math.max(runEnd, span.end);
  }
  const steps = stepsOf(measured);
  const rawSpans = measured;
  const generationSpans = rawSpans.filter((span) => span.role === "rollout");
  const stepSpans = rawSpans.filter(
    (span) =>
      span.role !== "rollout" &&
      span.kind !== "stall",
  );
  const sync = rawSpans.some(
    (span) => span.role === "driver" && span.name === "generate_rollouts",
  );
  const async = !sync && generationSpans.some((generation) =>
    stepSpans.some(
      (step) => generation.start < step.end && step.start < generation.end,
    ),
  );
  const spans = nest(clipStalls(rawSpans, async));
  if (sync) mergeSyncGenerationSpans(spans);

  for (const [index, span] of spans.entries()) {
    span.key = `${span.rolloutId}:${span.role}:${span.name}:${span.start.toFixed(3)}:${index}`;
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
      child.key ??= `${parent.key}:child:${child.name}`;
      hydrateNestedChildren(child);
    }
  }
  for (const span of spans) {
    hydrateNestedChildren(span);
  }

  const visibleSpans = spans;
  const rows = rowsOf(visibleSpans, async);
  const groups = rows.length ? [{ ...GROUPS[0], rows }] : [];
  for (const span of spans) delete span.parent;

  return {
    runStart,
    span: Math.max(runEnd - runStart, 1e-6),
    async,
    groups,
    steps: steps.map((step) => ({
      ...step,
      offset: step.start - runStart,
      duration: step.end - step.start,
    })),
    categories: [...new Set(spans.map((span) => span.category))].sort(
      (a, b) => Object.keys(CATEGORIES).indexOf(a) - Object.keys(CATEGORIES).indexOf(b),
    ),
  };
}

export function timingRunStart(timings) {
  const measured = collect(timings);
  // Avoid spreading one argument per invocation; long runs can exceed the call stack.
  let start = Infinity;
  for (const span of measured) {
    start = Math.min(start, span.start);
  }
  return measured.length ? start : null;
}

export function fmtSecs(s, unit = null) {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return "—";
  const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
  if (unit === "ms") return `${trim(n * 1000)}ms`;
  if (unit === "s") return `${trim(n)}s`;
  if (n > 0 && n < 0.01) return `${trim(n * 1000)}ms`;
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
