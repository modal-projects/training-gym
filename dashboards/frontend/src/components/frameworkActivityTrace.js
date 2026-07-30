const ROLE_ORDER = ["rollout", "driver", "actor", "critic", "step"];
const WAIT_PHASES = new Set(["wait_for_rollout", "wait_for_next_rollout"]);
const REWARD_PHASES = new Set([
  "custom_reward",
  "custom_reward_post_process",
]);
const ROOT_PHASES = {
  rollout: new Set([
    "generate_rollouts",
    "evaluate_rollouts",
    "evaluate_rollouts_before",
    "evaluate_rollouts_after",
  ]),
  training: new Set(["train_model"]),
};

const GROUPS = [
  {
    key: "rollout",
    label: "Rollout pipeline",
    summaryLabel: "Generation",
  },
  {
    key: "training",
    label: "Training pipeline",
    summaryLabel: "Training",
  },
  {
    key: "system",
    label: "System / waits",
    summaryLabel: "Coordination / I/O",
  },
  {
    key: "legacy",
    label: "Timing",
    summaryLabel: "Timing",
  },
];

export const TRACE_GROUP_COLORS = {
  rollout: "var(--color-c-dataviz-primary-1, #adeaab)",
  training: "var(--color-c-dataviz-primary-7, #648fe0)",
  system: "var(--color-c-gray-40, #747474)",
  legacy: "var(--color-c-dataviz-primary-other, #6d6161)",
};

export function timelineGroup(substep) {
  if (substep.timelineGroup) return substep.timelineGroup;
  if (substep.role === "step") return "legacy";
  if (REWARD_PHASES.has(substep.phase) || substep.role === "rollout") {
    return "rollout";
  }
  if (substep.role === "actor" || substep.role === "critic") {
    return "training";
  }
  return "system";
}

export function isWait(substep) {
  return substep.activityKind
    ? substep.activityKind === "wait"
    : WAIT_PHASES.has(substep.phase);
}

function mergeIntervals(intervals) {
  const ordered = intervals
    .filter(
      (interval) =>
        Number.isFinite(interval.start) &&
        Number.isFinite(interval.end) &&
        interval.end >= interval.start,
    )
    .sort((left, right) => left.start - right.start);
  const merged = [];
  for (const interval of ordered) {
    const current = merged.at(-1);
    if (!current || interval.start > current.end) {
      merged.push({ ...interval });
    } else {
      current.end = Math.max(current.end, interval.end);
    }
  }
  return merged;
}

function totalDuration(intervals) {
  return intervals.reduce(
    (total, interval) => total + interval.end - interval.start,
    0,
  );
}

function overlappingDuration(left, right) {
  let leftIndex = 0;
  let rightIndex = 0;
  let total = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const start = Math.max(left[leftIndex].start, right[rightIndex].start);
    const end = Math.min(left[leftIndex].end, right[rightIndex].end);
    total += Math.max(0, end - start);
    if (left[leftIndex].end < right[rightIndex].end) {
      leftIndex += 1;
    } else {
      rightIndex += 1;
    }
  }
  return total;
}

export function buildActivityMetrics(steps, rolloutStats) {
  const substeps = steps.flatMap((step) =>
    step.substeps.map((substep) => ({
      ...substep,
      step: step.n,
      start: Number(substep.start),
      end: Number(substep.start) + Number(substep.duration),
    })),
  );
  const rolloutIntervals = mergeIntervals(
    substeps.filter(
      (substep) =>
        timelineGroup(substep) === "rollout" &&
        substep.phase === "generate_rollouts",
    ),
  );
  const trainingIntervals = mergeIntervals(
    substeps.filter(
      (substep) =>
        timelineGroup(substep) === "training" &&
        substep.phase === "train_model",
    ),
  );
  const sourceRolloutIds = new Set(
    substeps
      .filter(
        (substep) =>
          substep.phase === "generate_rollouts" &&
          substep.activityRolloutId != null,
      )
      .map((substep) => Number(substep.activityRolloutId)),
  );
  const samples = (rolloutStats || [])
    .filter((rollout) => sourceRolloutIds.has(Number(rollout.rollout_id)))
    .reduce((total, rollout) => total + (Number(rollout.total) || 0), 0);
  const rolloutDuration = totalDuration(rolloutIntervals);
  const trainingDuration = totalDuration(trainingIntervals);
  const overlap = overlappingDuration(rolloutIntervals, trainingIntervals);
  const executions = new Map();
  for (const substep of substeps) {
    const sequence = Number(substep.executionSequence);
    if (!Number.isFinite(sequence)) continue;
    const key = `${substep.step}:${substep.role}`;
    executions.set(key, Math.max(executions.get(key) || 0, sequence));
  }
  return {
    hasFrameworkActivity: substeps.some(
      (substep) =>
        substep.timelineGroup ||
        (substep.role !== "step" && ROLE_ORDER.includes(substep.role)),
    ),
    rolloutCount: sourceRolloutIds.size,
    throughput:
      samples > 0 && rolloutDuration > 0 ? samples / rolloutDuration : null,
    overlap:
      trainingDuration > 0 ? Math.min(1, overlap / trainingDuration) : null,
    retries: Array.from(executions.values()).reduce(
      (total, sequence) => total + Math.max(0, sequence - 1),
      0,
    ),
  };
}

function trackLabel(group, substep, labelFor) {
  const phase = substep.displayName || labelFor(substep.phase);
  if (group !== "training") return phase;
  const role = substep.role;
  return `${phase} · ${role[0].toUpperCase()}${role.slice(1)}`;
}

function isSummaryPhase(group, substep) {
  if (group === "system" || group === "legacy") return true;
  if (substep.parentPhase) return false;
  return ROOT_PHASES[group]?.has(substep.phase) ?? false;
}

export function buildFrameworkTimeline(
  steps,
  stepTimes,
  expandedGroups,
  labelFor,
  phaseOrder,
) {
  const substeps = steps.flatMap((step) =>
    step.substeps
      .filter(
        (substep) =>
          Number.isFinite(Number(substep.start)) &&
          Number.isFinite(Number(substep.duration)),
      )
      .map((substep) => ({
        ...substep,
        step: step.n,
        end: Number(substep.start) + Number(substep.duration),
      })),
  );
  if (!substeps.length) {
    return { start: 0, duration: 0, groups: [], tracks: [] };
  }
  const stepStarts = steps
    .map((step) => Number((stepTimes || {})[step.key]?.start))
    .filter(Number.isFinite);
  const stepEnds = steps
    .map((step) => Number((stepTimes || {})[step.key]?.end))
    .filter(Number.isFinite);
  const start = Math.min(
    ...substeps.map((substep) => Number(substep.start)),
    ...stepStarts,
  );
  const end = Math.max(
    ...substeps.map((substep) => substep.end),
    ...stepEnds,
  );
  const duration = Math.max(end - start, 0.001);
  const position = (value) => ((value - start) / duration) * 100;
  const positionSubsteps = (items) =>
    items
      .sort((left, right) => (right.duration ?? 0) - (left.duration ?? 0))
      .map((substep) => ({
        ...substep,
        left: Math.max(0, Math.min(100, position(Number(substep.start)))),
        width: Math.max(
          0,
          Math.min(
            100 - position(Number(substep.start)),
            (Number(substep.duration) / duration) * 100,
          ),
        ),
      }));
  const groups = [];
  for (const definition of GROUPS) {
    const groupSubsteps = substeps.filter(
      (substep) =>
        timelineGroup(substep) === definition.key &&
        substep.phase !== "full_step",
    );
    if (!groupSubsteps.length) continue;
    const summaryCandidates = groupSubsteps.filter((substep) =>
      isSummaryPhase(definition.key, substep),
    );
    const summarySubsteps = summaryCandidates.length
      ? summaryCandidates
      : groupSubsteps;
    const childSubsteps =
      definition.key === "system" || definition.key === "legacy"
        ? groupSubsteps
        : groupSubsteps.filter(
            (substep) => !isSummaryPhase(definition.key, substep),
          );
    const children = Array.from(
      childSubsteps.reduce((tracks, substep) => {
        const key = `${substep.role}:${substep.phase}`;
        if (!tracks.has(key)) {
          tracks.set(key, {
            key: `${definition.key}:${key}`,
            label: trackLabel(definition.key, substep, labelFor),
            substeps: [],
          });
        }
        tracks.get(key).substeps.push(substep);
        return tracks;
      }, new Map()).values(),
    )
      .sort((left, right) => {
        const leftPhase = left.key.split(":").at(-1);
        const rightPhase = right.key.split(":").at(-1);
        const leftIndex = phaseOrder.indexOf(leftPhase);
        const rightIndex = phaseOrder.indexOf(rightPhase);
        return (
          (leftIndex < 0 ? phaseOrder.length : leftIndex) -
            (rightIndex < 0 ? phaseOrder.length : rightIndex) ||
          left.label.localeCompare(right.label)
        );
      })
      .map((track) => ({
        ...track,
        group: definition.key,
        isSummary: false,
        substeps: positionSubsteps(track.substeps),
      }));
    groups.push({
      ...definition,
      expanded: expandedGroups.has(definition.key),
      summary: {
        key: `${definition.key}:summary`,
        label: definition.label,
        group: definition.key,
        expanded: expandedGroups.has(definition.key),
        expandable: children.length > 0,
        isSummary: true,
        substeps: positionSubsteps(summarySubsteps).map((substep) => ({
          ...substep,
          summaryColor: TRACE_GROUP_COLORS[definition.key],
        })),
      },
      children,
    });
  }
  return {
    start,
    duration,
    groups,
    tracks: groups.flatMap((group) => [
      group.summary,
      ...(group.expanded ? group.children : []),
    ]),
  };
}
