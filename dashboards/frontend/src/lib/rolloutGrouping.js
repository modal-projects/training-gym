export function rolloutIndex(sample) {
  const index = sample?.rollout_index ?? sample?.metadata?.rollout_id;
  if (index == null || index === "") return null;
  const n = Number(index);
  return Number.isFinite(n) ? n : null;
}

export function groupByRollout(samples) {
  const groups = new Map();
  (samples || []).forEach((sample, i) => {
    const index = rolloutIndex(sample);
    const key = index ?? `sample:${i}`;
    const group = groups.get(key);
    if (group) group.push(i);
    else groups.set(key, [i]);
  });
  return [...groups.values()];
}

export function rolloutScores(samples, groups = groupByRollout(samples)) {
  return groups.map((positions) => {
    const sum = positions.reduce((a, i) => a + (Number(samples[i]?.score) || 0), 0);
    return sum / positions.length;
  });
}
