// Shared time-range model for the run dashboard charts, mirroring the
// `{ start, end, live }` range the Modal dashboard threads through its chart
// controls (`app-dashboard-navigation.ts`, `MetricsRangeDropdown.svelte`,
// `ChartControls.svelte`, `ZoomOutButton.svelte`). Times are epoch seconds.
//
// Training-run charts plot against rollout ids rather than wall-clock time, so
// this module also owns the piecewise-linear mapping between a rollout id and
// the moment that rollout was recorded, used to translate a brush selection on
// a chart back into a time range.

import { toEpochSeconds } from "./format.js";

const MIN = 60;
const HOUR = MIN * 60;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;
const MONTH = 30 * DAY;
const YEAR = 365 * DAY;

const DURATION_UNITS = [
  { unit: "y", claims: YEAR - DAY, length: YEAR, round: true },
  { unit: "mo", claims: MONTH - HOUR, length: MONTH, round: true },
  { unit: "w", claims: WEEK - HOUR, length: WEEK, round: true },
  { unit: "d", claims: DAY - MIN, length: DAY, round: false },
  { unit: "h", claims: HOUR, length: HOUR, round: false },
  { unit: "m", claims: MIN, length: MIN, round: false },
  { unit: "s", claims: 1, length: 1, round: false },
];

/** Format seconds to the single largest fitting unit: 3600 → "1h", 90061 → "1d". */
export function formatDurationShort(seconds) {
  for (const { unit, claims, length, round } of DURATION_UNITS) {
    if (seconds < claims) continue;
    const value = round ? Math.round(seconds / length) : Math.floor(seconds / length);
    return `${Math.max(1, value)}${unit}`;
  }
  return "0s";
}

export function getTimezoneAbbr(date = new Date()) {
  try {
    return (
      new Intl.DateTimeFormat("en-US", { timeZoneName: "short" })
        .formatToParts(date)
        .find((p) => p.type === "timeZoneName")?.value ?? ""
    );
  } catch {
    return "";
  }
}

export function getTimeRangeParams(timeRange) {
  const start = timeRange?.start;
  const end = timeRange?.end;
  const live = timeRange?.live ?? false;
  const duration = start !== undefined && end !== undefined ? end - start : undefined;
  return { start, end, live, duration };
}

/**
 * Resolve the user's selection against the run's clock. `selection` is
 * `null` for "entire run"; otherwise `{ start, end, live }`. In live mode the
 * window keeps its duration and slides so that `end === now`.
 */
export function resolveTimeRange(selection, { runStart, now }) {
  if (!selection) {
    return { start: runStart, end: now, live: true, entireRun: true };
  }
  const { start, end, live } = selection;
  if (live) {
    const duration = Math.max(0, end - start);
    return { start: now - duration, end: now, live: true, entireRun: false };
  }
  return { start, end, live: false, entireRun: false };
}

/** Port of Modal's `ZoomOutButton`: triple the window, kept centred, capped at `now`. */
export function zoomOutRange({ start, end, live }, { now, maxDuration }) {
  const duration = end - start;
  const newDuration = Math.min(3 * duration, maxDuration);
  const durationIncrease = newDuration - duration;
  const newEnd = Math.min(end + durationIncrease / 2, now);
  return { start: newEnd - newDuration, end: newEnd, live: live && newEnd === now };
}

export function shiftRangeLeft({ start, end }) {
  const duration = end - start;
  return { start: start - duration, end: end - duration, live: false };
}

export function shiftRangeRight({ start, end }, { now }) {
  const duration = end - start;
  const newEnd = Math.min(end + duration, now);
  return { start: newEnd - duration, end: newEnd, live: newEnd === now };
}

/**
 * Sorted `(x, t)` knots — one per rollout that carries a timestamp — with
 * `t` forced monotonic so the mapping inverts cleanly.
 */
export function rolloutTimeKnots(rollouts) {
  const knots = [];
  for (const r of rollouts || []) {
    const x = Number(r?.rollout_id);
    const t = toEpochSeconds(r?.created_at);
    if (!Number.isFinite(x) || t == null || t <= 0) continue;
    knots.push({ x, t });
  }
  knots.sort((a, b) => a.x - b.x);
  for (let i = 1; i < knots.length; i++) {
    if (knots[i].t < knots[i - 1].t) knots[i].t = knots[i - 1].t;
  }
  return knots;
}

function interpolate(knots, key, otherKey, value, fallbackSlope) {
  if (!knots.length) return value;
  if (knots.length === 1) {
    return knots[0][otherKey] + (value - knots[0][key]) * fallbackSlope;
  }
  let lo = 0;
  let hi = knots.length - 1;
  if (value <= knots[lo][key]) hi = 1;
  else if (value >= knots[hi][key]) lo = hi - 1;
  else {
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (knots[mid][key] <= value) lo = mid;
      else hi = mid;
    }
  }
  const a = knots[lo];
  const b = knots[hi];
  const span = b[key] - a[key];
  const slope = span ? (b[otherKey] - a[otherKey]) / span : fallbackSlope;
  return a[otherKey] + (value - a[key]) * slope;
}

// Seconds per rollout assumed when the data can't tell us (single rollout,
// or two rollouts written in the same second).
const DEFAULT_SECONDS_PER_ROLLOUT = 60;

/** Rollout id (fractional allowed) → epoch seconds, extrapolating past the ends. */
export function rolloutToTime(knots, x) {
  return interpolate(knots, "x", "t", x, DEFAULT_SECONDS_PER_ROLLOUT);
}

/** Epoch seconds → rollout id (fractional). */
export function timeToRollout(knots, t) {
  return interpolate(knots, "t", "x", t, 1 / DEFAULT_SECONDS_PER_ROLLOUT);
}

/**
 * Translate an x-domain selection on a rollout-indexed chart into a time
 * range. A selection that spans the whole run collapses back to `null`
 * (entire run) so the controls read "entire run" instead of a near-identical
 * explicit window.
 */
export function rolloutDomainToTimeRange(knots, [x0, x1], { runStart, now }) {
  const lo = Math.min(x0, x1);
  const hi = Math.max(x0, x1);
  let start = rolloutToTime(knots, lo);
  let end = rolloutToTime(knots, hi);
  start = Math.max(runStart, start);
  end = Math.min(now, end);
  if (!(end > start)) return null;
  if (start <= runStart && end >= now) return null;
  return { start, end, live: end >= now };
}

/** Rollouts whose timestamp falls inside `[start, end]`. Untimed rollouts are dropped. */
export function filterRolloutsByRange(rollouts, range) {
  if (!range || range.entireRun) return rollouts;
  return (rollouts || []).filter((r) => {
    const t = toEpochSeconds(r?.created_at);
    return t != null && t >= range.start && t <= range.end;
  });
}
