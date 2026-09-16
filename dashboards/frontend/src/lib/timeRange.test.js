import assert from "node:assert/strict";
import test from "node:test";

import {
  filterRolloutsByRange,
  formatDurationShort,
  resolveTimeRange,
  rolloutDomainToTimeRange,
  rolloutTimeKnots,
  rolloutToTime,
  shiftRangeLeft,
  shiftRangeRight,
  timeToRollout,
  zoomOutRange,
} from "./timeRange.js";

const T0 = 1_700_000_000;
const rollouts = [
  { rollout_id: 0, created_at: T0 },
  { rollout_id: 1, created_at: T0 + 100 },
  { rollout_id: 2, created_at: T0 + 200 },
  { rollout_id: 4, created_at: T0 + 600 },
];

test("formatDurationShort picks the largest fitting unit", () => {
  assert.equal(formatDurationShort(3600), "1h");
  assert.equal(formatDurationShort(23 * 3600 + 59 * 60), "1d");
  assert.equal(formatDurationShort(90), "1m");
  assert.equal(formatDurationShort(0), "0s");
});

test("resolveTimeRange: entire run spans runStart..now, live windows slide", () => {
  assert.deepEqual(resolveTimeRange(null, { runStart: 10, now: 50 }), {
    start: 10,
    end: 50,
    live: true,
    entireRun: true,
  });
  const live = resolveTimeRange({ start: 0, end: 30, live: true }, { runStart: 10, now: 100 });
  assert.deepEqual(live, { start: 70, end: 100, live: true, entireRun: false });
  const paused = resolveTimeRange({ start: 20, end: 40, live: false }, { runStart: 10, now: 100 });
  assert.deepEqual(paused, { start: 20, end: 40, live: false, entireRun: false });
});

test("zoomOutRange triples the window, centred, capped at now", () => {
  const out = zoomOutRange({ start: 100, end: 200, live: false }, { now: 1000, maxDuration: 1e6 });
  assert.deepEqual(out, { start: 0, end: 300, live: false });
  const pinned = zoomOutRange({ start: 900, end: 1000, live: true }, { now: 1000, maxDuration: 1e6 });
  assert.deepEqual(pinned, { start: 700, end: 1000, live: true });
  const capped = zoomOutRange({ start: 0, end: 100, live: false }, { now: 1000, maxDuration: 150 });
  assert.equal(capped.end - capped.start, 150);
});

test("shift left/right move by one duration and re-arm live at now", () => {
  assert.deepEqual(shiftRangeLeft({ start: 100, end: 200 }), { start: 0, end: 100, live: false });
  assert.deepEqual(shiftRangeLeft({ start: 50, end: 150 }, { minStart: 0 }), {
    start: 0,
    end: 100,
    live: false,
  });
  assert.deepEqual(shiftRangeRight({ start: 100, end: 200 }, { now: 1000 }), {
    start: 200,
    end: 300,
    live: false,
  });
  assert.deepEqual(shiftRangeRight({ start: 850, end: 950 }, { now: 1000 }), {
    start: 900,
    end: 1000,
    live: true,
  });
});

test("rollout<->time mapping interpolates between knots and extrapolates past them", () => {
  const knots = rolloutTimeKnots(rollouts);
  assert.equal(knots.length, 4);
  assert.equal(rolloutToTime(knots, 1), T0 + 100);
  assert.equal(rolloutToTime(knots, 1.5), T0 + 150);
  assert.equal(rolloutToTime(knots, 3), T0 + 400);
  assert.equal(rolloutToTime(knots, 5), T0 + 800);
  assert.equal(rolloutToTime(knots, -1), T0 - 100);
  assert.equal(timeToRollout(knots, T0 + 150), 1.5);
  assert.equal(timeToRollout(knots, T0 + 400), 3);
});

test("rollout<->time mapping tolerates a single knot and unordered/untimed input", () => {
  const single = rolloutTimeKnots([{ rollout_id: 3, created_at: T0 }]);
  assert.equal(rolloutToTime(single, 4), T0 + 60);
  const messy = rolloutTimeKnots([
    { rollout_id: 2, created_at: T0 + 50 },
    { rollout_id: 1, created_at: T0 + 100 },
    { rollout_id: 5, created_at: null },
    { rollout_id: "x", created_at: T0 },
  ]);
  assert.deepEqual(messy, [
    { x: 1, t: T0 + 100 },
    { x: 2, t: T0 + 100.5 },
  ]);
});

test("rollouts recorded in the same second still map to a non-empty window", () => {
  const knots = rolloutTimeKnots([
    { rollout_id: 3, created_at: T0 },
    { rollout_id: 4, created_at: T0 + 100 },
    { rollout_id: 5, created_at: T0 + 100 },
    { rollout_id: 6, created_at: T0 + 100 },
    { rollout_id: 7, created_at: T0 + 100.2 },
  ]);
  for (let i = 1; i < knots.length; i++) assert.ok(knots[i].t > knots[i - 1].t);
  assert.ok(knots[3].t < T0 + 100.2);
  const range = rolloutDomainToTimeRange(knots, [4, 5], { runStart: T0, now: T0 + 1000 });
  assert.ok(range && range.end > range.start);
  assert.ok(Math.abs(timeToRollout(knots, range.start) - 4) < 1e-9);
  assert.ok(Math.abs(timeToRollout(knots, range.end) - 5) < 1e-9);
});

test("rolloutDomainToTimeRange clamps to the run and collapses a full-run selection", () => {
  const knots = rolloutTimeKnots(rollouts);
  const clock = { runStart: T0, now: T0 + 1000 };
  assert.deepEqual(rolloutDomainToTimeRange(knots, [1, 2], clock), {
    start: T0 + 100,
    end: T0 + 200,
    live: false,
  });
  assert.deepEqual(rolloutDomainToTimeRange(knots, [2, 1], clock), {
    start: T0 + 100,
    end: T0 + 200,
    live: false,
  });
  assert.equal(rolloutDomainToTimeRange(knots, [-10, 50], clock), null);
  assert.deepEqual(rolloutDomainToTimeRange(knots, [2, 50], clock), {
    start: T0 + 200,
    end: T0 + 1000,
    live: true,
  });
  assert.equal(rolloutDomainToTimeRange(knots, [2, 2], clock), undefined);
});

test("rolloutDomainToTimeRange snaps a window panned past either end to that boundary", () => {
  const knots = rolloutTimeKnots(rollouts);
  const clock = { runStart: T0, now: T0 + 1000 };
  // Ids past rollout 4 extrapolate at 200s each, so [12, 13] lies entirely
  // after `now`; it comes back as the last 200s.
  assert.deepEqual(rolloutDomainToTimeRange(knots, [12, 13], clock), {
    start: T0 + 800,
    end: T0 + 1000,
    live: true,
  });
  assert.deepEqual(rolloutDomainToTimeRange(knots, [-3, -2], clock), {
    start: T0,
    end: T0 + 100,
    live: false,
  });
});

test("filterRolloutsByRange keeps timed rollouts inside the window", () => {
  const all = filterRolloutsByRange(rollouts, { entireRun: true });
  assert.equal(all, rollouts);
  const windowed = filterRolloutsByRange(rollouts, { start: T0 + 100, end: T0 + 250 });
  assert.deepEqual(
    windowed.map((r) => r.rollout_id),
    [1, 2],
  );
  assert.deepEqual(filterRolloutsByRange([{ rollout_id: 9 }], { start: 0, end: 1e12 }), []);
});
