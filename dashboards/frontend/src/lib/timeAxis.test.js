import assert from "node:assert/strict";
import test from "node:test";

import { formatTimeTick, pickTimeInterval, timeTicks } from "./timeAxis.js";

const local = (...parts) => new Date(...parts).getTime() / 1000;

test("picks the smallest calendar step that fits the tick budget", () => {
  assert.equal(pickTimeInterval(60 * 60, 6)[0], 15 * 60);
  assert.equal(pickTimeInterval(10 * 60, 5)[0], 5 * 60);
  assert.equal(pickTimeInterval(20, 5)[0], 5);
  assert.equal(pickTimeInterval(3 * 24 * 3600, 3)[0], 24 * 3600);
});

test("ticks snap to local clock boundaries and stay inside the range", () => {
  const start = local(2026, 8, 15, 10, 7, 30);
  const end = local(2026, 8, 15, 11, 2, 0);
  const ticks = timeTicks(start, end, 6);
  assert.ok(ticks.length >= 3 && ticks.length <= 6, String(ticks.length));
  for (const t of ticks) {
    assert.ok(t >= start && t <= end);
    const d = new Date(t * 1000);
    assert.equal(d.getSeconds(), 0);
    assert.equal(d.getMinutes() % 15, 0);
  }
  assert.equal(ticks[0], local(2026, 8, 15, 10, 15));
});

test("day-scale ticks land on local midnight", () => {
  const start = local(2026, 8, 13, 5);
  const end = local(2026, 8, 17, 20);
  const ticks = timeTicks(start, end, 4);
  assert.ok(ticks.length >= 2);
  for (const t of ticks) {
    const d = new Date(t * 1000);
    assert.equal(d.getHours(), 0);
    assert.equal(d.getMinutes(), 0);
  }
});

test("returns nothing for an empty or inverted range", () => {
  assert.deepEqual(timeTicks(10, 10, 5), []);
  assert.deepEqual(timeTicks(20, 10, 5), []);
  assert.deepEqual(timeTicks(NaN, 10, 5), []);
});

test("labels use the coarsest unit that changed", () => {
  assert.equal(formatTimeTick(local(2026, 8, 15, 10, 7, 30)), ":30");
  assert.equal(formatTimeTick(local(2026, 8, 15, 10, 15)), "10:15");
  assert.equal(formatTimeTick(local(2026, 8, 15, 22, 15)), "10:15");
  assert.equal(formatTimeTick(local(2026, 8, 15, 9)), "09 AM");
  assert.equal(formatTimeTick(local(2026, 8, 15, 12)), "12 PM");
  assert.equal(formatTimeTick(local(2026, 8, 15, 0)), "Tue 15");
  assert.equal(formatTimeTick(local(2026, 8, 1)), "September");
  assert.equal(formatTimeTick(local(2026, 0, 1)), "2026");
});
