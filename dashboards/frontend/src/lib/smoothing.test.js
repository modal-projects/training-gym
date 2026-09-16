import assert from "node:assert/strict";
import test from "node:test";

import { trailingMean } from "./smoothing.js";

const rows = [1, 2, 3, 4, 5, 6, 7].map((y, i) => ({ x: i, y }));

test("trailingMean averages the current row with the previous window - 1 rows", () => {
  const out = trailingMean(rows, ["y"], 5);
  assert.deepEqual(
    out.map((r) => r.y),
    [1, 1.5, 2, 2.5, 3, 4, 5],
  );
  assert.deepEqual(
    out.map((r) => r.x),
    rows.map((r) => r.x),
  );
});

test("trailingMean smooths every requested key independently", () => {
  const out = trailingMean(
    [
      { x: 0, y: 0, p90: 10 },
      { x: 1, y: 2, p90: 20 },
    ],
    ["y", "p90"],
    2,
  );
  assert.deepEqual(out[1], { x: 1, y: 1, p90: 15 });
});

test("trailingMean skips non-finite values without breaking the window", () => {
  const out = trailingMean(
    [
      { x: 0, y: 2 },
      { x: 1, y: null },
      { x: 2, y: 4 },
    ],
    ["y"],
    3,
  );
  assert.equal(out[1].y, null);
  assert.equal(out[2].y, 3);
});

test("trailingMean with a window of 1 returns the rows unchanged", () => {
  assert.equal(trailingMean(rows, ["y"], 1), rows);
});
