import test from "node:test";
import assert from "node:assert/strict";
import { isSft, lossPoints, lossEmptyMessage } from "./trainingType.js";

test("training type defaults to RL and recognizes SFT", () => {
  assert.equal(isSft({}), false);
  assert.equal(isSft({ training_type: "sft" }), true);
  assert.equal(isSft({ training_type: "rl" }), false);
});

test("loss chart preserves zero, rejects missing values, and uses completed steps", () => {
  assert.deepEqual(lossPoints([
    { step: 0, loss: 2 }, { step: 1, loss: null }, { step: 2, loss: 0 }, { step: 3, loss: NaN },
  ]), [{ x: 1, y: 2 }, { x: 3, y: 0 }]);
});

test("empty states distinguish an active run from missing historical data", () => {
  assert.match(lossEmptyMessage({ status: "running" }), /first completed/);
  assert.match(lossEmptyMessage({ status: "completed" }), /No training loss/);
});
