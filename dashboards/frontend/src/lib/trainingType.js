export function isSft(run) {
  return run?.training_type === "sft";
}

export function lossPoints(steps) {
  return steps
    .filter((step) => Number.isFinite(step.loss) && Number.isInteger(step.step))
    .map((step) => ({ x: step.step + 1, y: step.loss }));
}

export function lossEmptyMessage(run) {
  return run?.status === "running"
    ? "Waiting for the first completed training step."
    : "No training loss was recorded. Older runs may not include SFT metrics.";
}
