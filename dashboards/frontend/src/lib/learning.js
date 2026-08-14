// Helpers shared by the learning-agent pages.

// Observatory run states → StatusPill variants. The observatory uses
// running / finished / failed (plus stale variants); anything unknown
// renders as a live run.
export function runPillStatus(state) {
  const s = String(state || "").toLowerCase();
  if (s === "finished" || s === "done" || s === "completed") return "completed";
  if (s === "failed" || s === "error") return "failed";
  if (s === "stale" || s === "stalled") return "stopped";
  return "running";
}

export function fmtScore(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(3);
}

export function fmtGpuHours(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(1)}h`;
}

export function fmtSeconds(s) {
  if (typeof s !== "number" || !Number.isFinite(s) || s <= 0) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

// Total classified learning-tool uses from index_row.learning_counts.
export function learningActionCount(run) {
  const counts = run?.learning_counts;
  if (!counts || typeof counts !== "object") return null;
  return Object.values(counts).reduce(
    (sum, v) => sum + (Number.isFinite(Number(v)) ? Number(v) : 0),
    0,
  );
}
