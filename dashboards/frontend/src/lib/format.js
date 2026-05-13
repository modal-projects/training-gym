function unwrapValue(value) {
  if (value && typeof value === "object" && "value" in value) return value.value;
  return value;
}

export function toEpochSeconds(value) {
  const ts = unwrapValue(value);
  if (ts == null || ts === "") return null;

  if (typeof ts === "number") {
    if (!Number.isFinite(ts)) return null;
    if (ts > 1e12) return ts / 1000;
    return ts;
  }

  if (typeof ts === "string") {
    const trimmed = ts.trim();
    if (!trimmed) return null;

    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return toEpochSeconds(numeric);

    const parsedMs = Date.parse(trimmed);
    if (!Number.isNaN(parsedMs)) return parsedMs / 1000;
    return null;
  }

  if (ts instanceof Date) {
    const ms = ts.getTime();
    if (Number.isNaN(ms)) return null;
    return ms / 1000;
  }

  return null;
}

export function fmtDate(ts) {
  const seconds = toEpochSeconds(ts);
  if (seconds == null) return "—";
  const d = new Date(seconds * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

export function fmtDuration(start, end) {
  const startTs = toEpochSeconds(start);
  if (startTs == null) return "—";
  const endTs = toEpochSeconds(end) ?? Date.now() / 1000;
  let secs = Math.max(0, Math.floor(endTs - startTs));
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function fmtCluster(summary) {
  if (!summary) return "—";
  const gpu = summary.gpu_type || "?";
  const nodes = summary.actor_num_nodes || 0;
  const gpusPerNode = summary.actor_num_gpus_per_node || 0;
  if (!nodes && !gpusPerNode) return gpu || "—";
  const totalGpus = nodes * gpusPerNode;
  return `${totalGpus}x ${gpu}`;
}

export function fmtLr(lr) {
  if (!lr) return "—";
  if (lr < 0.001) return lr.toExponential(1);
  return String(lr);
}

export function truncateId(id) {
  if (!id) return "—";
  if (id.length <= 12) return id;
  return id.slice(0, 12) + "…";
}
