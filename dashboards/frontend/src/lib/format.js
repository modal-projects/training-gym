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

export function getGroupTags(run) {
  const tags = run?.group_tags || run?.metadata?.group_tags;
  const groupId = tags?.group_id || run?.group_id || run?.metadata?.group_id || "";
  if (!groupId && (!tags || typeof tags !== "object")) return null;

  const overrides =
    tags?.overrides && typeof tags.overrides === "object" ? tags.overrides : {};
  const rawTags = Array.isArray(tags?.tags) ? tags.tags : [];
  const displayTags = rawTags.length
    ? rawTags
    : Object.entries(overrides).map(([key, value]) => ({
        key,
        label: key.split(".").at(-1)?.replace(/_/g, " ") || key,
        value,
      }));

  return {
    group_id: groupId,
    axes: Array.isArray(tags?.axes) ? tags.axes : Object.keys(overrides),
    overrides,
    tags: displayTags,
  };
}

export function formatTagValue(value) {
  if (value == null) return "—";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
