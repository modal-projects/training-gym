// Helpers for the Metrics tab: W&B-style panel grouping of mirrored metric
// keys, plus the `[step, value]` rows -> LineChart rows conversion.

// Keys without a `/` prefix land in the same panel group W&B uses for them.
export const UNGROUPED = "Charts";

export function groupOf(key) {
  const slash = String(key).indexOf("/");
  return slash > 0 ? key.slice(0, slash) : UNGROUPED;
}

// Every whitespace-separated term must appear somewhere in the key
// (case-insensitive), so "loss train" and "train/loss" both find `train/loss`.
export function matchesSearch(key, search) {
  const terms = String(search || "")
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  if (!terms.length) return true;
  const haystack = String(key).toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

const GROUP_ORDER = ["train", "rollout", "eval", "perf"];

function groupRank(name) {
  if (name === UNGROUPED) return -1;
  const index = GROUP_ORDER.indexOf(name);
  return index === -1 ? GROUP_ORDER.length : index;
}

// `[{ name, keys }]` sorted so the groups people look at first come first,
// then alphabetically; keys inside a group are alphabetical.
export function groupMetricKeys(keys, search = "") {
  const groups = new Map();
  for (const key of keys || []) {
    if (typeof key !== "string" || !key || !matchesSearch(key, search)) continue;
    const name = groupOf(key);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(key);
  }
  return [...groups.entries()]
    .map(([name, members]) => ({ name, keys: members.sort() }))
    .sort(
      (a, b) => groupRank(a.name) - groupRank(b.name) || a.name.localeCompare(b.name),
    );
}

// `[[step, value], ...]` -> `[{ x, y }]`, dropping anything non-finite.
export function seriesToRows(rows) {
  if (!Array.isArray(rows)) return [];
  const out = [];
  for (const row of rows) {
    if (!Array.isArray(row) || row[0] == null || row[1] == null) continue;
    const x = Number(row[0]);
    const y = Number(row[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) out.push({ x, y });
  }
  return out;
}

export function seriesStats(rows) {
  if (!rows.length) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const row of rows) {
    if (row.y < min) min = row.y;
    if (row.y > max) max = row.y;
  }
  return { min, max, latest: rows[rows.length - 1].y, count: rows.length };
}

// Compact numeric formatting for axis/tooltip/stat readouts: integers stay
// integers, small magnitudes switch to exponent form, everything else gets
// four significant digits.
export function formatMetricValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (Number.isInteger(value) && Math.abs(value) < 1e9) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
    return value.toExponential(2);
  }
  return String(Number(value.toPrecision(4)));
}
