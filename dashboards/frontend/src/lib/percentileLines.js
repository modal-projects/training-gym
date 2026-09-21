export const PERCENTILE_LINES = [
  { key: "pmin", stat: "min", label: "pmin", color: "var(--color-c-gray-40, #747474)", dash: "3 3" },
  { key: "p50", stat: "p50", label: "p50", color: "var(--color-c-yellow-80, #d1c05f)" },
  { key: "p90", stat: "p90", label: "p90", color: "var(--color-c-orange-80, #d18e50)" },
  { key: "p99", stat: "p99", label: "p99", color: "var(--color-c-red-80, #cb5f5f)" },
  { key: "pmax", stat: "max", label: "pmax", color: "var(--color-c-gray-60, #a3a3a3)", dash: "3 3" },
];

function finiteOrNull(value) {
  const n = Number(value);
  return value == null || !Number.isFinite(n) ? null : n;
}

export function percentileRowFields(stats) {
  const out = {};
  for (const line of PERCENTILE_LINES) out[line.key] = finiteOrNull(stats?.[line.stat]);
  return out;
}
