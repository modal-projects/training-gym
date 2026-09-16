// Trailing moving average over chart rows. Each output row's `keys` are the
// mean of that key over the current row and the `window - 1` rows before it
// (fewer at the start of the series). Rows whose value for a key is not a
// finite number are skipped for that key and stay non-finite in the output.
function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function trailingMean(rows, keys, window = 5) {
  const size = Math.max(1, Math.floor(Number(window) || 1));
  if (size === 1 || !rows.length) return rows;
  return rows.map((row, index) => {
    const out = { ...row };
    const slice = rows.slice(Math.max(0, index - size + 1), index + 1);
    for (const key of keys) {
      if (!finite(row[key])) continue;
      const values = slice.map((r) => r[key]).filter(finite);
      out[key] = values.reduce((sum, v) => sum + v, 0) / values.length;
    }
    return out;
  });
}
