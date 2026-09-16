// Trailing moving average over chart rows. Each output row's `keys` are the
// mean of that key over the current row and the `window - 1` rows before it
// (fewer at the start of the series). A row whose value for a key is not a
// finite number stays non-finite in the output and ends the window for the
// rows after it, so a gap is never averaged across.
function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function trailingMean(rows, keys, window = 5) {
  const size = Math.max(1, Math.floor(Number(window) || 1));
  if (size === 1 || !rows.length) return rows;
  return rows.map((row, index) => {
    const out = { ...row };
    for (const key of keys) {
      if (!finite(row[key])) continue;
      let sum = 0;
      let count = 0;
      for (let i = index; i >= 0 && index - i < size && finite(rows[i][key]); i--) {
        sum += rows[i][key];
        count++;
      }
      out[key] = sum / count;
    }
    return out;
  });
}
