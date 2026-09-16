function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

// Trailing mean over `window` rows; a non-finite value ends the window.
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
