// "Nice" axis ticks: multiples of 1, 2 or 5 × 10^k that fall inside
// `[min, max]`, roughly `count` of them.
export function niceTicks(min, max, count = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min) || count < 1) return [];
  const rough = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= rough);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step / 1e6; v += step) {
    out.push(Number(v.toFixed(12)));
  }
  return out;
}
