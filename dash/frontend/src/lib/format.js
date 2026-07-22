export function fmtScore(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(1);
}

export function fmtDelta(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}`;
}

export function fmtRank(v) {
  if (v === null || v === undefined) return "—";
  return `#${v}`;
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
