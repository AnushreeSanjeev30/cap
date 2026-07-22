// Categorical palette used for comparing multiple selected countries across charts.
export const SERIES_PALETTE = [
  "#57c7b8", // teal
  "#e0a458", // amber
  "#a78bfa", // violet
  "#fb7185", // rose
  "#6ee7b7", // emerald
  "#60a5fa", // blue
  "#f0d264", // gold
  "#f472b6", // pink
  "#94a3b8", // slate
  "#34d399", // green
];

export function colorForIndex(i) {
  return SERIES_PALETTE[i % SERIES_PALETTE.length];
}

export const DIMENSION_META = {
  D1: { key: "D1", label: "Cultural Capital", color: "var(--d1)", short: "Culture" },
  D2: { key: "D2", label: "Innovation & Knowledge", color: "var(--d2)", short: "Innovation" },
  D3: { key: "D3", label: "Political Legitimacy", color: "var(--d3)", short: "Politics" },
  D4: { key: "D4", label: "Institutional Quality", color: "var(--d4)", short: "Institutions" },
  D5: { key: "D5", label: "Human Development", color: "var(--d5)", short: "Human Dev" },
};

export const DIMENSION_HEX = {
  D1: "#e0a458",
  D2: "#57c7b8",
  D3: "#a78bfa",
  D4: "#6ee7b7",
  D5: "#fb7185",
  other: "#8a97a8",
};
