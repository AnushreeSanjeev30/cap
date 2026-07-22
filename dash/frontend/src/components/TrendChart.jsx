import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { useSoftPowerData } from "../lib/DataContext.jsx";
import { colorForIndex } from "../lib/colors";
import { METRICS } from "./ControlBar";
import "./Panel.css";

function buildChartData(selected, timeseries, metric) {
  const yearMap = new Map();
  selected.forEach((iso3) => {
    (timeseries[iso3] || []).forEach((r) => {
      if (!yearMap.has(r.year)) yearMap.set(r.year, { year: r.year });
      const v = r[metric];
      if (v !== undefined && v !== null) yearMap.get(r.year)[iso3] = v;
    });
  });
  return [...yearMap.values()].sort((a, b) => a.year - b.year);
}

function CustomTooltip({ active, payload, label, selected, nameMap }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-year mono">{label}</div>
      {selected.map((iso3, i) => {
        const entry = payload.find((p) => p.dataKey === iso3);
        if (!entry || entry.value === undefined || entry.value === null) return null;
        return (
          <div className="chart-tooltip-row" key={iso3}>
            <span className="chart-tooltip-dot" style={{ background: colorForIndex(i) }} />
            <span className="chart-tooltip-name">{nameMap[iso3] || iso3}</span>
            <span className="chart-tooltip-val mono">{entry.value?.toFixed?.(1) ?? entry.value}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function TrendChart({ selected, timeseries, loading, yStart, yEnd, metric }) {
  const { nameMap } = useSoftPowerData();
  const data = buildChartData(selected, timeseries || {}, metric);
  const metricLabel = METRICS.find((m) => m.key === metric)?.label || metric;

  return (
    <section className="panel chart-panel trend-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">01</span> Trajectory &middot; {metricLabel}</div>
        <div className="panel-note">Score range 0&ndash;100 &middot; {yStart}&ndash;{yEnd}</div>
      </div>
      {selected.length === 0 ? (
        <div className="chart-empty">Select at least one country to plot its trajectory.</div>
      ) : loading && data.length === 0 ? (
        <div className="chart-empty">Loading trajectory data&hellip;</div>
      ) : (
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={data} margin={{ top: 6, right: 18, left: -6, bottom: 0 }}>
            <CartesianGrid stroke="var(--border-soft)" vertical={false} />
            <XAxis dataKey="year" stroke="var(--text-faint)" tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }} tickMargin={8} />
            <YAxis stroke="var(--text-faint)" tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }} width={34} domain={metric === "score" ? [0, 100] : ["auto", "auto"]} />
            <Tooltip content={<CustomTooltip selected={selected} nameMap={nameMap} />} cursor={{ stroke: "var(--border)" }} />
            {selected.map((iso3, i) => (
              <Line
                key={iso3}
                type="monotone"
                dataKey={iso3}
                name={nameMap[iso3] || iso3}
                stroke={colorForIndex(i)}
                strokeWidth={2.25}
                dot={false}
                activeDot={{ r: 4 }}
                connectNulls
                isAnimationActive={false}
              />
            ))}
            <Legend wrapperStyle={{ fontSize: 12, fontFamily: "var(--font-mono)", paddingTop: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </section>
  );
}
