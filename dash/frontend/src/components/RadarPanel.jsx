import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip, Legend } from "recharts";
import { useSoftPowerData } from "../lib/DataContext.jsx";
import { pointNear } from "../lib/hooks";
import { colorForIndex, DIMENSION_META } from "../lib/colors";
import "./Panel.css";

const DIM_ORDER = ["D1", "D2", "D3", "D4", "D5"];

export default function RadarPanel({ selected, timeseries, loading, endYear }) {
  const { nameMap } = useSoftPowerData();
  const shown = selected.slice(0, 6);
  const data = DIM_ORDER.map((d) => {
    const row = { dimension: DIMENSION_META[d].short, full: DIMENSION_META[d].label };
    shown.forEach((iso3) => {
      const p = pointNear((timeseries || {})[iso3], endYear);
      row[iso3] = p ? p[d] ?? null : null;
    });
    return row;
  });

  return (
    <section className="panel chart-panel radar-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">02</span> Capital Profile</div>
        <div className="panel-note">Snapshot near {endYear}</div>
      </div>
      {shown.length === 0 ? (
        <div className="chart-empty">Select countries to compare their capital profile.</div>
      ) : loading ? (
        <div className="chart-empty">Loading capital profile&hellip;</div>
      ) : (
        <ResponsiveContainer width="100%" height={340}>
          <RadarChart data={data} outerRadius="72%">
            <PolarGrid stroke="var(--border-soft)" />
            <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11, fill: "var(--text-dim)", fontFamily: "var(--font-mono)" }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9, fill: "var(--text-faint)" }} tickCount={5} />
            {shown.map((iso3, i) => (
              <Radar
                key={iso3}
                name={nameMap[iso3] || iso3}
                dataKey={iso3}
                stroke={colorForIndex(i)}
                fill={colorForIndex(i)}
                fillOpacity={0.08}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
            <Tooltip
              contentStyle={{ background: "#0d1119", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
              labelFormatter={(label, payload) => payload?.[0]?.payload?.full || label}
            />
            <Legend wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
          </RadarChart>
        </ResponsiveContainer>
      )}
    </section>
  );
}
