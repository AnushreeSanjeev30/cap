import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from "recharts";
import { useSoftPowerData } from "../lib/DataContext.jsx";
import { colorForIndex } from "../lib/colors";
import { describeFeature } from "../lib/featureDescriptions";
import "./Panel.css";

function GlobalTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip feature-tooltip">
      <div className="chart-tooltip-year">{row.feature}</div>
      <div className="tooltip-value mono">{row.importance.toFixed(4)} importance</div>
      <p>{describeFeature(row.raw || row.feature.toLowerCase().replaceAll(" ", "_"), row.feature)}</p>
    </div>
  );
}

export default function GlobalDriversPanel() {
  const { globalImportance, loading } = useSoftPowerData();
  const data = (globalImportance || []).slice(0, 12);
  return (
    <section className="panel chart-panel global-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">06</span> What Moves Soft Power Globally</div>
        <div className="panel-note">Ensemble feature importance, all 195 states</div>
      </div>
      <p className="driver-summary">
        Aggregated across the ensemble model, these indicators carry the most weight in predicting
        soft power scores worldwide &mdash; a general map of what tends to matter, independent of any single country.
      </p>
      {loading && data.length === 0 ? (
        <div className="chart-empty">Loading global model weights&hellip;</div>
      ) : (
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
            <CartesianGrid stroke="var(--border-soft)" horizontal={false} />
            <XAxis type="number" stroke="var(--text-faint)" tick={{ fontSize: 10, fontFamily: "var(--font-mono)" }} />
            <YAxis type="category" dataKey="feature" width={190} stroke="var(--text-faint)" tick={{ fontSize: 11.5, fill: "var(--text-dim)" }} />
            <Tooltip content={<GlobalTooltip />} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]} isAnimationActive={false}>
              {data.map((_, i) => (
                <Cell key={i} fill={colorForIndex(i)} fillOpacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </section>
  );
}
