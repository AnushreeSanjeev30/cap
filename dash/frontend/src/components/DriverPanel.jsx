import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, LabelList } from "recharts";
import { useSoftPowerData } from "../lib/DataContext.jsx";
import { useDrivers } from "../lib/hooks";
import { DIMENSION_HEX, DIMENSION_META } from "../lib/colors";
import { describeFeature } from "../lib/featureDescriptions";
import "./Panel.css";

function DriverTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  const dimension = DIMENSION_META[row.dimension]?.label || "Other";
  return (
    <div className="chart-tooltip feature-tooltip">
      <div className="chart-tooltip-year">{row.feature}</div>
      <div className="tooltip-value mono">
        {row.shap > 0 ? "+" : ""}{row.shap.toFixed(3)} SHAP · {dimension}
      </div>
      <p>{describeFeature(row.raw, row.feature)}</p>
    </div>
  );
}

export default function DriverPanel({ focusCountry }) {
  const { nameMap, latestByIso3 } = useSoftPowerData();
  const { data: rows, loading } = useDrivers(focusCountry);
  const info = focusCountry ? latestByIso3[focusCountry] : null;
  const data = (rows || []).map((r) => ({ ...r, abs: Math.abs(r.shap) }));

  return (
    <section className="panel chart-panel driver-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">04</span> Driver Analysis</div>
        <div className="panel-note">{focusCountry ? `${nameMap[focusCountry] || focusCountry} \u00b7 model attribution` : "Select a country"}</div>
      </div>

      {!focusCountry ? (
        <div className="chart-empty">Pick a country from the ledger to see what is moving its score.</div>
      ) : loading && data.length === 0 ? (
        <div className="chart-empty">Loading driver attribution&hellip;</div>
      ) : (
        <>
          <p className="driver-summary">
            For <strong>{nameMap[focusCountry] || focusCountry}</strong>, the model's largest contributors to its current
            soft power score are shown below &mdash; bars extending right push the score up, bars extending
            left pull it down. {info?.kalman_regime && (
              <>Its current regime is classified as <strong>{info.kalman_regime}</strong>.</>
            )}
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data} layout="vertical" margin={{ top: 4, right: 40, left: 4, bottom: 4 }}>
              <CartesianGrid stroke="var(--border-soft)" horizontal={false} />
              <XAxis type="number" stroke="var(--text-faint)" tick={{ fontSize: 10, fontFamily: "var(--font-mono)" }} />
              <YAxis
                type="category"
                dataKey="feature"
                width={170}
                stroke="var(--text-faint)"
                tick={{ fontSize: 11.5, fill: "var(--text-dim)" }}
              />
              <Tooltip content={<DriverTooltip />} />
              <Bar dataKey="shap" radius={[4, 4, 4, 4]} isAnimationActive={false}>
                {data.map((d, i) => (
                  <Cell key={i} fill={DIMENSION_HEX[d.dimension] || DIMENSION_HEX.other} fillOpacity={d.direction === "up" ? 1 : 0.55} />
                ))}
                <LabelList
                  dataKey="shap"
                  position="right"
                  formatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`}
                  style={{ fill: "var(--text-dim)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="dim-legend">
            {Object.values(DIMENSION_META).map((d) => (
              <span className="dim-legend-item" key={d.key}>
                <span className="dim-legend-dot" style={{ background: `var(--${d.key.toLowerCase()})` }} />
                {d.short}
              </span>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
