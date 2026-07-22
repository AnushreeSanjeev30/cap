import { ComposedChart, ResponsiveContainer, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { useSoftPowerData } from "../lib/DataContext.jsx";
import { useForecast, useTimeseries } from "../lib/hooks";
import { fmtScore } from "../lib/format";
import "./Panel.css";

export default function ForecastPanel({ focusCountry }) {
  const { nameMap, latestByIso3 } = useSoftPowerData();
  const info = focusCountry ? latestByIso3[focusCountry] : null;
  const { data: fc, loading: fcLoading } = useForecast(focusCountry);
  const { data: tsMap, loading: histLoading } = useTimeseries(focusCountry ? [focusCountry] : [], 2018, 2024);
  const history = focusCountry ? tsMap[focusCountry] || [] : [];

  const data = [
    ...history.map((h) => ({ year: h.year, actual: h.score })),
    ...(fc || []).map((f) => ({ year: f.year, forecast: f.score, band: [f.lo, f.hi] })),
  ];
  const loading = fcLoading || histLoading;

  return (
    <section className="panel chart-panel forecast-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">05</span> 5&ndash;Year Outlook</div>
        <div className="panel-note">{focusCountry ? `${nameMap[focusCountry] || focusCountry}` : "Select a country"}</div>
      </div>

      {!focusCountry ? (
        <div className="chart-empty">Pick a country to view its Kalman-filtered forecast.</div>
      ) : loading && data.length === 0 ? (
        <div className="chart-empty">Loading forecast&hellip;</div>
      ) : (
        <>
          <div className="badge-row">
            {info?.stability_class && <span className="badge">Stability: <strong>{info.stability_class}</strong></span>}
            {info?.volatility_tier && <span className="badge">Volatility: <strong>{info.volatility_tier}</strong></span>}
            {info?.rank && <span className="badge">Global rank: <strong>#{info.rank}</strong></span>}
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={data} margin={{ top: 6, right: 18, left: -6, bottom: 0 }}>
              <CartesianGrid stroke="var(--border-soft)" vertical={false} />
              <XAxis dataKey="year" stroke="var(--text-faint)" tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }} />
              <YAxis stroke="var(--text-faint)" tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }} width={34} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: "#0d1119", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                formatter={(v, n) => (n === "band" ? [`${v[0].toFixed(1)} \u2013 ${v[1].toFixed(1)}`, "95% CI"] : [fmtScore(v), n === "actual" ? "Observed" : "Forecast"])}
              />
              <Area dataKey="band" stroke="none" fill="var(--accent)" fillOpacity={0.12} isAnimationActive={false} />
              <Line type="monotone" dataKey="actual" stroke="var(--text-dim)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="forecast" stroke="var(--accent)" strokeWidth={2.25} strokeDasharray="5 4" dot={{ r: 3 }} connectNulls isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="panel-note forecast-note">Solid grey = observed score &middot; dashed teal = Kalman state-space forecast &middot; shaded band = 95% confidence interval.</p>
        </>
      )}
    </section>
  );
}
