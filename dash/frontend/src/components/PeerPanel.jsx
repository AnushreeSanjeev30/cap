import { useSoftPowerData } from "../lib/DataContext.jsx";
import { usePeers } from "../lib/hooks";
import { fmtScore } from "../lib/format";
import "./Panel.css";

export default function PeerPanel({ focusCountry }) {
  const { nameMap } = useSoftPowerData();
  const { data, loading, error } = usePeers(focusCountry);
  const rows = data || [];

  return (
    <section className="panel chart-panel peer-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">07</span> Peer Country Analysis</div>
        <div className="panel-note">{focusCountry ? `${nameMap[focusCountry] || focusCountry} · vector similarity` : "Select a country"}</div>
      </div>

      {!focusCountry ? (
        <div className="chart-empty">Select any country to see its nearest peers from the embedding space.</div>
      ) : loading && rows.length === 0 ? (
        <div className="chart-empty">Loading peer countries...</div>
      ) : error ? (
        <div className="chart-empty">
          Peer endpoint is not available. Restart the backend so /api/peers/{focusCountry} is loaded.
        </div>
      ) : (
        rows.length === 0 ? (
          <div className="chart-empty">No peer countries found for {nameMap[focusCountry] || focusCountry}.</div>
        ) : (
          <div className="peer-list">
            {rows.map((row) => (
              <div className="peer-row" key={row.iso3}>
                <div className="peer-main">
                  <span className="peer-name">{row.name}</span>
                  <span className="peer-iso mono">{row.iso3}</span>
                </div>
                <div className="peer-bar" aria-label={`Similarity ${row.similarity}`}>
                  <span style={{ width: `${Math.max(2, Math.min(100, row.similarity * 100))}%` }} />
                </div>
                <div className="peer-meta mono">
                  <span>{(row.similarity * 100).toFixed(1)}% similar</span>
                  {row.rank && <span>#{row.rank}</span>}
                  {row.score !== null && row.score !== undefined && <span>{fmtScore(row.score)}</span>}
                </div>
                {row.explanation?.dimensions?.length > 0 && (
                  <div className="peer-explanation">
                    <span className="peer-explanation-label">Shared profile</span>
                    <span>{row.explanation.dimensions.map((dimension) => dimension.name).join(" · ")}</span>
                    {row.explanation.indicators?.length > 0 && (
                      <span className="peer-explanation-detail">
                        Strongest matches: {row.explanation.indicators.map((indicator) => indicator.label).join(", ")}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )
      )}
    </section>
  );
}
