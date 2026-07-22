import { useEffect, useState } from "react";
import { fmtDelta, fmtScore } from "../lib/format";
import { api } from "../lib/api";
import Sparkline from "./Sparkline";
import "./Panel.css";
import "./Leaderboard.css";

export default function Leaderboard({ yStart, yEnd, deltas, loading, focusCountry, onFocusCountry }) {
  const [tab, setTab] = useState("risers");
  const [sparklines, setSparklines] = useState({});

  const list = deltas || [];
  const sorted = [...list].sort((a, b) => (tab === "risers" ? b.delta - a.delta : a.delta - b.delta));
  const shown = sorted.slice(0, 12);

  useEffect(() => {
    if (!shown.length) return;
    const iso3List = shown.map((r) => r.iso3);
    let cancelled = false;
    api
      .timeseries(iso3List, yStart, yEnd)
      .then((data) => !cancelled && setSparklines(data))
      .catch(() => !cancelled && setSparklines({}));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, yStart, yEnd, JSON.stringify(shown.map((r) => r.iso3))]);

  return (
    <section className="panel chart-panel leaderboard-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="tag-index">03</span> Rise &amp; Fall Ledger</div>
        <div className="lb-tabs">
          <button className={`lb-tab ${tab === "risers" ? "is-active" : ""}`} onClick={() => setTab("risers")}>Top Risers</button>
          <button className={`lb-tab ${tab === "fallers" ? "is-active" : ""}`} onClick={() => setTab("fallers")}>Top Fallers</button>
        </div>
      </div>
      <p className="panel-note lb-hint">Ranked by change in composite score, {yStart} &rarr; {yEnd}. Click a row to load it into Driver Analysis &amp; Forecast below.</p>

      {loading && shown.length === 0 ? (
        <div className="chart-empty">Loading live rankings&hellip;</div>
      ) : (
        <div className="lb-list">
          {shown.map((row) => {
            const pts = sparklines[row.iso3] || [];
            const isFocus = row.iso3 === focusCountry;
            return (
              <button
                key={row.iso3}
                className={`lb-row ${isFocus ? "is-focus" : ""}`}
                onClick={() => onFocusCountry(row.iso3)}
              >
                <span className="lb-name">
                  <span className="lb-country">{row.name}</span>
                  <span className="lb-iso mono">{row.iso3}</span>
                </span>
                <span className="lb-spark"><Sparkline points={pts} color={row.delta >= 0 ? "var(--up)" : "var(--down)"} /></span>
                <span className="lb-scores mono">
                  {fmtScore(row.startScore)} &rarr; {fmtScore(row.endScore)}
                </span>
                <span className={`lb-delta mono ${row.delta >= 0 ? "is-up" : "is-down"}`}>
                  {row.delta >= 0 ? "\u25B2" : "\u25BC"} {fmtDelta(row.delta)}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
