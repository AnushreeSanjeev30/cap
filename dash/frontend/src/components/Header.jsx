import { fmtDelta } from "../lib/format";
import "./Header.css";

export default function Header({ yStart, yEnd, deltas, deltasLoading }) {
  const list = deltas || [];
  const sorted = [...list].sort((a, b) => b.delta - a.delta);
  const risers = sorted.slice(0, 10);
  const fallers = sorted.slice(-10).reverse();

  const tape = [];
  const max = Math.max(risers.length, fallers.length);
  for (let i = 0; i < max; i++) {
    if (risers[i]) tape.push({ ...risers[i], dir: "up" });
    if (fallers[i]) tape.push({ ...fallers[i], dir: "down" });
  }

  return (
    <header className="hdr">
      <div className="hdr-top">
        <div className="hdr-id">
          <div className="eyebrow">Global Influence Tracking &middot; {yStart}&ndash;{yEnd}</div>
          <h1 className="hdr-title">Soft&nbsp;Power Intelligence</h1>
        </div>
        <div className="hdr-meta">
          <div className="hdr-meta-row">
            <span className="tag-index">195 STATES</span>
            <span className="tag-index">2000&ndash;2024</span>
            <span className="tag-index">5 CAPITAL DIMENSIONS</span>
          </div>
          <p className="hdr-sub">
            A comparative ledger of national soft power &mdash; composite scores, trajectories,
            and the measured drivers behind each rise or fall.
          </p>
        </div>
      </div>

      <div className="ticker" role="marquee" aria-label="Top soft power movers">
        <div className="ticker-track">
          {deltasLoading || tape.length === 0 ? (
            <span className="ticker-item ticker-loading mono">Loading live rankings&hellip;</span>
          ) : (
            [...tape, ...tape].map((t, i) => (
              <span className={`ticker-item ticker-${t.dir}`} key={i}>
                <span className="ticker-arrow">{t.dir === "up" ? "\u25B2" : "\u25BC"}</span>
                <span className="ticker-name">{t.iso3}</span>
                <span className="ticker-val mono">{fmtDelta(t.delta)}</span>
              </span>
            ))
          )}
        </div>
      </div>
    </header>
  );
}
