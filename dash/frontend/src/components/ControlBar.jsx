import CountryPicker from "./CountryPicker";
import "./ControlBar.css";

const YEAR_MIN = 2000;
const YEAR_MAX = 2024;

const METRICS = [
  { key: "score", label: "Composite Score" },
  { key: "D1", label: "Cultural Capital" },
  { key: "D2", label: "Innovation & Knowledge" },
  { key: "D3", label: "Political Legitimacy" },
  { key: "D4", label: "Institutional Quality" },
  { key: "D5", label: "Human Development" },
];

export default function ControlBar({ selected, onChangeSelected, yStart, yEnd, onChangeYears, metric, onChangeMetric }) {
  function handleStart(e) {
    const v = Math.min(Number(e.target.value), yEnd - 1);
    onChangeYears(v, yEnd);
  }
  function handleEnd(e) {
    const v = Math.max(Number(e.target.value), yStart + 1);
    onChangeYears(yStart, v);
  }

  return (
    <div className="control-bar panel">
      <CountryPicker selected={selected} onChange={onChangeSelected} />

      <div className="control-block">
        <label className="eyebrow">Comparison Window</label>
        <div className="range-readout mono">{yStart} &rarr; {yEnd}</div>
        <div className="range-wrap">
          <input
            type="range"
            min={YEAR_MIN}
            max={YEAR_MAX}
            value={yStart}
            onChange={handleStart}
            className="range-input range-start"
          />
          <input
            type="range"
            min={YEAR_MIN}
            max={YEAR_MAX}
            value={yEnd}
            onChange={handleEnd}
            className="range-input range-end"
          />
        </div>
      </div>

      <div className="control-block">
        <label className="eyebrow" htmlFor="metric-select">Trend Metric</label>
        <select
          id="metric-select"
          className="metric-select"
          value={metric}
          onChange={(e) => onChangeMetric(e.target.value)}
        >
          {METRICS.map((m) => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
      </div>
    </div>
  );
}

export { METRICS };
