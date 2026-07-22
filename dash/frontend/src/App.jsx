import { useEffect, useState } from "react";
import Header from "./components/Header";
import ControlBar from "./components/ControlBar";
import TrendChart from "./components/TrendChart";
import RadarPanel from "./components/RadarPanel";
import Leaderboard from "./components/Leaderboard";
import DriverPanel from "./components/DriverPanel";
import ForecastPanel from "./components/ForecastPanel";
import GlobalDriversPanel from "./components/GlobalDriversPanel";
import FocusCountrySelect from "./components/FocusCountrySelect";
import PeerPanel from "./components/PeerPanel";
import { useSoftPowerData } from "./lib/DataContext.jsx";
import { useTimeseries, useDeltas } from "./lib/hooks";
import { API_BASE_URL } from "./lib/api";
import "./App.css";

const TABS = [
  { key: "overview", label: "Overview" },
  { key: "movers", label: "Rise & Fall" },
  { key: "explain", label: "SHAP Drivers" },
  { key: "outlook", label: "Forecast & Peers" },
];

export default function App() {
  const { latest, loading: refLoading, error: refError } = useSoftPowerData();

  const [selected, setSelected] = useState([]);
  const [years, setYears] = useState([2010, 2024]);
  const [metric, setMetric] = useState("score");
  const [focusCountry, setFocusCountry] = useState(null);
  const [initialized, setInitialized] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const [yStart, yEnd] = years;

  // Seed the default selection once the live "latest" ranking arrives.
  useEffect(() => {
    if (!initialized && latest.length) {
      const defaults = latest.slice(0, 5).map((c) => c.iso3);
      setSelected(defaults);
      setFocusCountry(defaults[0]);
      setInitialized(true);
    }
  }, [initialized, latest]);

  const { data: timeseries, loading: tsLoading } = useTimeseries(selected, yStart, yEnd);
  const { data: deltas, loading: deltasLoading } = useDeltas(yStart, yEnd);

  function handleChangeYears(a, b) {
    setYears([a, b]);
  }

  function handleChangeSelected(next) {
    setSelected(next);
    if (next.length && !next.includes(focusCountry)) {
      setFocusCountry(next[0]);
    }
    if (!next.length) setFocusCountry(null);
  }

  if (refError) {
    return (
      <div className="api-error">
        <h2>Can't reach the API</h2>
        <p>
          Looked for it at <code className="mono">{API_BASE_URL}</code>. Make sure the backend is running
          and <code className="mono">VITE_API_BASE_URL</code> points at it.
        </p>
        <p className="mono api-error-detail">{String(refError.message || refError)}</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Header yStart={yStart} yEnd={yEnd} deltas={deltas} deltasLoading={deltasLoading} />

      <ControlBar
        selected={selected}
        onChangeSelected={handleChangeSelected}
        yStart={yStart}
        yEnd={yEnd}
        onChangeYears={handleChangeYears}
        metric={metric}
        onChangeMetric={setMetric}
      />

      <nav className="output-tabs" aria-label="Dashboard outputs">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`output-tab ${activeTab === tab.key ? "is-active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {activeTab === "overview" && (
          <div className="grid-2">
            <TrendChart selected={selected} timeseries={timeseries} loading={tsLoading} yStart={yStart} yEnd={yEnd} metric={metric} />
            <RadarPanel selected={selected} timeseries={timeseries} loading={tsLoading} endYear={yEnd} />
          </div>
        )}

        {activeTab === "movers" && (
          <Leaderboard
            yStart={yStart}
            yEnd={yEnd}
            deltas={deltas}
            loading={deltasLoading}
            focusCountry={focusCountry}
            onFocusCountry={(iso3) => {
              setFocusCountry(iso3);
              setActiveTab("explain");
            }}
          />
        )}

        {activeTab === "explain" && (
          <>
            <FocusCountrySelect value={focusCountry} onChange={setFocusCountry} label="Country for SHAP Explanation" />
            <div className="grid-2">
              <DriverPanel focusCountry={focusCountry} />
              <GlobalDriversPanel />
            </div>
          </>
        )}

        {activeTab === "outlook" && (
          <>
            <FocusCountrySelect value={focusCountry} onChange={setFocusCountry} label="Country for Forecast and Peers" />
            <div className="grid-2">
              <ForecastPanel focusCountry={focusCountry} />
              <PeerPanel focusCountry={focusCountry} />
            </div>
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Composite scores blend five capital dimensions &mdash; culture, innovation, political legitimacy,
          institutions, and human development &mdash; from World Bank, UNESCO, Freedom House, Heritage Foundation,
          and national statistics sources, modeled with a gradient-boosted ensemble and Kalman state-space forecast.
        </p>
        <p className="app-footer-note mono">
          Soft Power Intelligence &middot; Live API + PostgreSQL &middot; {API_BASE_URL}
        </p>
      </footer>
    </div>
  );
}
