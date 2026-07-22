import { useEffect, useMemo, useRef, useState } from "react";
import { useSoftPowerData } from "../lib/DataContext.jsx";
import { colorForIndex } from "../lib/colors";
import "./CountryPicker.css";

const MAX_SELECT = 10;

export default function CountryPicker({ selected, onChange }) {
  const { countries, loading } = useSoftPowerData();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef(null);

  useEffect(() => {
    function onClick(e) {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return countries;
    return countries.filter(
      (c) => c.name.toLowerCase().includes(q) || c.iso3.toLowerCase().includes(q)
    );
  }, [query, countries]);

  function toggle(iso3) {
    if (selected.includes(iso3)) {
      onChange(selected.filter((s) => s !== iso3));
    } else if (selected.length < MAX_SELECT) {
      onChange([...selected, iso3]);
    }
  }

  return (
    <div className="cpicker" ref={rootRef}>
      <label className="eyebrow cpicker-label">Compare Countries</label>
      <button
        type="button"
        className="cpicker-trigger"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <div className="cpicker-chips">
          {selected.length === 0 && (
            <span className="cpicker-placeholder">
              {loading && countries.length === 0 ? "Loading countries\u2026" : `Select up to ${MAX_SELECT} states\u2026`}
            </span>
          )}
          {selected.map((iso3, i) => {
            const c = countries.find((x) => x.iso3 === iso3);
            return (
              <span className="chip" key={iso3} style={{ "--chip-color": colorForIndex(i) }}>
                <span className="chip-dot" />
                {c ? c.name : iso3}
                <span
                  className="chip-x"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggle(iso3);
                  }}
                >
                  &times;
                </span>
              </span>
            );
          })}
        </div>
        <span className="cpicker-caret">{open ? "\u25B4" : "\u25BE"}</span>
      </button>

      {open && (
        <div className="cpicker-panel">
          <input
            autoFocus
            className="cpicker-search"
            placeholder="Search country or ISO3&hellip;"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="cpicker-list">
            {filtered.map((c) => {
              const isSel = selected.includes(c.iso3);
              const disabled = !isSel && selected.length >= MAX_SELECT;
              return (
                <div
                  key={c.iso3}
                  className={`cpicker-opt ${isSel ? "is-selected" : ""} ${disabled ? "is-disabled" : ""}`}
                  onClick={() => !disabled && toggle(c.iso3)}
                >
                  <span className={`cpicker-check ${isSel ? "checked" : ""}`}>{isSel ? "\u2713" : ""}</span>
                  <span className="cpicker-opt-name">{c.name}</span>
                  <span className="cpicker-opt-iso mono">{c.iso3}</span>
                </div>
              );
            })}
            {filtered.length === 0 && <div className="cpicker-empty">No matches</div>}
          </div>
        </div>
      )}
    </div>
  );
}
