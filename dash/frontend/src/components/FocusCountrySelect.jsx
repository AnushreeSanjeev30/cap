import { useSoftPowerData } from "../lib/DataContext.jsx";
import "./Panel.css";

export default function FocusCountrySelect({ value, onChange, label = "Focus Country" }) {
  const { countries, loading } = useSoftPowerData();

  return (
    <div className="focus-select panel">
      <label className="eyebrow" htmlFor="focus-country">{label}</label>
      <select
        id="focus-country"
        className="focus-country-select"
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading || countries.length === 0}
      >
        <option value="">{loading ? "Loading countries..." : "Select a country"}</option>
        {countries.map((country) => (
          <option key={country.iso3} value={country.iso3}>
            {country.name} ({country.iso3})
          </option>
        ))}
      </select>
    </div>
  );
}
