import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api } from "./api";

const DataContext = createContext(null);

export function DataProvider({ children }) {
  const [countries, setCountries] = useState([]);
  const [latest, setLatest] = useState([]);
  const [globalImportance, setGlobalImportance] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([api.countries(), api.latest(), api.globalImportance()])
      .then(([c, l, g]) => {
        if (cancelled) return;
        setCountries(c);
        setLatest(l);
        setGlobalImportance(g);
        setError(null);
      })
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const nameMap = useMemo(() => Object.fromEntries(countries.map((c) => [c.iso3, c.name])), [countries]);
  const latestByIso3 = useMemo(() => Object.fromEntries(latest.map((l) => [l.iso3, l])), [latest]);

  const value = { countries, latest, latestByIso3, globalImportance, nameMap, loading, error };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useSoftPowerData() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useSoftPowerData must be used within a DataProvider");
  return ctx;
}
