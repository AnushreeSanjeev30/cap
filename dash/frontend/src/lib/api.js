export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status} on ${path}: ${body || res.statusText}`);
  }
  return res.json();
}

export const api = {
  countries: () => getJSON("/api/countries"),
  latest: () => getJSON("/api/latest"),
  globalImportance: () => getJSON("/api/global-importance"),
  drivers: (iso3) => getJSON(`/api/drivers/${iso3}`),
  forecast: (iso3) => getJSON(`/api/forecast/${iso3}`),
  peers: (iso3) => getJSON(`/api/peers/${iso3}`),
  deltas: (yStart, yEnd) => getJSON(`/api/deltas?yStart=${yStart}&yEnd=${yEnd}`),
  timeseries: (iso3List, yStart, yEnd) => {
    const params = new URLSearchParams();
    iso3List.forEach((c) => params.append("iso3", c));
    params.set("yStart", yStart);
    params.set("yEnd", yEnd);
    return getJSON(`/api/timeseries?${params.toString()}`);
  },
};
