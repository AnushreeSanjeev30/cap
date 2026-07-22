import { useEffect, useRef, useState } from "react";
import { api } from "./api";

function useAsync(fn, deps, initial = null) {
  const [state, setState] = useState({ data: initial, loading: true, error: null });
  const seq = useRef(0);

  useEffect(() => {
    const mySeq = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn()
      .then((data) => {
        if (seq.current === mySeq) setState({ data, loading: false, error: null });
      })
      .catch((error) => {
        if (seq.current === mySeq) setState({ data: initial, loading: false, error });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}

export function useTimeseries(iso3List, yStart, yEnd) {
  const key = [...iso3List].sort().join(",");
  return useAsync(
    () => (iso3List.length ? api.timeseries(iso3List, yStart, yEnd) : Promise.resolve({})),
    [key, yStart, yEnd],
    {}
  );
}

export function useDeltas(yStart, yEnd) {
  return useAsync(() => api.deltas(yStart, yEnd), [yStart, yEnd], []);
}

export function useDrivers(iso3) {
  return useAsync(() => (iso3 ? api.drivers(iso3) : Promise.resolve([])), [iso3], []);
}

export function useForecast(iso3) {
  return useAsync(() => (iso3 ? api.forecast(iso3) : Promise.resolve([])), [iso3], []);
}

export function usePeers(iso3) {
  return useAsync(() => (iso3 ? api.peers(iso3) : Promise.resolve([])), [iso3], []);
}

// nearest available data point at or before target year, else nearest overall
export function pointNear(points, year) {
  if (!points || !points.length) return null;
  let best = points[0];
  let bestDist = Infinity;
  for (const p of points) {
    const d = Math.abs(p.year - year);
    if (d < bestDist) {
      bestDist = d;
      best = p;
    }
  }
  return best;
}
