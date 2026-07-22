import csv
import math
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

DIMENSIONS = {
    "rnd_pct_gdp_sp": "innovation",
    "internet_pct_sp": "innovation",
    "ai_publications": "innovation",
    "rd_researchers_per_mil": "innovation",
    "sci_journal_articles": "innovation",
    "hightech_exports_pct": "innovation",
    "ict_patents": "innovation",
    "tourist_arrivals": "culture",
    "unesco_total_sites": "culture",
    "unesco_cultural_sites": "culture",
    "property_rights": "governance",
    "govt_integrity": "governance",
    "judicial_effectiveness": "governance",
    "business_freedom": "governance",
    "investment_freedom": "governance",
    "fh_status_num": "politics",
    "fh_combined_score": "politics",
    "fh_total_score": "politics",
    "life_expectancy": "human",
    "infant_mortality": "human",
    "physicians_per_1k": "human",
    "tertiary_enroll_pct": "human",
}


def clean_float(value):
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def clean_int(value):
    number = clean_float(value)
    return None if number is None else int(round(number))


def read_csv(name):
    with open(OUTPUT_DIR / name, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def nearest_by_year(points, target):
    return min(points, key=lambda p: abs(p["year"] - target))


def feature_label(name):
    cleaned = name.replace("_lag1", "").replace("_lag2", "").replace("_slope", "")
    return cleaned.replace("_", " ").title()


def volatility_tier(value):
    if value is None:
        return None
    if value >= 2.5:
        return "High"
    if value >= 1.25:
        return "Medium"
    return "Low"


class FileDataStore:
    def __init__(self):
        panel = read_csv("master_soft_power_panel.csv")
        refs = read_csv("country_reference_table.csv")
        kalman = read_csv("kalman_results.csv")
        summary = read_csv("kalman_summary.csv")
        forecast = read_csv("kalman_forecast_5yr.csv")
        shap_country = read_csv("shap_country.csv")
        shap_global = read_csv("shap_global.csv")

        self.country_names = self._build_country_names(panel, refs)
        self.timeseries = self._build_timeseries(panel, kalman)
        self.countries = sorted(
            [
                {"iso3": iso3, "name": self.country_names.get(iso3, iso3)}
                for iso3 in self.timeseries
            ],
            key=lambda row: row["name"],
        )
        self.latest = self._build_latest(summary, shap_country)
        self.drivers_by_iso3 = self._build_drivers(shap_country)
        self.forecast_by_iso3 = self._build_forecast(forecast)
        self.global_importance = self._build_global_importance(shap_global)
        self.peer_vectors = self._build_peer_vectors()

    def _build_country_names(self, panel, refs):
        names = {}
        for row in refs:
            if row.get("is_current") in ("1", 1, True, "True"):
                iso3 = (row.get("iso3") or "").upper()
                if len(iso3) == 3:
                    names[iso3] = row.get("canonical") or row.get("raw_name") or iso3
        for row in panel:
            iso3 = (row.get("iso3") or "").upper()
            if len(iso3) == 3:
                names.setdefault(iso3, row.get("canonical") or iso3)
        return names

    def _build_timeseries(self, panel, kalman):
        kalman_by_key = {
            ((row.get("iso3") or "").upper(), clean_int(row.get("year"))): row
            for row in kalman
        }
        rows_by_year = {}
        for row in panel:
            iso3 = (row.get("iso3") or "").upper()
            year = clean_int(row.get("year"))
            if len(iso3) != 3 or year is None:
                continue
            score = clean_float(row.get("soft_power_score"))
            krow = kalman_by_key.get((iso3, year))
            if krow:
                score = clean_float(krow.get("kalman_score")) or score
            rows_by_year.setdefault(year, []).append((iso3, score))

        ranks = {}
        for year, rows in rows_by_year.items():
            ordered = sorted([r for r in rows if r[1] is not None], key=lambda r: r[1], reverse=True)
            ranks.update({(iso3, year): i + 1 for i, (iso3, _) in enumerate(ordered)})

        by_iso3 = {}
        for row in panel:
            iso3 = (row.get("iso3") or "").upper()
            year = clean_int(row.get("year"))
            if len(iso3) != 3 or year is None:
                continue
            krow = kalman_by_key.get((iso3, year), {})
            score = clean_float(krow.get("kalman_score")) or clean_float(row.get("soft_power_score"))
            by_iso3.setdefault(iso3, []).append(
                {
                    "year": year,
                    "score": score,
                    "rank": ranks.get((iso3, year)) or clean_int(row.get("global_rank")),
                    "D1": clean_float(row.get("D1_Cultural_Capital")),
                    "D2": clean_float(row.get("D2_Innovation_Knowledge")),
                    "D3": clean_float(row.get("D3_Political_Legitimacy")),
                    "D4": clean_float(row.get("D4_Institutional_Quality")),
                    "D5": clean_float(row.get("D5_Human_Development")),
                    "gdp_per_capita_usd": clean_float(row.get("gdp_per_capita_usd")),
                    "tourist_arrivals": clean_float(row.get("tourist_arrivals")),
                    "internet_pct_sp": clean_float(row.get("internet_pct_sp")),
                    "unesco_total_sites": clean_float(row.get("unesco_total_sites")),
                    "rnd_pct_gdp_sp": clean_float(row.get("rnd_pct_gdp_sp")),
                    "life_expectancy": clean_float(row.get("life_expectancy")),
                    "hightech_exports_pct": clean_float(row.get("hightech_exports_pct")),
                }
            )
        for points in by_iso3.values():
            points.sort(key=lambda p: p["year"])
        return by_iso3

    def _build_latest(self, summary, shap_country):
        shap_by_iso3 = {(row.get("iso3") or "").upper(): row for row in shap_country}
        summary_rows = []
        for row in summary:
            iso3 = (row.get("iso3") or "").upper()
            points = self.timeseries.get(iso3, [])
            if not points:
                continue
            latest_point = max(points, key=lambda p: p["year"])
            score = clean_float(row.get("latest_score")) or latest_point.get("score")
            latest_ci = clean_float(row.get("latest_ci"))
            shape = shap_by_iso3.get(iso3, {})
            summary_rows.append(
                {
                    "iso3": iso3,
                    "name": self.country_names.get(iso3, iso3),
                    "year": latest_point["year"],
                    "score": score,
                    "ci_lower": score - latest_ci if score is not None and latest_ci is not None else None,
                    "ci_upper": score + latest_ci if score is not None and latest_ci is not None else None,
                    "rank": None,
                    "regime": None,
                    "volatility": clean_float(shape.get("score_volatility")) or clean_float(row.get("innovation_std")),
                    "influence_growth": clean_float(shape.get("influence_growth")),
                    "momentum_5y": clean_float(shape.get("momentum_5y")),
                    "volatility_tier": volatility_tier(clean_float(shape.get("score_volatility")) or clean_float(row.get("innovation_std"))),
                    "kalman_regime": row.get("stability_class") or None,
                    "trend_slope": clean_float(row.get("trend_slope")),
                    "stability_class": row.get("stability_class") or None,
                }
            )
        summary_rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or -1)))
        for rank, row in enumerate(summary_rows, 1):
            row["rank"] = rank
        return summary_rows

    def _build_drivers(self, shap_country):
        by_iso3 = {}
        ignored = {"iso3", "top_kpi", "top_kpi_dim"}
        for row in shap_country:
            iso3 = (row.get("iso3") or "").upper()
            features = []
            for key, value in row.items():
                if key in ignored:
                    continue
                shap = clean_float(value)
                if shap is None:
                    continue
                base_key = key.replace("_lag1", "").replace("_lag2", "").replace("_slope", "")
                features.append(
                    {
                        "feature": feature_label(key),
                        "raw": key,
                        "shap": shap,
                        "direction": "up" if shap >= 0 else "down",
                        "dimension": DIMENSIONS.get(base_key, "other"),
                    }
                )
            features.sort(key=lambda r: abs(r["shap"]), reverse=True)
            by_iso3[iso3] = features[:10]
        return by_iso3

    def _build_forecast(self, forecast):
        by_iso3 = {}
        for row in forecast:
            iso3 = (row.get("iso3") or "").upper()
            by_iso3.setdefault(iso3, []).append(
                {
                    "year": clean_int(row.get("forecast_year")),
                    "score": clean_float(row.get("forecast_score")),
                    "lo": clean_float(row.get("ci_lower_95")),
                    "hi": clean_float(row.get("ci_upper_95")),
                }
            )
        for points in by_iso3.values():
            points.sort(key=lambda p: p["year"] or 0)
        return by_iso3

    def _build_global_importance(self, shap_global):
        rows = []
        for row in shap_global:
            rows.append(
                {
                    "feature": feature_label(row.get("kpi") or ""),
                    "raw": row.get("kpi") or "",
                    "importance": clean_float(row.get("mean_abs_shap")) or 0,
                }
            )
        rows.sort(key=lambda r: r["importance"], reverse=True)
        return rows[:12]

    def _build_peer_vectors(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            return {}

        path = OUTPUT_DIR / "country_embeddings.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        if "iso3" in df.columns:
            df = df.set_index("iso3")
        vectors = {}
        for iso3, row in df.iterrows():
            values = [clean_float(v) or 0 for v in row.to_list()]
            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                vectors[str(iso3).upper()] = {"values": values, "norm": norm}
        return vectors

    def get_timeseries(self, iso3_list, y_start, y_end):
        return {
            iso3: [
                point for point in self.timeseries.get(iso3, [])
                if y_start <= point["year"] <= y_end
            ]
            for iso3 in iso3_list
        }

    def get_deltas(self, y_start, y_end):
        rows = []
        for iso3, points in self.timeseries.items():
            usable = [p for p in points if p.get("score") is not None]
            if not usable:
                continue
            start = nearest_by_year(usable, y_start)
            end = nearest_by_year(usable, y_end)
            if start.get("score") is None or end.get("score") is None:
                continue
            rows.append(
                {
                    "iso3": iso3,
                    "name": self.country_names.get(iso3, iso3),
                    "startYear": start["year"],
                    "startScore": start["score"],
                    "endYear": end["year"],
                    "endScore": end["score"],
                    "delta": round(end["score"] - start["score"], 2),
                }
            )
        return sorted(rows, key=lambda r: r["delta"], reverse=True)

    def get_peers(self, iso3, limit=8):
        iso3 = iso3.upper()
        target = self.peer_vectors.get(iso3)
        if not target:
            return []
        rows = []
        for other_iso3, other in self.peer_vectors.items():
            if other_iso3 == iso3:
                continue
            dot = sum(a * b for a, b in zip(target["values"], other["values"]))
            similarity = dot / (target["norm"] * other["norm"])
            latest = next((r for r in self.latest if r["iso3"] == other_iso3), {})
            rows.append(
                {
                    "iso3": other_iso3,
                    "name": self.country_names.get(other_iso3, other_iso3),
                    "similarity": round(similarity, 4),
                    "score": latest.get("score"),
                    "rank": latest.get("rank"),
                    "stability_class": latest.get("stability_class"),
                }
            )
        rows.sort(key=lambda r: r["similarity"], reverse=True)
        return rows[:limit]


@lru_cache(maxsize=1)
def get_file_store():
    return FileDataStore()
