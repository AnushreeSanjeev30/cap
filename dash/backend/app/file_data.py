import csv
import math
import pickle
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

PEER_DIMENSIONS = {
    "Culture": {"tourist_arrivals", "unesco_total_sites", "unesco_cultural_sites"},
    "Innovation": {
        "rnd_pct_gdp_sp", "internet_pct_sp", "ai_publications",
        "rd_researchers_per_mil", "sci_journal_articles", "hightech_exports_pct",
        "ict_patents",
    },
    "Politics": {"fh_status_num", "fh_combined_score", "fh_total_score", "trade_pct_gdp", "investment_freedom"},
    "Governance": {"property_rights", "govt_integrity", "judicial_effectiveness", "business_freedom"},
    "Human development": {"life_expectancy", "infant_mortality", "physicians_per_1k", "tertiary_enroll_pct"},
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


def strip_feature_suffix(name):
    return name.replace("_lag1", "").replace("_lag2", "").replace("_slope", "")


def feature_label(name):
    return strip_feature_suffix(name).replace("_", " ").title()


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
        self.peer_features = self._build_peer_features()

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
        # Group lag/slope variants of the same base indicator (e.g.
        # infant_mortality and infant_mortality_lag1) under one bar before
        # ranking. Without this, two variants of the same concept can both
        # land in the top 12 with the identical display label (feature_label()
        # strips the _lag1/_lag2/_slope suffix), which breaks the dashboard's
        # vertical bar chart: Recharts' categorical Y-axis can't distinguish
        # two rows with the same category text, so bars below the collision
        # render misaligned with their own tooltip data.
        groups = {}
        for row in shap_global:
            raw = row.get("kpi") or ""
            base = strip_feature_suffix(raw)
            importance = clean_float(row.get("mean_abs_shap")) or 0
            g = groups.setdefault(base, {"raw": raw, "importance": 0.0})
            g["importance"] += importance
            # Prefer the un-lagged variant's raw name for the tooltip lookup
            # if we later see it (groups may first encounter a _lag1 row).
            if raw == base:
                g["raw"] = raw

        rows = [
            {"feature": feature_label(g["raw"]), "raw": g["raw"], "importance": g["importance"]}
            for g in groups.values()
        ]
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

        # Prefer the model pipeline's normalized embedding matrix so cosine
        # similarities match FAISS-space peers used during training.
        matrix_path = OUTPUT_DIR / "artifacts" / "embedding_matrix.npy"
        if matrix_path.exists():
            try:
                import numpy as np

                matrix = np.load(matrix_path)
                if matrix.shape[0] == len(df.index):
                    vectors = {}
                    for i, iso3 in enumerate(df.index):
                        values = [float(v) for v in matrix[i].tolist()]
                        norm = math.sqrt(sum(v * v for v in values))
                        if norm > 0:
                            vectors[str(iso3).upper()] = {"values": values, "norm": norm}
                    if vectors:
                        return vectors
            except Exception:
                # Fall through to raw parquet vectors if artifacts are not usable.
                pass

        vectors = {}
        for iso3, row in df.iterrows():
            values = [clean_float(v) or 0 for v in row.to_list()]
            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                vectors[str(iso3).upper()] = {"values": values, "norm": norm}
        return vectors

    def _build_peer_features(self):
        """Load the feature values used by the FAISS peer vectors."""
        try:
            import numpy as np
            import pandas as pd

            path = OUTPUT_DIR / "country_embeddings.parquet"
            scaler_path = OUTPUT_DIR / "artifacts" / "embedding_scaler.pkl"
            if not path.exists() or not scaler_path.exists():
                return {}
            df = pd.read_parquet(path)
            if "iso3" in df.columns:
                df = df.set_index("iso3")
            with open(scaler_path, "rb") as handle:
                scaler = pickle.load(handle)
            values = scaler.transform(df.values).astype("float32")
            return {
                "columns": list(df.columns),
                "raw": df,
                "standardized": {
                    str(iso3).upper(): values[i]
                    for i, iso3 in enumerate(df.index)
                },
            }
        except Exception:
            return {}

    def _explain_peer_pair(self, iso3, other_iso3):
        import numpy as np

        features = self.peer_features
        if not features or iso3 not in features["standardized"] or other_iso3 not in features["standardized"]:
            return {"dimensions": [], "indicators": []}

        columns = features["columns"]
        left = features["standardized"][iso3]
        right = features["standardized"][other_iso3]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0:
            return {"dimensions": [], "indicators": []}

        contributions = {}
        dimension_scores = {name: 0.0 for name in PEER_DIMENSIONS}
        for index, column in enumerate(columns):
            base = column.removesuffix("_mean").removesuffix("_std")
            contribution = float(left[index] * right[index] / denominator)
            contributions[column] = contribution
            for dimension, indicators in PEER_DIMENSIONS.items():
                if base in indicators:
                    dimension_scores[dimension] += contribution

        dimensions = [
            {"name": name, "contribution": round(value, 4)}
            for name, value in sorted(dimension_scores.items(), key=lambda item: item[1], reverse=True)
            if value > 0
        ][:3]

        raw = features["raw"]
        indicators = []
        for column, contribution in sorted(contributions.items(), key=lambda item: item[1], reverse=True):
            if not column.endswith("_mean") or contribution <= 0:
                continue
            indicators.append({
                "label": column.removesuffix("_mean").replace("_", " ").title(),
                "contribution": round(contribution, 4),
                "countryValue": clean_float(raw.loc[iso3, column]),
                "peerValue": clean_float(raw.loc[other_iso3, column]),
            })
            if len(indicators) == 3:
                break
        return {"dimensions": dimensions, "indicators": indicators}

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
            similarity = max(-1.0, min(1.0, similarity))
            latest = next((r for r in self.latest if r["iso3"] == other_iso3), {})
            rows.append(
                {
                    "iso3": other_iso3,
                    "name": self.country_names.get(other_iso3, other_iso3),
                    "similarity": round(similarity, 4),
                    "score": latest.get("score"),
                    "rank": latest.get("rank"),
                    "stability_class": latest.get("stability_class"),
                    "explanation": self._explain_peer_pair(iso3, other_iso3),
                }
            )
        rows.sort(key=lambda r: r["similarity"], reverse=True)
        return rows[:limit]


@lru_cache(maxsize=1)
def get_file_store():
    return FileDataStore()
