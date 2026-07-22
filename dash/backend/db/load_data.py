"""
Load seed JSON data (backend/seed_data/*.json) into PostgreSQL.

Usage:
    DATABASE_URL=postgresql://user:pass@host:5432/dbname python db/load_data.py

This is the "static export -> database" bridge: it reads the same compact JSON
files that used to be bundled directly into the frontend, and inserts them into
the tables defined in db/schema.sql. Re-run it any time you have a fresh export
from the modeling pipeline (it truncates and reloads, so it's safe to re-run).
"""
import json
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

SEED_DIR = Path(__file__).resolve().parent.parent / "seed_data"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://softpower:softpower_dev@localhost:5432/softpower"
)


def load_json(name):
    with open(SEED_DIR / name) as f:
        return json.load(f)


def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    print("Truncating existing tables...")
    cur.execute(
        "TRUNCATE forecast, drivers, latest_snapshot, timeseries, global_importance, countries RESTART IDENTITY CASCADE;"
    )

    # ---------- countries ----------
    countries = load_json("countries.json")
    execute_values(
        cur,
        "INSERT INTO countries (iso3, name) VALUES %s",
        [(c["iso3"], c["name"]) for c in countries],
    )
    print(f"countries: {len(countries)}")

    # ---------- timeseries ----------
    ts = load_json("timeseries.json")
    rows = []
    for iso3, points in ts.items():
        for p in points:
            rows.append(
                (
                    iso3,
                    p["year"],
                    p.get("score"),
                    p.get("rank"),
                    p.get("D1"),
                    p.get("D2"),
                    p.get("D3"),
                    p.get("D4"),
                    p.get("D5"),
                    p.get("gdp_per_capita_usd"),
                    p.get("tourist_arrivals"),
                    p.get("internet_pct_sp"),
                    p.get("unesco_total_sites"),
                    p.get("rnd_pct_gdp_sp"),
                    p.get("life_expectancy"),
                    p.get("hightech_exports_pct"),
                )
            )
    execute_values(
        cur,
        """INSERT INTO timeseries
           (iso3, year, score, rank, d1_culture, d2_innovation, d3_politics, d4_institutions, d5_human_dev,
            gdp_per_capita_usd, tourist_arrivals, internet_pct_sp, unesco_total_sites, rnd_pct_gdp_sp,
            life_expectancy, hightech_exports_pct)
           VALUES %s""",
        rows,
    )
    print(f"timeseries: {len(rows)}")

    # ---------- latest_snapshot ----------
    latest = load_json("latest.json")
    execute_values(
        cur,
        """INSERT INTO latest_snapshot
           (iso3, year, score, ci_lower, ci_upper, rank, regime, volatility, influence_growth,
            momentum_5y, volatility_tier, kalman_regime, trend_slope, stability_class)
           VALUES %s""",
        [
            (
                l["iso3"], l["year"], l["score"], l["ci_lower"], l["ci_upper"], l["rank"],
                l.get("regime"), l.get("volatility"), l.get("influence_growth"), l.get("momentum_5y"),
                l.get("volatility_tier"), l.get("kalman_regime"), l.get("trend_slope"), l.get("stability_class"),
            )
            for l in latest
        ],
    )
    print(f"latest_snapshot: {len(latest)}")

    # ---------- drivers ----------
    drivers = load_json("drivers.json")
    rows = []
    for iso3, feats in drivers.items():
        for i, f in enumerate(feats):
            rows.append((iso3, i, f["feature"], f.get("raw"), f["shap"], f["direction"], f["dimension"]))
    execute_values(
        cur,
        "INSERT INTO drivers (iso3, rank_order, feature, raw_feature, shap, direction, dimension) VALUES %s",
        rows,
    )
    print(f"drivers: {len(rows)}")

    # ---------- forecast ----------
    forecast = load_json("forecast.json")
    rows = []
    for iso3, points in forecast.items():
        for p in points:
            rows.append((iso3, p["year"], p["score"], p["lo"], p["hi"]))
    execute_values(
        cur, "INSERT INTO forecast (iso3, year, score, ci_lower_95, ci_upper_95) VALUES %s", rows
    )
    print(f"forecast: {len(rows)}")

    # ---------- global_importance ----------
    gi = load_json("global_importance.json")
    execute_values(
        cur,
        "INSERT INTO global_importance (feature, importance, rank_order) VALUES %s",
        [(g["feature"], g["importance"], i) for i, g in enumerate(gi)],
    )
    print(f"global_importance: {len(gi)}")

    conn.commit()
    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Load failed: {e}", file=sys.stderr)
        sys.exit(1)
