import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .file_data import get_file_store

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ModuleNotFoundError:
    create_engine = None
    text = None
    SQLAlchemyError = Exception

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://softpower:softpower_dev@localhost:5432/softpower"
)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "auto").lower()

app = FastAPI(
    title="Soft Power Intelligence API",
    description="Serves comparative soft power scores, trends, drivers, and forecasts from PostgreSQL or real model artifacts.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_methods=["GET"],
    allow_headers=["*"],
)

engine = (
    create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)
    if create_engine
    else None
)


def rows_to_dicts(result):
    return [dict(row._mapping) for row in result]


def use_files():
    return DATA_SOURCE == "files"


def with_fallback(db_fn, file_fn):
    if use_files():
        return file_fn()
    if engine is None:
        if DATA_SOURCE == "db":
            raise RuntimeError("SQLAlchemy is not installed; install backend requirements or set DATA_SOURCE=files.")
        return file_fn()
    try:
        return db_fn()
    except SQLAlchemyError:
        if DATA_SOURCE == "db":
            raise
        return file_fn()


@app.get("/health")
def health():
    if use_files():
        get_file_store()
        return {"status": "ok", "dataSource": "files"}
    if engine is None:
        if DATA_SOURCE == "db":
            raise RuntimeError("SQLAlchemy is not installed; install backend requirements or set DATA_SOURCE=files.")
        get_file_store()
        return {"status": "ok", "dataSource": "files", "postgres": "sqlalchemy-not-installed"}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "dataSource": "postgres"}
    except SQLAlchemyError:
        if DATA_SOURCE == "db":
            raise
        get_file_store()
        return {"status": "ok", "dataSource": "files", "postgres": "unavailable"}


@app.get("/api/countries")
def get_countries():
    def db():
        with engine.connect() as conn:
            result = conn.execute(text("SELECT iso3, name FROM countries ORDER BY name"))
            return rows_to_dicts(result)

    return with_fallback(db, lambda: get_file_store().countries)


@app.get("/api/latest")
def get_latest():
    def db():
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """SELECT l.iso3, c.name, l.year, l.score, l.ci_lower, l.ci_upper, l.rank,
                              l.regime, l.volatility, l.influence_growth, l.momentum_5y,
                              l.volatility_tier, l.kalman_regime, l.trend_slope, l.stability_class
                       FROM latest_snapshot l
                       JOIN countries c ON c.iso3 = l.iso3
                       ORDER BY l.rank"""
                )
            )
            return rows_to_dicts(result)

    return with_fallback(db, lambda: get_file_store().latest)


@app.get("/api/timeseries")
def get_timeseries(
    iso3: list[str] = Query(default=[], description="Repeat to request multiple countries, e.g. ?iso3=USA&iso3=CHN"),
    y_start: int = Query(default=2000, alias="yStart"),
    y_end: int = Query(default=2024, alias="yEnd"),
):
    if not iso3:
        raise HTTPException(400, "Provide at least one iso3 query parameter")
    iso3_upper = [c.upper() for c in iso3]

    def db():
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """SELECT iso3, year, score, rank,
                              d1_culture AS "D1", d2_innovation AS "D2", d3_politics AS "D3",
                              d4_institutions AS "D4", d5_human_dev AS "D5",
                              gdp_per_capita_usd, tourist_arrivals, internet_pct_sp,
                              unesco_total_sites, rnd_pct_gdp_sp, life_expectancy, hightech_exports_pct
                       FROM timeseries
                       WHERE iso3 = ANY(:iso3) AND year BETWEEN :y_start AND :y_end
                       ORDER BY iso3, year"""
                ),
                {"iso3": iso3_upper, "y_start": y_start, "y_end": y_end},
            )
            rows = rows_to_dicts(result)

        by_country: dict[str, list] = {c: [] for c in iso3_upper}
        for r in rows:
            country = r.pop("iso3")
            by_country.setdefault(country, []).append(r)
        return by_country

    return with_fallback(db, lambda: get_file_store().get_timeseries(iso3_upper, y_start, y_end))


@app.get("/api/deltas")
def get_deltas(y_start: int = Query(alias="yStart"), y_end: int = Query(alias="yEnd")):
    """Nearest-year score change for every country between two years."""

    def db():
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    WITH bounds AS (
                        SELECT iso3,
                               (SELECT year FROM timeseries t2 WHERE t2.iso3 = t.iso3
                                 ORDER BY ABS(year - :y_start) LIMIT 1) AS start_year,
                               (SELECT year FROM timeseries t3 WHERE t3.iso3 = t.iso3
                                 ORDER BY ABS(year - :y_end) LIMIT 1) AS end_year
                        FROM timeseries t
                        GROUP BY iso3
                    )
                    SELECT c.iso3, c.name,
                           ts_start.year AS "startYear", ts_start.score AS "startScore",
                           ts_end.year AS "endYear", ts_end.score AS "endScore",
                           ROUND((ts_end.score - ts_start.score)::numeric, 2) AS delta
                    FROM bounds b
                    JOIN countries c ON c.iso3 = b.iso3
                    JOIN timeseries ts_start ON ts_start.iso3 = b.iso3 AND ts_start.year = b.start_year
                    JOIN timeseries ts_end ON ts_end.iso3 = b.iso3 AND ts_end.year = b.end_year
                    WHERE ts_start.score IS NOT NULL AND ts_end.score IS NOT NULL
                    ORDER BY delta DESC
                    """
                ),
                {"y_start": y_start, "y_end": y_end},
            )
            return rows_to_dicts(result)

    return with_fallback(db, lambda: get_file_store().get_deltas(y_start, y_end))


@app.get("/api/drivers/{iso3}")
def get_drivers(iso3: str):
    iso3 = iso3.upper()

    def db():
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """SELECT feature, raw_feature AS raw, shap, direction, dimension
                       FROM drivers WHERE iso3 = :iso3 ORDER BY rank_order"""
                ),
                {"iso3": iso3},
            )
            return rows_to_dicts(result)

    rows = with_fallback(db, lambda: get_file_store().drivers_by_iso3.get(iso3, []))
    if not rows:
        raise HTTPException(404, f"No driver data for {iso3}")
    return rows


@app.get("/api/forecast/{iso3}")
def get_forecast(iso3: str):
    iso3 = iso3.upper()

    def db():
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """SELECT year, score, ci_lower_95 AS lo, ci_upper_95 AS hi
                       FROM forecast WHERE iso3 = :iso3 ORDER BY year"""
                ),
                {"iso3": iso3},
            )
            return rows_to_dicts(result)

    rows = with_fallback(db, lambda: get_file_store().forecast_by_iso3.get(iso3, []))
    if not rows:
        raise HTTPException(404, f"No forecast data for {iso3}")
    return rows


@app.get("/api/peers/{iso3}")
def get_peers(iso3: str, limit: int = Query(default=8, ge=1, le=25)):
    iso3 = iso3.upper()
    if not isinstance(limit, int):
        limit = 8
    rows = get_file_store().get_peers(iso3, limit)
    if not rows:
        raise HTTPException(404, f"No peer data for {iso3}")
    return rows


@app.get("/api/global-importance")
def get_global_importance():
    def db():
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT feature, importance FROM global_importance ORDER BY rank_order")
            )
            return rows_to_dicts(result)

    return with_fallback(db, lambda: get_file_store().global_importance)
