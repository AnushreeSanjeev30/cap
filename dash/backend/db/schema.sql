-- Soft Power Intelligence — schema
-- Run once against a fresh database: psql $DATABASE_URL -f db/schema.sql

CREATE TABLE IF NOT EXISTS countries (
    iso3        CHAR(3) PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timeseries (
    iso3                  CHAR(3) NOT NULL REFERENCES countries(iso3) ON DELETE CASCADE,
    year                  SMALLINT NOT NULL,
    score                 REAL,
    rank                  SMALLINT,
    d1_culture            REAL,
    d2_innovation         REAL,
    d3_politics           REAL,
    d4_institutions       REAL,
    d5_human_dev          REAL,
    gdp_per_capita_usd    REAL,
    tourist_arrivals      REAL,
    internet_pct_sp       REAL,
    unesco_total_sites    REAL,
    rnd_pct_gdp_sp        REAL,
    life_expectancy       REAL,
    hightech_exports_pct  REAL,
    PRIMARY KEY (iso3, year)
);
CREATE INDEX IF NOT EXISTS idx_timeseries_iso3 ON timeseries (iso3);
CREATE INDEX IF NOT EXISTS idx_timeseries_year ON timeseries (year);

CREATE TABLE IF NOT EXISTS latest_snapshot (
    iso3               CHAR(3) PRIMARY KEY REFERENCES countries(iso3) ON DELETE CASCADE,
    year               SMALLINT,
    score              REAL,
    ci_lower           REAL,
    ci_upper           REAL,
    rank               SMALLINT,
    regime             TEXT,
    volatility         REAL,
    influence_growth   REAL,
    momentum_5y        REAL,
    volatility_tier    TEXT,
    kalman_regime      TEXT,
    trend_slope        REAL,
    stability_class    TEXT
);

CREATE TABLE IF NOT EXISTS drivers (
    id           SERIAL PRIMARY KEY,
    iso3         CHAR(3) NOT NULL REFERENCES countries(iso3) ON DELETE CASCADE,
    rank_order   SMALLINT NOT NULL,
    feature      TEXT NOT NULL,
    raw_feature  TEXT,
    shap         REAL,
    direction    TEXT,
    dimension    TEXT
);
CREATE INDEX IF NOT EXISTS idx_drivers_iso3 ON drivers (iso3);

CREATE TABLE IF NOT EXISTS forecast (
    id          SERIAL PRIMARY KEY,
    iso3        CHAR(3) NOT NULL REFERENCES countries(iso3) ON DELETE CASCADE,
    year        SMALLINT NOT NULL,
    score       REAL,
    ci_lower_95 REAL,
    ci_upper_95 REAL,
    UNIQUE (iso3, year)
);
CREATE INDEX IF NOT EXISTS idx_forecast_iso3 ON forecast (iso3);

CREATE TABLE IF NOT EXISTS global_importance (
    id          SERIAL PRIMARY KEY,
    feature     TEXT NOT NULL,
    importance  REAL NOT NULL,
    rank_order  SMALLINT NOT NULL
);
