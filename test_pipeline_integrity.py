"""
test_pipeline_integrity.py
============================
Smoke tests for the soft-power pipeline's plumbing: does everything load,
does everything have the shape it's supposed to, does the SoftPowerAgent
interface answer basic queries without crashing.

This is deliberately NOT model_trust_suite.py's job repeated. That suite
answers "is the model's behavior trustworthy" (overfitting, calibration,
leakage). This file answers "does the pipeline run correctly at all" --
schema drift, duplicate rows, a script that crashes on load, a file that's
silently missing a column something downstream depends on. Every test here
is a regression test for a real bug found and fixed in this repo:
  - duplicate (iso3, year) rows in the panel (NLD/VNM merge bug)
  - SoftPowerAgent crashing on __init__ (stale artifact paths)
  - a display-label collision in the dashboard's driver chart

Run the fast tests (default, a few seconds):
    .venv/Scripts/python.exe -m pytest test_pipeline_integrity.py -v

Run everything including the slow ones (re-runs pipeline scripts, ~1-2 min):
    .venv/Scripts/python.exe -m pytest test_pipeline_integrity.py -v -m slow --run-slow
"""

import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"


# ─────────────────────────────────────────────────────────────────────────
# Fixtures -- load each artifact once per test session, not once per test.
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def master_panel():
    return pd.read_csv(OUT / "master_soft_power_panel.csv")


@pytest.fixture(scope="session")
def master_phase_b():
    return pd.read_parquet(OUT / "master_phase_b.parquet")


@pytest.fixture(scope="session")
def kalman_results():
    return pd.read_csv(OUT / "kalman_results.csv")


@pytest.fixture(scope="session")
def kalman_forecast():
    return pd.read_csv(OUT / "kalman_forecast_5yr.csv")


@pytest.fixture(scope="session")
def predictions():
    return pd.read_csv(OUT / "soft_power_predictions.csv")


@pytest.fixture(scope="session")
def shap_global():
    return pd.read_csv(OUT / "shap_global.csv")


@pytest.fixture(scope="session")
def agent():
    sys.path.insert(0, str(ROOT))
    from archive.phase4_continuous_learning import SoftPowerAgent
    return SoftPowerAgent()


# ─────────────────────────────────────────────────────────────────────────
# Panel-level schema and consistency checks
# ─────────────────────────────────────────────────────────────────────────

def test_master_panel_has_required_columns(master_panel):
    required = {"iso3", "year", "soft_power_score", "global_rank",
                "outlier_feature_coverage", "is_outlier"}
    missing = required - set(master_panel.columns)
    assert not missing, f"master_soft_power_panel.csv is missing columns: {missing}"


def test_master_panel_no_duplicate_country_year(master_panel):
    """Regression test for the NLD/VNM merge bug (archive/fix_duplicate_rows.py)."""
    dupes = master_panel.groupby(["iso3", "year"]).size()
    offenders = dupes[dupes > 1]
    assert offenders.empty, (
        f"Duplicate (iso3, year) rows found: {offenders.to_dict()}. "
        f"Run archive/fix_duplicate_rows.py."
    )


def test_master_phase_b_no_duplicate_country_year(master_phase_b):
    col = "iso3" if "iso3" in master_phase_b.columns else "country_iso3"
    dupes = master_phase_b.groupby([col, "year"]).size()
    offenders = dupes[dupes > 1]
    assert offenders.empty, (
        f"Duplicate (iso3, year) rows found in master_phase_b.parquet: "
        f"{offenders.to_dict()}. Run archive/fix_duplicate_rows.py."
    )


def test_master_panel_year_range_sane(master_panel):
    assert master_panel["year"].min() >= 2000
    assert master_panel["year"].max() <= 2030, "year column looks corrupted (unreasonably far future)"


# ─────────────────────────────────────────────────────────────────────────
# Kalman output checks
# ─────────────────────────────────────────────────────────────────────────

def test_kalman_results_schema(kalman_results):
    required = {"iso3", "year", "raw_score", "kalman_score", "kalman_std",
                "ci_lower_95", "ci_upper_95", "ci_width", "innovation"}
    missing = required - set(kalman_results.columns)
    assert not missing, f"kalman_results.csv is missing columns: {missing}"


def test_kalman_results_ci_ordering(kalman_results):
    """A confidence interval where upper < lower means the filter (or the
    file) is broken, not just imprecise."""
    sub = kalman_results.dropna(subset=["ci_upper_95", "ci_lower_95"])
    bad = sub[sub["ci_upper_95"] < sub["ci_lower_95"]]
    assert bad.empty, f"{len(bad)} rows have ci_upper_95 < ci_lower_95"


def test_kalman_forecast_covers_5_horizons_per_country(kalman_forecast):
    horizon_sets = kalman_forecast.groupby("iso3")["horizon"].apply(
        lambda s: tuple(sorted(s.tolist())))
    bad = horizon_sets[horizon_sets != (1, 2, 3, 4, 5)]
    assert bad.empty, f"Countries missing a full 1-5 year horizon: {bad.index.tolist()}"


def test_kalman_forecast_no_missing_values(kalman_forecast):
    required = ["forecast_score", "ci_lower_95", "ci_upper_95", "forecast_std"]
    nan_counts = kalman_forecast[required].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs found in forecast output: {nan_counts[nan_counts > 0].to_dict()}"


def test_kalman_forecast_uncertainty_grows_with_horizon(kalman_forecast):
    """forecast_std should be non-decreasing across horizon 1->5 for every
    country -- this is the core state-space math (P_h = P_last + h*Q). A
    flat or shrinking std would mean the forecast degenerated into
    something else (e.g. a plain persistence copy with no real model
    behind it)."""
    violations = []
    for iso3, grp in kalman_forecast.groupby("iso3"):
        stds = grp.sort_values("horizon")["forecast_std"].tolist()
        if any(b < a - 1e-9 for a, b in zip(stds, stds[1:])):
            violations.append(iso3)
    assert not violations, f"forecast_std decreases with horizon for: {violations}"


# ─────────────────────────────────────────────────────────────────────────
# Predictions / ranking checks
# ─────────────────────────────────────────────────────────────────────────

def test_predictions_schema(predictions):
    required = {"iso3", "score", "ci_lower", "ci_upper", "global_rank", "regime"}
    missing = required - set(predictions.columns)
    assert not missing, f"soft_power_predictions.csv is missing columns: {missing}"


def test_global_rank_is_a_clean_permutation(predictions):
    ranks = predictions["global_rank"].dropna()
    assert ranks.duplicated().sum() == 0, "global_rank has duplicate values -- two countries tied for the same rank"
    assert ranks.min() == 1, "global_rank does not start at 1"
    assert ranks.max() == len(ranks), "global_rank has gaps (max rank != number of countries)"


def test_scores_in_plausible_range(predictions):
    assert predictions["score"].between(0, 120).all(), \
        "soft_power score outside a plausible 0-120 range -- check for a units/scaling bug"


# ─────────────────────────────────────────────────────────────────────────
# SHAP output checks
# ─────────────────────────────────────────────────────────────────────────

def test_shap_global_schema(shap_global):
    required = {"kpi", "mean_abs_shap", "mean_shap", "dimension", "rank"}
    missing = required - set(shap_global.columns)
    assert not missing, f"shap_global.csv is missing columns: {missing}"


def test_shap_global_csv_has_multiple_variants_per_indicator(shap_global):
    """Not a bug: shap_global.csv is a full per-feature table, so a base
    indicator and its _lag1/_lag2/_slope variants legitimately both appear.
    Aggregating them is the dashboard layer's job (see
    test_dashboard_global_importance_top12_no_label_collisions below), not
    this file's. This test just documents that the raw data does contain
    that overlap, so the dashboard-layer test isn't accidentally checking a
    condition that can never occur."""
    top12 = shap_global.sort_values("mean_abs_shap", ascending=False).head(12)
    base_names = top12["kpi"].str.replace(r"_lag[12]$|_slope$", "", regex=True)
    assert base_names.duplicated().any(), (
        "Expected the raw top-12 SHAP rows to include more than one variant "
        "of some base indicator (this is normal) -- if this now fails, the "
        "dashboard-layer test below may no longer be exercising anything."
    )


def test_dashboard_global_importance_top12_no_label_collisions():
    """Regression test for the actual dashboard bug: the 'What Moves Soft
    Power Globally' chart showed two bars both labeled 'Infant Mortality'
    (infant_mortality and infant_mortality_lag1 both landing in the raw
    top 12), which also broke the bar/tooltip alignment for a feature below
    the collision. Fixed in dash/backend/app/file_data.py's
    _build_global_importance() by aggregating variants of the same base
    indicator before ranking. This test exercises that fixed function
    directly, not the raw shap_global.csv."""
    dash_backend = ROOT / "dash" / "backend"
    if str(dash_backend) not in sys.path:
        sys.path.insert(0, str(dash_backend))
    from app.file_data import FileDataStore

    store = FileDataStore()
    labels = [row["feature"] for row in store.global_importance]
    assert len(labels) == len(set(labels)), (
        f"Dashboard's global_importance has duplicate display labels: {labels}"
    )


# ─────────────────────────────────────────────────────────────────────────
# SoftPowerAgent smoke tests
# ─────────────────────────────────────────────────────────────────────────

def test_agent_loads(agent):
    assert agent.output_df is not None
    assert len(agent.output_df) > 100, "expected ~195 countries in output_df"


def test_agent_country_report_known_country(agent):
    report = agent.get_country_report("USA")
    assert "error" not in report
    for key in ("latent_score", "global_rank", "5y_forecast", "peer_nations"):
        assert key in report, f"get_country_report is missing '{key}'"
    assert 0 <= report["latent_score"] <= 120
    assert len(report["5y_forecast"]) == 5
    assert len(report["peer_nations"]) == 5


def test_agent_country_report_unknown_country_does_not_crash(agent):
    report = agent.get_country_report("ZZZ")
    assert report.get("error"), "an unknown iso3 code should return an error dict, not raise"


def test_agent_global_rankings_sorted(agent):
    rankings = agent.global_rankings(10)
    ranks = rankings["global_rank"].tolist()
    assert ranks == sorted(ranks), "global_rankings() is not sorted by rank"
    assert ranks[0] == 1


def test_agent_regime_clusters_cover_all_countries(agent):
    clusters = agent.regime_clusters()
    assert clusters["n_countries"].sum() == len(agent.output_df)


def test_agent_what_if_returns_effect(agent):
    result = agent.what_if("RUS", "property_rights", 10.0)
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_agent_what_if_bad_feature_does_not_crash(agent):
    result = agent.what_if("USA", "not_a_real_feature", 5.0)
    assert "error" in result


# ─────────────────────────────────────────────────────────────────────────
# Slow tests: actually re-run pipeline scripts end to end. Opt-in via
# --run-slow since these take real time and rewrite output/ files.
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_kalman_script_runs_to_completion():
    result = subprocess.run(
        [sys.executable, "archive/kalman_softpower_complete.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"kalman_softpower_complete.py exited {result.returncode}\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )


@pytest.mark.slow
def test_trust_suite_runs_and_reports_no_new_failures():
    result = subprocess.run(
        [sys.executable, "model_trust_suite.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"model_trust_suite.py crashed:\n{result.stderr[-2000:]}"
    report = (ROOT / "trust_report.md").read_text(encoding="utf-8")
    import re
    m = re.search(r"\*\*Summary:\*\* (\d+) pass, (\d+) warn, (\d+) fail", report)
    assert m, "could not parse trust_report.md summary line"
    fails = int(m.group(3))
    assert fails <= 1, (
        f"trust_report.md now shows {fails} FAILs (expected at most the known "
        f"delta-model FAIL) -- something regressed."
    )
