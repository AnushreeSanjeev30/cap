"""
phase_a_preprocessing.py
==========================
Reimplementation of the pre-processing stage that produced the norm_*/
is_outlier/outlier_score* columns already present in
output/master_soft_power_panel.csv. The original ("Phase A") code was a
notebook, referenced by archive/phase_b_improvements.py's docstring
("Run this after your existing Phase A notebook output") but never
committed to this repo -- there was no way to regenerate, audit, or rerun
that stage of the pipeline at all (see MODEL_TRUST_GUIDE.md /
KALMAN_FILTER_RATIONALE.md, and the architecture-diagram review that
first flagged this as a gap).

This is a clean reimplementation, not a recovery of the original code --
there's no way to guarantee it matches the exact original logic (imputation
strategy, log-transform choices, Isolation Forest hyperparameters are all
underdetermined by the outputs alone). What it does instead: implements
each documented pre-processing step in a defensible, standard way, and
validates itself against the *existing* norm_*/outlier_* columns already
in the panel, reporting agreement rather than assuming it. It does NOT
overwrite those columns -- it writes comparison columns (suffixed _v2) and
prints correlation/agreement stats, so a mismatch is visible rather than
silently replacing trusted values with a guess.

Steps:
  1. Missing value imputation      -- per-country median, falling back to
                                       per-year cross-country median, then
                                       global median, for indicators with
                                       no country history at all.
  2. Log + robust scaling          -- log1p for right-skewed count-like
     normalization                    indicators (tourism, patents, R&D
                                       spend, etc.), then RobustScaler
                                       (median/IQR) for all indicators, so
                                       a handful of extreme countries don't
                                       compress everyone else's range.
  3. Temporal alignment            -- confirms every (iso3, year) combination
                                       in the country's observed year range
                                       has exactly one row (already true of
                                       the current panel; this step is a
                                       check, not a rebuild).
  4. Outlier detection             -- Isolation Forest over the imputed,
     (Isolation Forest)               normalized indicator matrix, one
                                       score per row.

Run from working/:
    .venv/Scripts/python.exe archive/phase_a_preprocessing.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

PANEL_PATH = 'output/master_soft_power_panel.csv'
COUNTRY_COL = 'iso3'
YEAR_COL = 'year'

RAW_INDICATORS = [
    'tourist_arrivals', 'trade_pct_gdp', 'unesco_total_sites', 'unesco_cultural_sites',
    'internet_pct_sp', 'rnd_pct_gdp_sp', 'ai_publications', 'hightech_exports_pct',
    'sci_journal_articles', 'ict_patents', 'fh_combined_score', 'fh_status_num',
    'govt_integrity', 'judicial_effectiveness', 'property_rights', 'business_freedom',
    'investment_freedom', 'life_expectancy', 'tertiary_enroll_pct', 'physicians_per_1k',
    'infant_mortality',
]

# Right-skewed, count-like indicators that plausibly got a log transform
# before scaling (large range, heavy tail toward a handful of big countries).
LOG_TRANSFORM = [
    'tourist_arrivals', 'unesco_total_sites', 'unesco_cultural_sites',
    'ai_publications', 'sci_journal_articles', 'ict_patents',
]


def impute(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        by_country = df.groupby(COUNTRY_COL)[col].transform(lambda s: s.fillna(s.median()))
        by_year = df.groupby(YEAR_COL)[col].transform(lambda s: s.fillna(s.median()))
        df[col] = by_country.fillna(by_year).fillna(df[col].median())
    return df


def normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    values = df[cols].copy()
    for col in LOG_TRANSFORM:
        if col in values.columns:
            values[col] = np.log1p(values[col].clip(lower=0))
    scaled = RobustScaler().fit_transform(values)
    for i, col in enumerate(cols):
        df[f'norm_{col}_v2'] = scaled[:, i]
    return df


def check_temporal_alignment(df: pd.DataFrame) -> None:
    gaps = 0
    for iso3, grp in df.groupby(COUNTRY_COL):
        years = sorted(grp[YEAR_COL].unique())
        expected = set(range(years[0], years[-1] + 1))
        missing = expected - set(years)
        if missing:
            gaps += 1
    print(f"  Countries with a gap inside their own observed year range: {gaps} of {df[COUNTRY_COL].nunique()}")
    if gaps == 0:
        print("  Every country's observed years are contiguous -- no interpolation needed on this panel.")


def detect_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    X = df[cols].copy()
    coverage = 1 - X.isna().mean(axis=1)
    X = X.fillna(X.median())

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    raw_score = -iso.fit_predict(X)  # 1 = outlier, -1 -> flip to 1; normal stays 0... see below
    anomaly_score = -iso.decision_function(X)  # higher = more anomalous

    df['outlier_feature_coverage_v2'] = coverage
    df['outlier_score_raw_v2'] = anomaly_score
    lo, hi = anomaly_score.min(), anomaly_score.max()
    df['outlier_score_norm_v2'] = (anomaly_score - lo) / (hi - lo) if hi > lo else 0.0
    df['is_outlier_v2'] = (iso.predict(X) == -1).astype(int)
    return df


def main():
    print("Loading panel...")
    df = pd.read_csv(PANEL_PATH)
    cols = [c for c in RAW_INDICATORS if c in df.columns]
    print(f"  {len(df)} rows | {len(cols)} raw indicator columns\n")

    print("Step 1: imputation")
    df = impute(df, cols)
    print(f"  Remaining NaNs after imputation: {int(df[cols].isna().sum().sum())}\n")

    print("Step 2: log + robust scaling normalization")
    df = normalize(df, cols)

    print("Step 3: temporal alignment check")
    check_temporal_alignment(df)
    print()

    print("Step 4: outlier detection (Isolation Forest)")
    df = detect_outliers(df, cols)
    print(f"  Flagged as outliers: {int(df['is_outlier_v2'].sum())} of {len(df)} rows "
          f"({df['is_outlier_v2'].mean():.1%})\n")

    print("─── Validation against existing columns ───\n")
    if 'is_outlier' in df.columns:
        agree = (df['is_outlier'] == df['is_outlier_v2']).mean()
        print(f"  is_outlier agreement (reimplementation vs. existing): {agree:.1%}")
    if 'outlier_score_raw' in df.columns:
        corr = df['outlier_score_raw'].corr(df['outlier_score_raw_v2'])
        print(f"  outlier_score_raw correlation (reimplementation vs. existing): {corr:.3f}")
    if 'outlier_feature_coverage' in df.columns:
        corr = df['outlier_feature_coverage'].corr(df['outlier_feature_coverage_v2'])
        print(f"  outlier_feature_coverage correlation: {corr:.3f} "
              f"(should be ~1.0 -- this one is just 'fraction observed', not modeled)")

    norm_corrs = []
    for col in cols:
        existing = f'norm_{col}'
        v2 = f'norm_{col}_v2'
        if existing in df.columns and v2 in df.columns:
            c = df[existing].corr(df[v2])
            norm_corrs.append((col, c))
    if norm_corrs:
        print("\n  norm_* correlation (reimplementation vs. existing), per indicator:")
        for col, c in sorted(norm_corrs, key=lambda t: t[1]):
            print(f"    {col:<26} {c:.3f}")
        mean_corr = np.mean([c for _, c in norm_corrs])
        print(f"\n  Mean norm_* correlation: {mean_corr:.3f}")

    out_path = 'output/phase_a_preprocessing_validation.csv'
    compare_cols = ([COUNTRY_COL, YEAR_COL, 'is_outlier', 'is_outlier_v2',
                     'outlier_score_raw', 'outlier_score_raw_v2'])
    compare_cols = [c for c in compare_cols if c in df.columns]
    df[compare_cols].to_csv(out_path, index=False)
    print(f"\nWrote row-level comparison to {out_path} for manual spot-checking.")
    print("\nNOTE: this script does NOT overwrite the panel's existing norm_*/outlier_*")
    print("columns -- see the module docstring for why. Treat the correlations above")
    print("as evidence of how reproducible the original pre-processing stage is, not")
    print("as a pass/fail gate.")


if __name__ == '__main__':
    main()
