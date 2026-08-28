"""
regenerate_shap_outputs.py
============================
output/shap_global.csv, output/shap_country.csv, and output/ridge_coefficients.csv
had no generating script anywhere in the repo (see MODEL_TRUST_GUIDE.md section 3
and CHANGES_EXPLAINED.md) -- they were static snapshots from before the
duplicate-row fix (archive/fix_duplicate_rows.py) and before phase_c_complete.py's
multicollinearity fix, with no way to refresh them. This script recreates all
three against the model currently deployed at output/artifacts/xgb_model.pkl and
the current (deduplicated) output/master_phase_b.parquet.

It deliberately explains the SAME feature set the deployed model was actually
trained on (self.model.feature_cols, 92 features) rather than an idealized
corrected set -- SHAP values are only meaningful for the model that's actually
running. Note: that feature set still includes `unesco_cultural_sites`, which
CHANGES_EXPLAINED.md's multicollinearity fix excluded from feature_importance.csv
specifically (built by phase_c_complete.py) but not from the model trained by
phase_c_model_ready.py, which is what output/artifacts/xgb_model.pkl actually is.
That's a separate, pre-existing inconsistency between the two phase-C scripts,
out of scope for this fix -- flagged here for visibility, not corrected.

Run from working/:
    .venv/Scripts/python.exe archive/regenerate_shap_outputs.py
"""

import pickle
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

COUNTRY_COL = 'iso3'
YEAR_COL = 'year'
TARGET_COL = 'soft_power_composite_raw'

TEMPORAL_INDICATORS = [
    'tourist_arrivals', 'unesco_total_sites', 'unesco_cultural_sites',
    'internet_pct_sp', 'rnd_pct_gdp_sp', 'sci_journal_articles',
    'ai_publications', 'hightech_exports_pct',
    'fh_combined_score', 'trade_pct_gdp', 'investment_freedom',
    'govt_integrity', 'judicial_effectiveness', 'property_rights', 'business_freedom',
    'life_expectancy', 'tertiary_enroll_pct', 'physicians_per_1k', 'infant_mortality',
]
DIM_SCORE_COLS = [
    'D1_Cultural_Influence_score', 'D2_Innovation_Knowledge_score',
    'D3_Political_Legitimacy_score', 'D4_Institutional_Quality_score',
    'D5_Human_Development_score',
]
# Current D1-D5 grouping per phase_c_model_ready.py -- used for the 'dimension'
# label on each base KPI. Engineered features (lags, rolls, slopes, year_norm)
# get 'other', same convention the original (stale) shap_global.csv used.
DIMENSION_OF = {
    'tourist_arrivals': 'cultural', 'unesco_total_sites': 'cultural', 'unesco_cultural_sites': 'cultural',
    'internet_pct_sp': 'innovation', 'rnd_pct_gdp_sp': 'innovation', 'sci_journal_articles': 'innovation',
    'ai_publications': 'innovation', 'hightech_exports_pct': 'innovation',
    'fh_combined_score': 'political', 'trade_pct_gdp': 'political', 'investment_freedom': 'political',
    'govt_integrity': 'institutional', 'judicial_effectiveness': 'institutional',
    'property_rights': 'institutional', 'business_freedom': 'institutional',
    'life_expectancy': 'human_dev', 'tertiary_enroll_pct': 'human_dev',
    'physicians_per_1k': 'human_dev', 'infant_mortality': 'human_dev',
}


def build_model_features(df: pd.DataFrame, trend_df: pd.DataFrame) -> pd.DataFrame:
    """Same feature construction as phase_c_model_ready.py's build_model_features,
    reproduced here rather than imported (importing that module re-runs and
    overwrites the whole pipeline as a side effect)."""
    df = df.sort_values([COUNTRY_COL, YEAR_COL]).copy()
    df['target'] = df.groupby(COUNTRY_COL)[TARGET_COL].shift(-1)

    indicators = [c for c in TEMPORAL_INDICATORS if c in df.columns]
    dim_cols = [c for c in DIM_SCORE_COLS if c in df.columns]

    for ind in indicators:
        df[f'{ind}_lag1'] = df.groupby(COUNTRY_COL)[ind].shift(1)
        df[f'{ind}_lag2'] = df.groupby(COUNTRY_COL)[ind].shift(2)
    for dim in dim_cols:
        df[f'{dim}_lag1'] = df.groupby(COUNTRY_COL)[dim].shift(1)

    df['score_roll3_mean'] = df.groupby(COUNTRY_COL)[TARGET_COL].transform(
        lambda x: x.rolling(3, min_periods=2).mean())
    df['score_roll3_std'] = df.groupby(COUNTRY_COL)[TARGET_COL].transform(
        lambda x: x.rolling(3, min_periods=2).std())
    df['year_norm'] = (df[YEAR_COL] - 2000) / 24.0

    slope_cols = [c for c in trend_df.columns
                  if c.endswith('_slope') or c in
                  ('score_volatility', 'influence_growth', 'momentum_5y', 'score_r2')]
    df = df.merge(trend_df[slope_cols].reset_index(), on=COUNTRY_COL, how='left')

    return df


def main():
    print("Loading model + data...")
    with open('output/artifacts/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    booster = model.models[0.50]
    feature_cols = model.feature_cols
    print(f"  Deployed model feature set: {len(feature_cols)} features")

    master = pd.read_parquet('output/master_phase_b.parquet')
    if 'country_iso3' in master.columns and COUNTRY_COL not in master.columns:
        master = master.rename(columns={'country_iso3': COUNTRY_COL})
    trend_df = pd.read_parquet('output/trend_features.parquet')

    full = build_model_features(master, trend_df)
    full_X = full.reindex(columns=feature_cols, fill_value=0).fillna(0)
    print(f"  Full panel for SHAP: {full_X.shape}")

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(full_X)

    # ── shap_global.csv ──────────────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    global_df = pd.DataFrame({
        'kpi': feature_cols,
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_signed,
    })
    base_name = global_df['kpi'].str.replace(r'_lag[12]$|_slope$', '', regex=True)
    global_df['dimension'] = base_name.map(DIMENSION_OF).fillna('other')
    global_df = global_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    global_df['rank'] = global_df.index + 1
    global_df.to_csv('output/shap_global.csv', index=False)
    print(f"  Wrote output/shap_global.csv ({len(global_df)} rows)")
    print(global_df.head(10).to_string(index=False))

    # ── shap_country.csv: latest available row per country ──────────────────
    full_with_shap = full.reset_index(drop=True).copy()
    shap_wide = pd.DataFrame(shap_values, columns=feature_cols)
    full_with_shap = pd.concat([full_with_shap[[COUNTRY_COL, YEAR_COL]], shap_wide], axis=1)
    latest_idx = full_with_shap.groupby(COUNTRY_COL)[YEAR_COL].idxmax()
    country_df = full_with_shap.loc[latest_idx].drop(columns=[YEAR_COL]).reset_index(drop=True)

    abs_vals = country_df[feature_cols].abs()
    country_df['top_kpi'] = abs_vals.idxmax(axis=1)
    top_base = country_df['top_kpi'].str.replace(r'_lag[12]$|_slope$', '', regex=True)
    country_df['top_kpi_dim'] = top_base.map(DIMENSION_OF)
    country_df.to_csv('output/shap_country.csv', index=False)
    print(f"  Wrote output/shap_country.csv ({len(country_df)} rows)")

    # ── ridge_coefficients.csv: same feature set, standardized, next-year target ──
    print("Fitting Ridge regression for coefficient comparison...")
    train = full.dropna(subset=['target'])
    X_train = train.reindex(columns=feature_cols, fill_value=0).fillna(0)
    y_train = train['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_scaled, y_train)

    ridge_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': ridge.coef_,
    })
    ridge_df['abs_coef'] = ridge_df['coefficient'].abs()
    ridge_df = ridge_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    ridge_df.to_csv('output/ridge_coefficients.csv', index=False)
    print(f"  Wrote output/ridge_coefficients.csv ({len(ridge_df)} rows)")
    print(ridge_df.head(10).to_string(index=False))

    print("\nDone.")


if __name__ == '__main__':
    main()
