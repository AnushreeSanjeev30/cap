"""
Diagnostic 1: Autocorrelation Test
====================================
Question: Is your R²=0.966 actually from learning soft power drivers,
or just because "next year ≈ this year" (autocorrelation)?

We compare 3 models on identical temporal CV folds:
  A) Naive baseline   — predicts next year = this year's score (no ML)
  B) Lag-only model   — XGBoost with ONLY score_lag1 as feature
  C) Full model       — XGBoost with all KPI features + lags

If (B) ≈ (C), your 23 KPIs are adding almost nothing.
If (C) >> (B), your features have real signal.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')


# ── CONFIG ──────────────────────────────────────────────────────────────────
# Adjust these paths to match your project layout
MASTER_PARQUET  = 'output/master_phase_b.parquet'
TREND_PARQUET   = 'output/trend_features.parquet'
COUNTRY_COL     = 'iso3'          # change if your column is named differently
YEAR_COL        = 'year'
SCORE_COL       = 'soft_power_composite_raw'   # your PCA score column
N_SPLITS        = 5

INDICATORS = [
    'tourist_arrivals',
    'unesco_total_sites',
    'unesco_cultural_sites',
    'internet_pct_sp',
    'rnd_pct_gdp_sp',
    'sci_journal_articles',
    'ai_publications',
    'hightech_exports_pct',
    'fh_combined_score',
    'trade_pct_gdp',
    'investment_freedom',
    'govt_integrity',
    'judicial_effectiveness',
    'property_rights',
    'business_freedom',
    'life_expectancy',
    'tertiary_enroll_pct',
    'physicians_per_1k',
    'infant_mortality'
]
# ────────────────────────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([COUNTRY_COL, YEAR_COL]).copy()

    # Target: next year's score
    df['target_score'] = df.groupby(COUNTRY_COL)[SCORE_COL].shift(-1)

    # Lag-1 of score (the key autocorrelation feature)
    df['score_lag1'] = df.groupby(COUNTRY_COL)[SCORE_COL].shift(1)
    df['score_lag2'] = df.groupby(COUNTRY_COL)[SCORE_COL].shift(2)
    df['score_roll3'] = (
        df.groupby(COUNTRY_COL)[SCORE_COL]
        .transform(lambda x: x.rolling(3, min_periods=2).mean())
    )

    # KPI lags
    for ind in INDICATORS:
        if ind in df.columns:
            df[f'{ind}_lag1'] = df.groupby(COUNTRY_COL)[ind].shift(1)

    df['year_norm'] = (df[YEAR_COL] - 2000) / 25.0
    return df.dropna(subset=['target_score', 'score_lag1']).copy()


def temporal_cv(model_df, feature_cols, label='model'):
    years = sorted(model_df[YEAR_COL].unique())
    fold_size = len(years) // (N_SPLITS + 1)
    maes, r2s = [], []

    for fold in range(N_SPLITS):
        cutoff = years[(fold + 1) * fold_size]
        end    = years[min((fold + 2) * fold_size, len(years) - 1)]

        train = model_df[model_df[YEAR_COL] < cutoff]
        test  = model_df[(model_df[YEAR_COL] >= cutoff) & (model_df[YEAR_COL] < end)]

        if len(test) < 10:
            continue

        X_tr = train[feature_cols].fillna(0)
        y_tr = train['target_score']
        X_te = test[feature_cols].fillna(0)
        y_te = test['target_score']

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mae = mean_absolute_error(y_te, preds)
        r2  = r2_score(y_te, preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"  [{label}] Fold {fold+1} | cutoff={cutoff} | MAE={mae:.3f} | R²={r2:.3f}")

    return {
        'label': label,
        'mae_mean': float(np.mean(maes)),
        'mae_std':  float(np.std(maes)),
        'r2_mean':  float(np.mean(r2s)),
        'r2_std':   float(np.std(r2s)),
    }


def naive_baseline(model_df):
    """Predict next year's score = this year's score. No ML at all."""
    years = sorted(model_df[YEAR_COL].unique())
    fold_size = len(years) // (N_SPLITS + 1)
    maes, r2s = [], []

    for fold in range(N_SPLITS):
        cutoff = years[(fold + 1) * fold_size]
        end    = years[min((fold + 2) * fold_size, len(years) - 1)]
        test   = model_df[(model_df[YEAR_COL] >= cutoff) & (model_df[YEAR_COL] < end)]

        if len(test) < 10:
            continue

        # Naive: predicted = score_lag1 (i.e. current year score)
        preds = test['score_lag1'].fillna(test['target_score'].mean())
        mae   = mean_absolute_error(test['target_score'], preds)
        r2    = r2_score(test['target_score'], preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"  [naive] Fold {fold+1} | cutoff={cutoff} | MAE={mae:.3f} | R²={r2:.3f}")

    return {
        'label': 'naive (next = current)',
        'mae_mean': float(np.mean(maes)),
        'mae_std':  float(np.std(maes)),
        'r2_mean':  float(np.mean(r2s)),
        'r2_std':   float(np.std(r2s)),
    }


def print_verdict(results: list):
    print("\n" + "="*65)
    print(f"{'MODEL':<30} {'MAE':>8} {'±':>6} {'R²':>8} {'±':>6}")
    print("="*65)
    for r in results:
        print(f"{r['label']:<30} {r['mae_mean']:>8.3f} {r['mae_std']:>6.3f} "
              f"{r['r2_mean']:>8.4f} {r['r2_std']:>6.4f}")
    print("="*65)

    naive   = next(r for r in results if 'naive' in r['label'])
    lag_only = next(r for r in results if 'lag-only' in r['label'])
    full    = next(r for r in results if 'full' in r['label'])

    kpi_gain_mae = lag_only['mae_mean'] - full['mae_mean']
    kpi_gain_r2  = full['r2_mean'] - lag_only['r2_mean']
    autocorr_r2  = lag_only['r2_mean'] - naive['r2_mean']

    print(f"\n📊 AUTOCORRELATION SHARE:")
    print(f"   Naive R²={naive['r2_mean']:.4f}  →  this is free signal from year-to-year persistence")
    print(f"   Lag-only R²={lag_only['r2_mean']:.4f}  →  XGBoost on score_lag1 alone")
    print(f"   Full model R²={full['r2_mean']:.4f}  →  all 23 KPIs + lags")

    print(f"\n📊 KPI CONTRIBUTION:")
    print(f"   MAE improvement from adding KPIs: {kpi_gain_mae:+.3f} points")
    print(f"   R² improvement from adding KPIs:  {kpi_gain_r2:+.4f}")

    if kpi_gain_r2 < 0.01:
        print("\n⚠️  VERDICT: KPIs add very little over the lag alone.")
        print("   Your high R² is largely autocorrelation.")
        print("   ACTION: Report lag-only as your baseline. Investigate")
        print("   whether your feature engineering is leaking the score.")
    elif kpi_gain_r2 < 0.03:
        print("\n⚡ VERDICT: KPIs add modest but real signal.")
        print("   Your model is partially autocorrelation-driven.")
        print("   ACTION: Highlight specific KPIs (feature importance) as")
        print("   genuine drivers. Frame R² carefully in your capstone.")
    else:
        print("\n✅ VERDICT: KPIs add meaningful signal beyond autocorrelation.")
        print("   Your model is learning real soft power dynamics.")


if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_parquet(MASTER_PARQUET)

    # Handle column name variants
    if 'country_iso3' in df.columns and COUNTRY_COL not in df.columns:
        df = df.rename(columns={'country_iso3': COUNTRY_COL})

    print(f"Dataset: {len(df)} rows, {df[COUNTRY_COL].nunique()} countries")
    print(f"Years: {df[YEAR_COL].min()} – {df[YEAR_COL].max()}")
    print(f"Score col '{SCORE_COL}' present: {SCORE_COL in df.columns}\n")

    model_df = build_features(df)

    # Filter to indicators actually present in the data
    available_inds = [i for i in INDICATORS if f'{i}_lag1' in model_df.columns]
    print(f"KPI indicators available: {len(available_inds)}/{len(INDICATORS)}\n")

    # Feature sets
    lag_only_cols = ['score_lag1', 'score_lag2', 'score_roll3', 'year_norm']
    full_cols     = lag_only_cols + [f'{i}_lag1' for i in available_inds]

    results = []

    print("── Running naive baseline ──")
    results.append(naive_baseline(model_df))

    print("\n── Running lag-only XGBoost ──")
    results.append(temporal_cv(model_df, lag_only_cols, label='lag-only XGBoost'))

    print("\n── Running full XGBoost (all KPIs) ──")
    results.append(temporal_cv(model_df, full_cols, label='full XGBoost (KPIs)'))

    print_verdict(results)

    # Save results
    with open('output/diagnostic_1_autocorrelation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved → output/diagnostic_1_autocorrelation.json")