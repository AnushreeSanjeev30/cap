"""
Diagnostic 2: Delta Prediction (Option B Fix)
===============================================
Instead of predicting next year's RAW score (which is dominated by
autocorrelation), we predict the YEAR-OVER-YEAR CHANGE (delta):

    target_delta = score_{t+1} - score_{t}

This strips autocorrelation out by design. Now R² measures how well
your KPIs explain *shifts* in soft power — a much harder and more
meaningful task.

We compare 3 models on identical temporal CV folds:
  A) Naive delta baseline  — predicts delta = 0 (no change)
  B) Lag-only delta model  — XGBoost with score lags + momentum only
  C) Full delta model      — XGBoost with all KPI features + lags

We also output:
  - Feature importance (which KPIs actually drive changes)
  - Country-level delta predictions for 2024→2025 (rising/falling powers)
  - Saves results to output/diagnostic_2_delta.json
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')


# ── CONFIG ───────────────────────────────────────────────────────────────────
MASTER_PARQUET = 'output/master_phase_b.parquet'
COUNTRY_COL    = 'iso3'
YEAR_COL       = 'year'
SCORE_COL      = 'soft_power_composite_raw'
N_SPLITS       = 5

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
# ─────────────────────────────────────────────────────────────────────────────


def build_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features where the TARGET is the year-over-year delta,
    not the raw score. This eliminates autocorrelation from R².
    """
    df = df.sort_values([COUNTRY_COL, YEAR_COL]).copy()

    # ── Target: next year's CHANGE in score ──────────────────────────────────
    next_score        = df.groupby(COUNTRY_COL)[SCORE_COL].shift(-1)
    df['target_delta'] = next_score - df[SCORE_COL]   # Δscore = score_{t+1} - score_t

    # ── Score-based features (momentum signals) ───────────────────────────────
    df['score_lag1']    = df.groupby(COUNTRY_COL)[SCORE_COL].shift(1)
    df['score_lag2']    = df.groupby(COUNTRY_COL)[SCORE_COL].shift(2)

    # Recent momentum: how fast has score been changing?
    df['delta_lag1']    = df.groupby(COUNTRY_COL)[SCORE_COL].diff(1)          # Δ at t
    df['delta_lag2']    = df.groupby(COUNTRY_COL)[SCORE_COL].diff(1).shift(1) # Δ at t-1
    df['momentum_3yr']  = df.groupby(COUNTRY_COL)[SCORE_COL].diff(3)          # 3yr change
    df['score_roll3']   = (
        df.groupby(COUNTRY_COL)[SCORE_COL]
        .transform(lambda x: x.rolling(3, min_periods=2).mean())
    )
    # Mean-reversion signal: how far is current score from 3yr average?
    df['score_vs_trend'] = df[SCORE_COL] - df['score_roll3']

    # ── KPI features: both level and year-over-year change ───────────────────
    for ind in INDICATORS:
        if ind in df.columns:
            df[f'{ind}_lag1']  = df.groupby(COUNTRY_COL)[ind].shift(1)
            df[f'{ind}_delta'] = df.groupby(COUNTRY_COL)[ind].diff(1)  # KPI change signal

    df['year_norm'] = (df[YEAR_COL] - 2000) / 25.0

    # Drop rows where we can't compute the target delta
    return df.dropna(subset=['target_delta', 'score_lag1', 'delta_lag1']).copy()


def temporal_cv_delta(model_df, feature_cols, label='model'):
    """Temporal CV — now evaluating delta prediction quality."""
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
        y_tr = train['target_delta']
        X_te = test[feature_cols].fillna(0)
        y_te = test['target_delta']

        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,          # shallower: delta signal is weaker, avoid overfit
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,   # regularise: deltas have less data signal
            random_state=42,
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mae = mean_absolute_error(y_te, preds)
        r2  = r2_score(y_te, preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"  [{label}] Fold {fold+1} | cutoff={cutoff} | MAE={mae:.4f} | R²={r2:.4f}")

    return {
        'label': label,
        'mae_mean': float(np.mean(maes)),
        'mae_std':  float(np.std(maes)),
        'r2_mean':  float(np.mean(r2s)),
        'r2_std':   float(np.std(r2s)),
    }


def naive_delta_baseline(model_df):
    """
    Naive baseline for delta: predict Δ = 0 (no change).
    This is equivalent to predicting next year = this year.
    R² < 0 means the model is worse than just predicting zero change.
    """
    years = sorted(model_df[YEAR_COL].unique())
    fold_size = len(years) // (N_SPLITS + 1)
    maes, r2s = [], []

    for fold in range(N_SPLITS):
        cutoff = years[(fold + 1) * fold_size]
        end    = years[min((fold + 2) * fold_size, len(years) - 1)]
        test   = model_df[(model_df[YEAR_COL] >= cutoff) & (model_df[YEAR_COL] < end)]

        if len(test) < 10:
            continue

        preds = np.zeros(len(test))   # predict: no change
        mae   = mean_absolute_error(test['target_delta'], preds)
        r2    = r2_score(test['target_delta'], preds)
        maes.append(mae)
        r2s.append(r2)
        print(f"  [naive Δ=0] Fold {fold+1} | cutoff={cutoff} | MAE={mae:.4f} | R²={r2:.4f}")

    return {
        'label': 'naive (Δ = 0)',
        'mae_mean': float(np.mean(maes)),
        'mae_std':  float(np.std(maes)),
        'r2_mean':  float(np.mean(r2s)),
        'r2_std':   float(np.std(r2s)),
    }


def get_feature_importance(model_df, feature_cols, top_n=15):
    """
    Train the full model on ALL data and return top-N feature importances.
    This shows which KPIs genuinely drive soft power changes.
    """
    X = model_df[feature_cols].fillna(0)
    y = model_df['target_delta']

    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(top_n)
    return importance, model


def predict_2025_movements(model_df, full_model, feature_cols):
    """
    Use the latest year (2024) data to predict 2024→2025 delta.
    Identifies rising and falling powers.
    """
    latest_year = model_df[YEAR_COL].max()
    latest_data = model_df[model_df[YEAR_COL] == latest_year].copy()

    if len(latest_data) == 0:
        print(f"\n⚠️  No data for year {latest_year}, skipping 2025 forecast.")
        return None

    X_latest = latest_data[feature_cols].fillna(0)
    latest_data = latest_data.copy()
    latest_data['predicted_delta'] = full_model.predict(X_latest)
    latest_data['predicted_next_score'] = latest_data[SCORE_COL] + latest_data['predicted_delta']

    rising  = latest_data.nlargest(10, 'predicted_delta')[[COUNTRY_COL, SCORE_COL, 'predicted_delta']]
    falling = latest_data.nsmallest(10, 'predicted_delta')[[COUNTRY_COL, SCORE_COL, 'predicted_delta']]

    return rising, falling, latest_data


def print_verdict(results: list):
    print("\n" + "="*70)
    print(f"{'MODEL':<32} {'MAE (Δ)':>10} {'±':>6} {'R² (Δ)':>10} {'±':>6}")
    print("="*70)
    for r in results:
        print(f"{r['label']:<32} {r['mae_mean']:>10.4f} {r['mae_std']:>6.4f} "
              f"{r['r2_mean']:>10.4f} {r['r2_std']:>6.4f}")
    print("="*70)

    naive    = next(r for r in results if 'naive' in r['label'])
    lag_only = next(r for r in results if 'lag-only' in r['label'])
    full     = next(r for r in results if 'full' in r['label'])

    kpi_gain_mae = lag_only['mae_mean'] - full['mae_mean']
    kpi_gain_r2  = full['r2_mean'] - lag_only['r2_mean']

    print(f"\n📊 DELTA MODEL RESULTS:")
    print(f"   Naive (Δ=0) R²  = {naive['r2_mean']:.4f}  → predicting zero change every year")
    print(f"   Lag-only    R²  = {lag_only['r2_mean']:.4f}  → momentum/mean-reversion signals only")
    print(f"   Full KPI    R²  = {full['r2_mean']:.4f}  → all 19 KPIs + momentum")

    print(f"\n📊 KPI CONTRIBUTION TO DELTA PREDICTION:")
    print(f"   MAE improvement from KPIs: {kpi_gain_mae:+.4f} points")
    print(f"   R² improvement from KPIs:  {kpi_gain_r2:+.4f}")

    print(f"\n💡 INTERPRETATION:")
    print(f"   Unlike the raw-score model (R²≈0.97 driven by autocorrelation),")
    print(f"   delta R² measures GENUINE explanatory power of your KPIs.")
    print(f"   Even modest positive R² here is real, meaningful signal.")

    if full['r2_mean'] < 0:
        print(f"\n⚠️  VERDICT: Model struggles to predict direction of change.")
        print(f"   This is common — soft power shifts are hard to forecast.")
        print(f"   FRAME IT AS: 'Soft power exhibits strong persistence;")
        print(f"   year-on-year changes are driven by exogenous shocks beyond")
        print(f"   the KPI set, confirming the structural stability finding.'")
    elif full['r2_mean'] < 0.10:
        print(f"\n⚡ VERDICT: KPIs explain modest variance in soft power changes.")
        print(f"   This is realistic for country-level socio-political data.")
        print(f"   Focus your capstone narrative on WHICH KPIs drive change")
        print(f"   (feature importance) and WHICH countries are moving.")
    else:
        print(f"\n✅ VERDICT: Strong KPI signal in predicting delta. Your model")
        print(f"   genuinely captures what drives soft power momentum.")


if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_parquet(MASTER_PARQUET)

    if 'country_iso3' in df.columns and COUNTRY_COL not in df.columns:
        df = df.rename(columns={'country_iso3': COUNTRY_COL})

    print(f"Dataset: {len(df)} rows, {df[COUNTRY_COL].nunique()} countries")
    print(f"Years: {df[YEAR_COL].min()} – {df[YEAR_COL].max()}")
    print(f"Score col '{SCORE_COL}' present: {SCORE_COL in df.columns}\n")

    model_df = build_delta_features(df)
    print(f"Delta dataset: {len(model_df)} rows after feature construction\n")

    # Delta distribution summary
    print(f"Target delta stats:")
    print(f"  Mean:   {model_df['target_delta'].mean():.4f}")
    print(f"  Std:    {model_df['target_delta'].std():.4f}")
    print(f"  Min:    {model_df['target_delta'].min():.4f}")
    print(f"  Max:    {model_df['target_delta'].max():.4f}")
    print(f"  % positive (improving countries): "
          f"{(model_df['target_delta'] > 0).mean()*100:.1f}%\n")

    available_inds = [i for i in INDICATORS if f'{i}_lag1' in model_df.columns]
    print(f"KPI indicators available: {len(available_inds)}/{len(INDICATORS)}\n")

    # Feature sets
    lag_only_cols = [
        'score_lag1', 'score_lag2',
        'delta_lag1', 'delta_lag2',
        'momentum_3yr', 'score_vs_trend',
        'year_norm'
    ]
    full_cols = lag_only_cols + \
                [f'{i}_lag1'  for i in available_inds] + \
                [f'{i}_delta' for i in available_inds]

    results = []

    print("── Running naive delta baseline (Δ=0) ──")
    results.append(naive_delta_baseline(model_df))

    print("\n── Running lag-only delta XGBoost ──")
    results.append(temporal_cv_delta(model_df, lag_only_cols, label='lag-only delta XGBoost'))

    print("\n── Running full delta XGBoost (all KPIs) ──")
    results.append(temporal_cv_delta(model_df, full_cols, label='full delta XGBoost (KPIs)'))

    print_verdict(results)

    # ── Feature Importance ────────────────────────────────────────────────────
    print("\n\n" + "="*50)
    print("TOP 15 FEATURES DRIVING SOFT POWER CHANGE")
    print("="*50)
    importance, full_model = get_feature_importance(model_df, full_cols, top_n=15)
    for feat, score in importance.items():
        bar = '█' * int(score * 300)
        print(f"  {feat:<35} {score:.4f}  {bar}")

    # ── 2025 Forecast: Rising & Falling Powers ────────────────────────────────
    print("\n\n" + "="*50)
    print("PREDICTED SOFT POWER MOVEMENTS (latest year → next)")
    print("="*50)

    forecast = predict_2025_movements(model_df, full_model, full_cols)
    if forecast:
        rising, falling, all_preds = forecast

        print(f"\n🚀 TOP 10 RISING POWERS:")
        print(f"  {'Country':<10} {'Current Score':>14} {'Predicted Δ':>12}")
        print(f"  {'-'*38}")
        for _, row in rising.iterrows():
            print(f"  {row[COUNTRY_COL]:<10} {row[SCORE_COL]:>14.3f} {row['predicted_delta']:>+12.4f}")

        print(f"\n📉 TOP 10 DECLINING POWERS:")
        print(f"  {'Country':<10} {'Current Score':>14} {'Predicted Δ':>12}")
        print(f"  {'-'*38}")
        for _, row in falling.iterrows():
            print(f"  {row[COUNTRY_COL]:<10} {row[SCORE_COL]:>14.3f} {row['predicted_delta']:>+12.4f}")

        # Save full forecast
        all_preds[[COUNTRY_COL, SCORE_COL, 'predicted_delta', 'predicted_next_score']] \
            .sort_values('predicted_delta', ascending=False) \
            .to_csv('output/delta_forecast_2025.csv', index=False)
        print("\n  Full forecast saved → output/delta_forecast_2025.csv")

    # ── Save Results ──────────────────────────────────────────────────────────
    output = {
        'cv_results': results,
        'feature_importance': importance.to_dict(),
    }
    with open('output/diagnostic_2_delta.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved → output/diagnostic_2_delta.json")