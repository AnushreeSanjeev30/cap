"""
phase_d_multi_model.py
=======================
Trains XGBoost, Random Forest, and LightGBM using the same temporal CV
as phase_c. Compares all three, selects the best, saves the ensemble
as the canonical model for ALL downstream steps (phase 3, agent, dashboard).

Run this AFTER phase_c_model_ready.py.

Inputs:
  output/master_phase_b.parquet
  output/trend_features.parquet
  output/cv_results.json          ← XGB CV results from phase_c

Outputs:
  output/model_comparison.csv           ← CV metrics: MAE, R², CI coverage
  output/ensemble_model.pkl             ← canonical model for all downstream
  output/best_model.pkl                 ← single best model (also saved separately)
  output/best_model_name.txt            ← which model won
  output/ensemble_predictions.csv       ← all 3 scores + ensemble per country
  output/rf_feature_importance.csv
  output/lgbm_feature_importance.csv
  output/xgb_feature_importance.csv     ← copied from phase_c
  output/combined_feature_importance.csv ← weighted ensemble importance
"""

import numpy as np
import pandas as pd
import pickle, json, os
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from archive.softpower_models import (
    SoftPowerXGB, SoftPowerRF, SoftPowerLGBM,
    SoftPowerEnsemble, temporal_cv
)

os.makedirs('output', exist_ok=True)
os.makedirs('output/artifacts', exist_ok=True)

COUNTRY_COL = 'iso3'
YEAR_COL    = 'year'
TARGET_COL  = 'soft_power_composite_raw'

TEMPORAL_INDICATORS = [
    'tourist_arrivals', 'unesco_total_sites', 'unesco_cultural_sites',
    'internet_pct_sp', 'rnd_pct_gdp_sp', 'sci_journal_articles',
    'ai_publications', 'hightech_exports_pct',
    'fh_combined_score', 'trade_pct_gdp', 'investment_freedom',
    'govt_integrity', 'judicial_effectiveness', 'property_rights', 'business_freedom',
    'life_expectancy', 'tertiary_enroll_pct', 'physicians_per_1k', 'infant_mortality',
]

print("═" * 60)
print("  PHASE D — MULTI-MODEL COMPARISON + ENSEMBLE")
print("═" * 60)


# ─── 1. LOAD AND REBUILD FEATURES (same as phase_c) ──────────────────────────

master = pd.read_parquet('output/master_phase_b.parquet')
trend  = pd.read_parquet('output/trend_features.parquet').reset_index()

TEMPORAL_INDICATORS = [c for c in TEMPORAL_INDICATORS if c in master.columns]
DIM_SCORE_COLS = [c for c in master.columns
                  if c.endswith('_score') and c.startswith('D')]

master_s = master.sort_values([COUNTRY_COL, YEAR_COL]).copy()
master_s['target'] = master_s.groupby(COUNTRY_COL)[TARGET_COL].shift(-1)

for ind in TEMPORAL_INDICATORS:
    master_s[f'{ind}_lag1'] = master_s.groupby(COUNTRY_COL)[ind].shift(1)
    master_s[f'{ind}_lag2'] = master_s.groupby(COUNTRY_COL)[ind].shift(2)

for dim in DIM_SCORE_COLS:
    master_s[f'{dim}_lag1'] = master_s.groupby(COUNTRY_COL)[dim].shift(1)

master_s['score_roll3_mean'] = (
    master_s.groupby(COUNTRY_COL)[TARGET_COL]
             .transform(lambda x: x.rolling(3, min_periods=2).mean()))
master_s['score_roll3_std'] = (
    master_s.groupby(COUNTRY_COL)[TARGET_COL]
             .transform(lambda x: x.rolling(3, min_periods=2).std()))
master_s['year_norm'] = (master_s[YEAR_COL] - 2000) / 24.0

slope_cols = [c for c in trend.columns
              if c.endswith('_slope') or c in
              ['score_volatility', 'influence_growth', 'momentum_5y', 'score_r2']]
master_s = master_s.merge(trend[[COUNTRY_COL] + slope_cols],
                           on=COUNTRY_COL, how='left')

model_df = master_s.dropna(subset=['target']).copy()

EXCLUDE = {COUNTRY_COL, YEAR_COL, 'target', TARGET_COL, 'canonical',
           'regime', 'soft_power_composite_adjusted', 'data_reliability'}
EXCLUDE.update(DIM_SCORE_COLS)
EXCLUDE.update([c for c in model_df.columns
                if c.startswith(('D1_','D2_','D3_','D4_','D5_'))
                and c not in TEMPORAL_INDICATORS])

FEATURE_COLS = [c for c in model_df.select_dtypes(include='number').columns
                if c not in EXCLUDE]

print(f"\n[Data] {len(model_df)} rows | {model_df[COUNTRY_COL].nunique()} countries")
print(f"[Data] {len(FEATURE_COLS)} features\n")


# ─── 2. TEMPORAL CV FOR ALL THREE MODELS ─────────────────────────────────────

print("─── Temporal Cross-Validation (4 folds) ───\n")

cv_results = {}
model_classes = [
    ('XGBoost',      SoftPowerXGB,  {}),
    ('RandomForest', SoftPowerRF,   {}),
    ('LightGBM',     SoftPowerLGBM, {}),
]

for name, cls, kwargs in model_classes:
    print(f"  [{name}] running CV...")
    metrics = temporal_cv(cls, model_df, FEATURE_COLS,
                          model_kwargs=kwargs)
    cv_results[name] = metrics
    print(f"    MAE={metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f}  "
          f"R²={metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}  "
          f"CI cov={metrics['ci_coverage']:.0%}\n")

# Also load XGB results from phase_c if available (for reference)
if os.path.exists('output/cv_results.json'):
    with open('output/cv_results.json') as f:
        xgb_phase_c = json.load(f)
    print(f"  [XGBoost phase_c reference] "
          f"MAE={xgb_phase_c['mae_mean']:.3f}  R²={xgb_phase_c['r2_mean']:.3f}")

# ─── 3. COMPARISON TABLE ─────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"  MODEL COMPARISON (Temporal CV)")
print(f"{'═'*60}\n")

comparison_rows = []
for name, metrics in cv_results.items():
    comparison_rows.append({
        'model':       name,
        'mae_mean':    metrics['mae_mean'],
        'mae_std':     metrics['mae_std'],
        'r2_mean':     metrics['r2_mean'],
        'r2_std':      metrics['r2_std'],
        'ci_coverage': metrics['ci_coverage'],
    })

comparison_df = (pd.DataFrame(comparison_rows)
                 .sort_values('r2_mean', ascending=False)
                 .reset_index(drop=True))

print(f"  {'Model':<18} {'MAE':>8} {'±':>6} {'R²':>8} {'±':>6} {'CI Cov':>8}")
print(f"  {'-'*58}")
for i, row in comparison_df.iterrows():
    best = ' ← BEST' if i == 0 else ''
    print(f"  {row['model']:<18} {row['mae_mean']:>8.3f} {row['mae_std']:>6.3f} "
          f"{row['r2_mean']:>8.3f} {row['r2_std']:>6.3f} "
          f"{row['ci_coverage']:>8.1%}{best}")

comparison_df.to_csv('output/model_comparison.csv', index=False)
best_model_name = comparison_df.iloc[0]['model']
print(f"\n  Best single model: {best_model_name}")


# ─── 4. TRAIN FINAL MODELS ON ALL DATA ───────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Training final models on full dataset...")
print(f"{'─'*60}\n")

X_all = model_df[FEATURE_COLS].fillna(0)
y_all = model_df['target']

# Train ensemble (trains all three internally)
ensemble = SoftPowerEnsemble()
print("  Fitting all models...")
ensemble.fit(X_all, y_all)

# Temporal validation split for compare() — use last 20% of years
all_years  = sorted(model_df[YEAR_COL].unique())
split_year = all_years[int(len(all_years) * 0.8)]
val_mask   = model_df[YEAR_COL] >= split_year
X_val = model_df.loc[val_mask, FEATURE_COLS].fillna(0)
y_val = model_df.loc[val_mask, 'target']

comparison_val = ensemble.compare(X_val, y_val)
print(f"\n  Best model (validation): {ensemble.best_name}")

# Set best model based on CV (more robust than single val set)
# Override if CV winner differs from val winner
cv_best = comparison_df.iloc[0]['model']
if cv_best != ensemble.best_name:
    print(f"  Note: CV best={cv_best}, val best={ensemble.best_name}. Using CV winner.")
    ensemble.best_name  = cv_best
    ensemble.best_model = ensemble.members[cv_best]
    ensemble.weights    = {n: (0.5 if n == cv_best else 0.25)
                           for n in ensemble.members}


# ─── 5. SAVE MODELS ──────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print("  Saving models...")

# Canonical ensemble (used by phase3, agent, dashboard)
ensemble.save('output/ensemble_model.pkl')

# Best single model (backward compat with phase_c consumers)
with open('output/best_model.pkl', 'wb') as f:
    pickle.dump(ensemble.best_model, f)

# Also overwrite xgb_model.pkl with best model for backward compat
with open('output/artifacts/xgb_model.pkl', 'wb') as f:
    pickle.dump(ensemble.best_model, f)

with open('output/best_model_name.txt', 'w') as f:
    f.write(ensemble.best_name)

print(f"  ✅ output/ensemble_model.pkl       ← use this everywhere downstream")
print(f"  ✅ output/best_model.pkl           ← best single model")
print(f"  ✅ output/artifacts/xgb_model.pkl  ← updated to best model")
print(f"  ✅ output/best_model_name.txt      ← '{ensemble.best_name}'")


# ─── 6. FEATURE IMPORTANCE COMPARISON ────────────────────────────────────────

print(f"\n{'─'*60}")
print("  Feature importance comparison...")

xgb_fi  = ensemble.xgb.feature_importance()
rf_fi   = ensemble.rf.feature_importance()
lgbm_fi = ensemble.lgbm.feature_importance()
ens_fi  = ensemble.feature_importance()

xgb_fi.to_csv('output/xgb_feature_importance.csv', index=False)
rf_fi.to_csv('output/rf_feature_importance.csv', index=False)
lgbm_fi.to_csv('output/lgbm_feature_importance.csv', index=False)
ens_fi.to_csv('output/combined_feature_importance.csv', index=False)

# Normalise for display
def norm(series):
    s = series / series.sum() * 100
    return s.round(1)

xgb_dict  = dict(zip(xgb_fi['feature'],  norm(xgb_fi['importance'])))
rf_dict   = dict(zip(rf_fi['feature'],   norm(rf_fi['importance'])))
lgbm_dict = dict(zip(lgbm_fi['feature'], norm(lgbm_fi['importance'])))

top_feats = list(dict.fromkeys(
    list(ens_fi['feature'][:6]) +
    list(xgb_fi['feature'][:4]) +
    list(rf_fi['feature'][:4]) +
    list(lgbm_fi['feature'][:4])
))[:15]

print(f"\n  {'Feature':<35} {'XGB':>7} {'RF':>7} {'LGBM':>7}  {'Ensemble':>9}")
print(f"  {'-'*70}")
ens_dict = dict(zip(ens_fi['feature'], norm(ens_fi['importance'])))
for feat in top_feats:
    xv = f"{xgb_dict.get(feat,0):.1f}%"  if feat in xgb_dict  else "—"
    rv = f"{rf_dict.get(feat,0):.1f}%"   if feat in rf_dict   else "—"
    lv = f"{lgbm_dict.get(feat,0):.1f}%" if feat in lgbm_dict else "—"
    ev = f"{ens_dict.get(feat,0):.1f}%"  if feat in ens_dict  else "—"
    print(f"  {feat:<35} {xv:>7} {rv:>7} {lv:>7}  {ev:>9}")


# ─── 7. ENSEMBLE PREDICTIONS FOR ALL COUNTRIES ───────────────────────────────

print(f"\n{'─'*60}")
print("  Generating ensemble predictions for all countries...")

latest_per_country = (
    model_df.sort_values([COUNTRY_COL, YEAR_COL])
            .groupby(COUNTRY_COL).last()
            .reset_index()
)
X_pred = latest_per_country[FEATURE_COLS].fillna(0)

# All model predictions
all_preds = ensemble.predict_all(X_pred)
all_preds[COUNTRY_COL] = latest_per_country[COUNTRY_COL].values

# Add country names if available
if 'canonical' in latest_per_country.columns:
    all_preds['country_name'] = latest_per_country['canonical'].values

# Ranking columns
for col in ['xgboost_score', 'randomforest_score', 'lightgbm_score', 'ensemble_score']:
    if col in all_preds.columns:
        rank_col = col.replace('_score', '_rank')
        all_preds[rank_col] = (all_preds[col]
                               .rank(ascending=False, method='min')
                               .astype(int))

all_preds = all_preds.sort_values('ensemble_rank' if 'ensemble_rank' in all_preds.columns
                                  else 'ensemble_score', ascending=False if 'ensemble_score' in all_preds.columns else True)

all_preds.to_csv('output/ensemble_predictions.csv', index=False)
all_preds.to_parquet('output/ensemble_predictions.parquet', index=False)

# Display top 20
disp = [c for c in ['country_name', COUNTRY_COL,
                     'xgboost_score', 'randomforest_score', 'lightgbm_score',
                     'ensemble_score', 'ensemble_rank']
        if c in all_preds.columns]
print(f"\n  Top 20 (ensemble ranking):")
print(all_preds[disp].head(20).to_string(index=False))


# ─── 8. SAVE CV RESULTS JSON (for downstream reference) ──────────────────────

all_cv = {
    name: metrics for name, metrics in cv_results.items()
}
all_cv['best_model'] = best_model_name
all_cv['ensemble_weights'] = ensemble.weights

with open('output/multi_model_cv_results.json', 'w') as f:
    json.dump(all_cv, f, indent=2)


# ─── 9. FINAL SUMMARY ────────────────────────────────────────────────────────

best_row = comparison_df.iloc[0]
print(f"\n{'═'*60}")
print(f"  PHASE D COMPLETE")
print(f"{'═'*60}")
print(f"\n  Winner:   {best_model_name}")
print(f"  Best R²:  {best_row['r2_mean']:.3f} ± {best_row['r2_std']:.3f}")
print(f"  Best MAE: {best_row['mae_mean']:.3f} pts")
print(f"\n  Ensemble weights: {ensemble.weights}")
print(f"\n  Files saved:")
print(f"    output/ensemble_model.pkl              ← USE THIS in phase3 + agent")
print(f"    output/best_model.pkl                  ← best single model")
print(f"    output/model_comparison.csv            ← CV performance table")
print(f"    output/ensemble_predictions.csv        ← all scores per country")
print(f"    output/combined_feature_importance.csv ← weighted importance")
print(f"    output/multi_model_cv_results.json     ← all CV metrics")
print(f"\n  NEXT STEP:")
print(f"    python phase3_causal_complete.py --track A")
print(f"  (phase3 will auto-load ensemble_model.pkl)")
print(f"{'═'*60}")
