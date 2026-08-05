"""
Phase C — Model-Ready Pipeline
Run this file next, after phase_b_improvements.py has completed.

Input  : output/master_phase_b.parquet
Outputs: output/trend_features.parquet
         output/country_embeddings.parquet
         output/faiss_index.bin
         output/xgb_model.pkl
         output/soft_power_predictions.parquet
         output/cv_results.json

What this file does:
  1. Loads master_phase_b with correct indicator list
  2. Separates static vs temporal features (leakage fix)
  3. Computes trend slope features (Phase 1)
  4. Builds FAISS country embedding index (Phase 1)
  5. Trains quantile XGBoost with temporal CV (Phase 2)
  6. Generates final prediction table with CI, rank, regime, forecast stub
"""

import numpy as np
import pandas as pd
import pickle, json, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import faiss
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)
os.makedirs('output/artifacts', exist_ok=True)

# ─── 0. CONFIGURATION ────────────────────────────────────────────────────────

# These are your 18 CLEAN indicators after Phase B collinearity audit
# (removed: fh_status_num, fh_total_score — redundant with fh_combined_score)
# (trade_pct_gdp and investment_freedom moved to D3)
TEMPORAL_INDICATORS = [
    # D1 - Cultural Influence (3 indicators)
    'tourist_arrivals',
    'unesco_total_sites',
    'unesco_cultural_sites',
    # D2 - Innovation & Knowledge (5 indicators)
    'internet_pct_sp',
    'rnd_pct_gdp_sp',
    'sci_journal_articles',
    'ai_publications',
    'hightech_exports_pct',
    # D3 - Political Legitimacy (3 indicators)
    'fh_combined_score',
    'trade_pct_gdp',
    'investment_freedom',
    # D4 - Institutional Quality (4 indicators)
    'govt_integrity',
    'judicial_effectiveness',
    'property_rights',
    'business_freedom',
    # D5 - Human Development (4 indicators - infant_mortality inverted in scoring)
    'life_expectancy',
    'tertiary_enroll_pct',
    'physicians_per_1k',
    'infant_mortality',
]

# Static features (near-zero temporal variation — use as country fixed effects ONLY)
# Do NOT use these as time-varying predictors in the XGBoost
STATIC_INDICATORS = [
    'unesco_total_area_ha',
    'unesco_danger_sites',
    'unesco_mixed_sites',
    # add any other static columns flagged in Phase B leakage check
]

# Dimension score columns produced by Phase B
DIM_SCORE_COLS = [
    'D1_Cultural_Influence_score',
    'D2_Innovation_Knowledge_score',
    'D3_Political_Legitimacy_score',
    'D4_Institutional_Quality_score',
    'D5_Human_Development_score',
]

TARGET_COL    = 'soft_power_composite_raw'   # from phase_b_improvements.py
COUNTRY_COL   = 'iso3'
YEAR_COL      = 'year'

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────

print("═" * 60)
print("  PHASE C — SOFT POWER PREDICTIVE PIPELINE")
print("═" * 60)

master = pd.read_parquet('output/master_phase_b.parquet')
print(f"\n[Load] {master.shape} | {master[COUNTRY_COL].nunique()} countries | "
      f"{master[YEAR_COL].min()}–{master[YEAR_COL].max()}")

# Keep only columns we actually have
TEMPORAL_INDICATORS = [c for c in TEMPORAL_INDICATORS if c in master.columns]
DIM_SCORE_COLS      = [c for c in DIM_SCORE_COLS if c in master.columns]
print(f"[Load] Temporal indicators: {len(TEMPORAL_INDICATORS)}")
print(f"[Load] Dimension scores available: {DIM_SCORE_COLS}")

# ─── 2. TREND SLOPE ANALYSIS ─────────────────────────────────────────────────

print("\n─── Phase C.1: Trend Slope Analysis ───\n")

def compute_trend_features(df, indicators, country_col=COUNTRY_COL, 
                            year_col=YEAR_COL, score_col=TARGET_COL):
    records = []
    for country, group in df.groupby(country_col):
        group = group.sort_values(year_col)
        years = group[year_col].values.astype(float)
        row   = {country_col: country}

        # Per-indicator OLS slope
        for ind in indicators:
            vals = group[ind].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() < 4:
                row[f'{ind}_slope'] = np.nan
            else:
                slope, _, r, _, _ = stats.linregress(years[mask], vals[mask])
                row[f'{ind}_slope'] = slope
                row[f'{ind}_r2']    = r ** 2

        # Composite score trend
        if score_col in group.columns:
            scores = group[score_col].values.astype(float)
            mask   = ~np.isnan(scores)
            if mask.sum() >= 5:
                slope, intercept, r, p, se = stats.linregress(years[mask], scores[mask])
                fitted    = intercept + slope * years[mask]
                residuals = scores[mask] - fitted

                row['score_slope']      = slope
                row['score_r2']         = r ** 2
                row['score_volatility'] = float(np.std(residuals))
                row['score_mean']       = float(np.nanmean(scores))
                row['influence_growth'] = slope / (row['score_mean'] + 1e-6)

                # Momentum: last 5 years vs prior 5 years
                n = mask.sum()
                if n >= 8:
                    recent = scores[mask][-5:].mean()
                    prior  = scores[mask][-10:-5].mean() if n >= 10 else scores[mask][:5].mean()
                    row['momentum_5y'] = float(recent - prior)
                else:
                    row['momentum_5y'] = np.nan

                # Regime classification
                if   slope >  0.4 and r**2 > 0.25: row['regime'] = 'rising'
                elif slope < -0.4 and r**2 > 0.25: row['regime'] = 'declining'
                else:                               row['regime'] = 'stable'

        records.append(row)

    trend_df = pd.DataFrame(records).set_index(country_col)
    print(f"  Trend features computed for {len(trend_df)} countries")
    return trend_df


trend_df = compute_trend_features(master, TEMPORAL_INDICATORS)
trend_df.to_parquet('output/trend_features.parquet')

# Regime summary
if 'regime' in trend_df.columns:
    print(f"\n  Regime distribution:")
    print(trend_df['regime'].value_counts().to_string())

# ─── 3. COUNTRY EMBEDDING VECTORS (FAISS) ────────────────────────────────────

print("\n─── Phase C.2: Country Similarity Vectors (FAISS) ───\n")

def build_embeddings(df, indicators, country_col=COUNTRY_COL, 
                     year_col=YEAR_COL, recent_years=5):
    latest_year = df[year_col].max()
    recent      = df[df[year_col] > latest_year - recent_years]
    
    records = []
    for country, grp in recent.groupby(country_col):
        vec = {country_col: country}
        for ind in indicators:
            vec[f'{ind}_mean'] = grp[ind].mean()
            vec[f'{ind}_std']  = grp[ind].std()
        records.append(vec)
    
    embed_df = pd.DataFrame(records).set_index(country_col)
    embed_df = embed_df.fillna(embed_df.mean())  # fill so FAISS doesn't break
    
    scaler = StandardScaler()
    matrix = scaler.fit_transform(embed_df.values).astype('float32')
    # Ensure matrix is C-contiguous for FAISS
    import numpy as np
    matrix = np.ascontiguousarray(matrix)
    faiss.normalize_L2(matrix)
    
    return embed_df, matrix, scaler


embed_df, embed_matrix, embed_scaler = build_embeddings(master, TEMPORAL_INDICATORS)

d = embed_matrix.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embed_matrix)

embed_df.to_parquet('output/country_embeddings.parquet')
faiss.write_index(index, 'output/artifacts/faiss_index.bin')
with open('output/artifacts/embedding_scaler.pkl', 'wb') as f:
    pickle.dump(embed_scaler, f)
np.save('output/artifacts/embedding_matrix.npy', embed_matrix)

print(f"  FAISS index: {index.ntotal} country vectors, dim={d}")

# Quick test — top 5 peers for India
def get_peers(iso3_query, k=5):
    countries = list(embed_df.index)
    if iso3_query not in countries:
        return []
    q_idx = countries.index(iso3_query)
    q_vec = embed_matrix[q_idx:q_idx+1]
    scores, idxs = index.search(q_vec, k+1)
    return [(countries[i], round(float(s), 3)) 
            for s, i in zip(scores[0], idxs[0]) if countries[i] != iso3_query][:k]

print(f"\n  Sample — top peers for IND: {get_peers('IND')}")
print(f"  Sample — top peers for USA: {get_peers('USA')}")

# ─── 4. FEATURE ENGINEERING FOR XGBOOST ─────────────────────────────────────

print("\n─── Phase C.3: Feature Engineering ───\n")

def build_model_features(df, trend_df, indicators, dim_score_cols,
                          country_col=COUNTRY_COL, year_col=YEAR_COL,
                          target_col=TARGET_COL):
    df = df.sort_values([country_col, year_col]).copy()
    
    # ── Target: predict NEXT year's composite score ─────────────────────────
    df['target'] = df.groupby(country_col)[target_col].shift(-1)
    
    # ── Lag features (t-1, t-2) for temporal indicators ────────────────────
    for ind in indicators:
        df[f'{ind}_lag1'] = df.groupby(country_col)[ind].shift(1)
        df[f'{ind}_lag2'] = df.groupby(country_col)[ind].shift(2)
    
    # ── Lag features for dimension scores ───────────────────────────────────
    for dim in dim_score_cols:
        df[f'{dim}_lag1'] = df.groupby(country_col)[dim].shift(1)
    
    # ── Rolling statistics ───────────────────────────────────────────────────
    df['score_roll3_mean'] = (
        df.groupby(country_col)[target_col]
          .transform(lambda x: x.rolling(3, min_periods=2).mean())
    )
    df['score_roll3_std'] = (
        df.groupby(country_col)[target_col]
          .transform(lambda x: x.rolling(3, min_periods=2).std())
    )
    
    # ── Year as a feature ────────────────────────────────────────────────────
    df['year_norm'] = (df[year_col] - 2000) / 24.0
    
    # ── Merge trend slope features (cross-sectional, not time-varying) ──────
    slope_cols = [c for c in trend_df.columns 
                  if c.endswith('_slope') or c in 
                  ['score_volatility', 'influence_growth', 'momentum_5y', 'score_r2']]
    
    df = df.merge(
        trend_df[slope_cols].reset_index(),
        on=country_col, how='left'
    )
    
    # ── Drop rows without a target ───────────────────────────────────────────
    model_df = df.dropna(subset=['target']).copy()
    
    print(f"  Model dataset: {len(model_df)} rows | "
          f"{model_df[country_col].nunique()} countries | "
          f"{model_df[year_col].min()}–{model_df[year_col].max()}")
    return model_df


model_df = build_model_features(master, trend_df, TEMPORAL_INDICATORS, DIM_SCORE_COLS)


def get_feature_cols(df, exclude=None):
    if exclude is None:
        exclude = [COUNTRY_COL, YEAR_COL, 'target', TARGET_COL,
                   'canonical', 'regime', 'soft_power_composite_adjusted',
                   'data_reliability'] + DIM_SCORE_COLS
    return [c for c in df.select_dtypes(include='number').columns
            if c not in exclude and not c.startswith('D1_') 
            and not c.startswith('D2_') and not c.startswith('D3_')
            and not c.startswith('D4_') and not c.startswith('D5_')]


FEATURE_COLS = get_feature_cols(model_df)
print(f"  Feature columns: {len(FEATURE_COLS)}")

# ─── 5. TEMPORAL CROSS-VALIDATION ────────────────────────────────────────────

print("\n─── Phase C.4: Temporal Cross-Validation ───\n")

class SoftPowerXGB:
    """Quantile XGBoost — predicts score + 80% confidence interval."""
    QUANTILES = [0.10, 0.50, 0.90]
    
    def __init__(self, n_estimators=500, lr=0.05, max_depth=6):
        self.params = dict(
            n_estimators   = n_estimators,
            learning_rate  = lr,
            max_depth      = max_depth,
            subsample      = 0.8,
            colsample_bytree = 0.75,
            min_child_weight = 5,
            reg_alpha      = 0.1,
            reg_lambda     = 1.0,
            tree_method    = 'hist',
            random_state   = 42,
        )
        self.models      = {}
        self.feature_cols = None
    
    def fit(self, X, y):
        self.feature_cols = list(X.columns)
        for q in self.QUANTILES:
            m = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                **self.params
            )
            m.fit(X, y, verbose=False)
            self.models[q] = m
        return self
    
    def predict(self, X):
        X = X[self.feature_cols]
        return pd.DataFrame({
            'score':    self.models[0.50].predict(X),
            'ci_lower': self.models[0.10].predict(X),
            'ci_upper': self.models[0.90].predict(X),
        })
    
    def feature_importance(self):
        fi = self.models[0.50].get_booster().get_score(importance_type='gain')
        return (pd.DataFrame({'feature': list(fi.keys()), 'importance': list(fi.values())})
                .sort_values('importance', ascending=False))


def temporal_cv(df, feature_cols, n_splits=4, target_col='target', year_col=YEAR_COL):
    years   = sorted(df[year_col].unique())
    fold_sz = len(years) // (n_splits + 1)
    maes, r2s, coverages = [], [], []

    for fold in range(n_splits):
        cutoff     = years[(fold + 1) * fold_sz]
        next_cut   = years[min((fold + 2) * fold_sz, len(years) - 1)]
        
        train = df[df[year_col] <  cutoff]
        test  = df[(df[year_col] >= cutoff) & (df[year_col] < next_cut)]
        
        if len(test) < 20:
            continue
        
        X_tr = train[feature_cols].fillna(0)
        y_tr = train[target_col]
        X_te = test[feature_cols].fillna(0)
        y_te = test[target_col]
        
        m = SoftPowerXGB(n_estimators=300)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        
        mae  = mean_absolute_error(y_te, preds['score'])
        r2   = r2_score(y_te, preds['score'])
        # CI coverage: what fraction of true values fall within 80% CI?
        coverage = ((y_te.values >= preds['ci_lower'].values) & 
                    (y_te.values <= preds['ci_upper'].values)).mean()
        
        maes.append(mae)
        r2s.append(r2)
        coverages.append(coverage)
        
        print(f"  Fold {fold+1} | train up to {cutoff} | test {cutoff}–{next_cut} | "
              f"MAE={mae:.3f} | R²={r2:.3f} | CI coverage={coverage:.0%}")
    
    results = {
        'mae_mean':      round(float(np.mean(maes)), 4),
        'mae_std':       round(float(np.std(maes)),  4),
        'r2_mean':       round(float(np.mean(r2s)),  4),
        'r2_std':        round(float(np.std(r2s)),   4),
        'ci_coverage':   round(float(np.mean(coverages)), 4),
        'target_ci_cov': 0.80,   # what we aimed for
    }
    
    print(f"\n  ── CV Summary ──")
    print(f"  MAE  : {results['mae_mean']:.3f} ± {results['mae_std']:.3f}")
    print(f"  R²   : {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
    print(f"  80% CI coverage: {results['ci_coverage']:.0%} "
          f"(target: 80%)")
    
    if results['r2_mean'] > 0.80:
        print("  Assessment: ✓ STRONG predictive power")
    elif results['r2_mean'] > 0.60:
        print("  Assessment: ~ MODERATE — acceptable for soft power (complex domain)")
    else:
        print("  Assessment: ✗ WEAK — consider adding more lag features or KOF data")
    
    return results


cv_results = temporal_cv(model_df, FEATURE_COLS)
with open('output/cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

# ─── 6. TRAIN FINAL MODEL (ALL DATA) ─────────────────────────────────────────

print("\n─── Phase C.5: Training Final Model ───\n")

X_all = model_df[FEATURE_COLS].fillna(0)
y_all = model_df['target']

final_model = SoftPowerXGB(n_estimators=600, lr=0.04)
final_model.fit(X_all, y_all)

with open('output/artifacts/xgb_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("  Final model trained and saved.")

# Feature importance
fi = final_model.feature_importance()
fi.to_csv('output/feature_importance.csv', index=False)
print(f"\n  Top 15 features by importance:")
print(fi.head(15).to_string(index=False))

# ─── 7. GENERATE PREDICTIONS FOR LATEST YEAR ─────────────────────────────────

print("\n─── Phase C.6: Generating Country-Level Predictions ───\n")

latest_year = model_df[YEAR_COL].max()
latest_data = model_df[model_df[YEAR_COL] == latest_year].copy()

preds = final_model.predict(latest_data[FEATURE_COLS].fillna(0))
preds[COUNTRY_COL] = latest_data[COUNTRY_COL].values
preds['canonical'] = latest_data['canonical'].values if 'canonical' in latest_data.columns else preds[COUNTRY_COL]
preds['year']      = latest_year

# Clip to valid range
preds['score']    = preds['score'].clip(0, 100)
preds['ci_lower'] = preds['ci_lower'].clip(0, 100)
preds['ci_upper'] = preds['ci_upper'].clip(0, 100)

# Global rank
preds['global_rank'] = preds['score'].rank(ascending=False, method='min').astype(int)

# Merge regime from trend features
if 'regime' in trend_df.columns:
    preds = preds.merge(
        trend_df[['regime', 'score_volatility', 'influence_growth', 'momentum_5y']].reset_index(),
        on=COUNTRY_COL, how='left'
    )

# Volatility tier
if 'score_volatility' in preds.columns:
    q33 = preds['score_volatility'].quantile(0.33)
    q66 = preds['score_volatility'].quantile(0.66)
    if q33 == q66:
        median_val = preds['score_volatility'].median()
        preds['volatility_tier'] = preds['score_volatility'].apply(
            lambda x: 'stable' if x <= median_val*0.8
            else ('volatile' if x > median_val*1.5 else 'moderate')
        )
    else:
        preds['volatility_tier'] = pd.cut(
            preds['score_volatility'],
            bins=[-np.inf, q33, q66, np.inf],
            labels=['stable', 'moderate', 'volatile']
        )

# CI width as uncertainty measure
preds['ci_width'] = preds['ci_upper'] - preds['ci_lower']

# Sort by rank
preds = preds.sort_values('global_rank')

preds.to_parquet('output/soft_power_predictions.parquet', index=False)
preds.to_csv('output/soft_power_predictions.csv', index=False)

print(f"  Predictions for {len(preds)} countries saved.")
print(f"\n  Top 20 countries:")

display_cols = ['canonical', 'score', 'ci_lower', 'ci_upper', 
                'global_rank', 'regime', 'volatility_tier']
display_cols = [c for c in display_cols if c in preds.columns]
print(preds[display_cols].head(20).to_string(index=False))

# ─── 8. SIMPLE 3-YEAR TREND FORECAST (without BSTS for now) ──────────────────

print("\n─── Phase C.7: 3-Year Trend Forecast ───\n")

def simple_trend_forecast(country_series, horizon=3):
    """
    Linear extrapolation from the last 5 years with uncertainty expansion.
    Use this as your forecast until you set up PyMC for BSTS.
    """
    series = country_series.dropna().tail(8)
    if len(series) < 4:
        return None
    
    x = np.arange(len(series), dtype=float)
    slope, intercept, r, p, se = stats.linregress(x, series.values)
    
    forecasts = []
    for h in range(1, horizon + 1):
        mean_h  = intercept + slope * (len(series) - 1 + h)
        # Uncertainty grows with horizon — conservative estimate
        ci_half = 1.65 * se * np.sqrt(h * 2)
        forecasts.append({
            'horizon':        h,
            'forecast_mean':  round(float(np.clip(mean_h, 0, 100)), 2),
            'forecast_ci_lo': round(float(np.clip(mean_h - ci_half, 0, 100)), 2),
            'forecast_ci_hi': round(float(np.clip(mean_h + ci_half, 0, 100)), 2),
        })
    return forecasts


forecast_records = []
for iso3 in preds[COUNTRY_COL]:
    series = (master[master[COUNTRY_COL]==iso3]
              .set_index(YEAR_COL)[TARGET_COL]
              .sort_index())
    fc = simple_trend_forecast(series, horizon=3)
    if fc:
        for row in fc:
            row[COUNTRY_COL] = iso3
            forecast_records.append(row)

forecast_df = pd.DataFrame(forecast_records)
forecast_df.to_parquet('output/forecasts.parquet', index=False)
print(f"  Forecasts generated for {forecast_df[COUNTRY_COL].nunique()} countries")

# Sample — India forecast
india_fc = forecast_df[forecast_df[COUNTRY_COL]=='IND']
if len(india_fc):
    print(f"\n  India 3-year forecast:")
    print(india_fc[['horizon','forecast_mean','forecast_ci_lo','forecast_ci_hi']].to_string(index=False))

# ─── 9. FINAL SUMMARY ────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"  PHASE C COMPLETE")
print(f"{'═'*60}")
print(f"\n  Model performance (temporal CV):")
print(f"    MAE : {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
print(f"    R²  : {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
print(f"    80% CI actual coverage: {cv_results['ci_coverage']:.0%}")
print(f"\n  Countries predicted : {len(preds)}")
print(f"  Top country         : {preds.iloc[0]['canonical']} "
      f"(score={preds.iloc[0]['score']:.1f})")
print(f"\n  Files saved:")
print(f"    output/soft_power_predictions.parquet  ← main result")
print(f"    output/forecasts.parquet               ← 3-year outlook")
print(f"    output/trend_features.parquet          ← slope analysis")
print(f"    output/feature_importance.csv          ← model explainability")
print(f"    output/cv_results.json                 ← validation metrics")
print(f"    output/artifacts/xgb_model.pkl         ← trained model")
print(f"    output/artifacts/faiss_index.bin       ← similarity index")
print(f"\n  → NEXT STEP:")
print(f"    If R² > 0.75: Run phase3_causal_modeling.py")
print(f"    If R² < 0.65: Run new_dataset_integration.py first, then re-run this")
print(f"{'═'*60}")