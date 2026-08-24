"""
Phase 2: Soft Power Predictive Model
- XGBoost / LightGBM for current-year score prediction (with uncertainty via quantile regression)
- BSTS (Bayesian Structural Time Series) for 5-year forecast per country
- Outputs: score, CI, global rank, temporal trend indicators
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from archive.softpower_models import SoftPowerXGB
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── 1. FEATURE ASSEMBLY ────────────────────────────────────────────────────

def assemble_model_features(master_df: pd.DataFrame,
                             trend_df: pd.DataFrame,
                             indicators: list,
                             country_col='country_iso3',
                             year_col='year',
                             score_col='pca_score') -> pd.DataFrame:
    """
    Merge base indicators + trend features for supervised learning.
    Each row = (country, year) observation.
    Target = pca_score (latent soft power composite).
    """
    # Lag the target by 1 year (predict next year's score from current features)
    master_df = master_df.sort_values([country_col, year_col])
    master_df['target_score'] = master_df.groupby(country_col)[score_col].shift(-1)
    
    # Lag features t-1 to avoid leakage
    for ind in indicators:
        master_df[f'{ind}_lag1'] = master_df.groupby(country_col)[ind].shift(1)
        master_df[f'{ind}_lag2'] = master_df.groupby(country_col)[ind].shift(2)
    
    # Rolling 3-year mean of composite score (momentum proxy)
    master_df['score_roll3'] = (
        master_df.groupby(country_col)[score_col]
        .transform(lambda x: x.rolling(3, min_periods=2).mean())
    )
    
    # Merge trend slope features
    if trend_df is not None:
        slope_cols = [c for c in trend_df.columns if c.endswith('_slope')]
        master_df = master_df.merge(
            trend_df[slope_cols + ['score_volatility', 'influence_growth', 'momentum_5y']],
            on=country_col, how='left'
        )
    
    # Year as a cyclical / numeric feature
    master_df['year_norm'] = (master_df[year_col] - 2000) / 25.0
    
    # Drop rows without a target
    model_df = master_df.dropna(subset=['target_score']).copy()
    
    print(f"[Features] Dataset: {len(model_df)} rows, "
          f"{model_df[country_col].nunique()} countries, "
          f"{model_df[year_col].min()}–{model_df[year_col].max()}")
    return model_df


def get_feature_cols(model_df: pd.DataFrame,
                     indicators: list,
                     exclude=['country_iso3', 'year', 'target_score', 
                              'pca_score', 'regime']) -> list:
    all_cols = list(model_df.columns)
    return [c for c in all_cols if c not in exclude]


# ─── 2. QUANTILE XGBOOST (score + confidence interval) ──────────────────────



# ─── 3. TEMPORAL CV EVALUATION ──────────────────────────────────────────────

def temporal_cross_validate(model_df: pd.DataFrame,
                             feature_cols: list,
                             target_col='target_score',
                             year_col='year',
                             n_splits=5) -> dict:
    """
    Time-series cross-validation: train on past, test on future.
    """
    years = sorted(model_df[year_col].unique())
    fold_size = len(years) // (n_splits + 1)
    
    maes, r2s = [], []
    
    for fold in range(n_splits):
        cutoff_year = years[(fold + 1) * fold_size]
        train = model_df[model_df[year_col] < cutoff_year]
        test  = model_df[model_df[year_col] >= cutoff_year]
        test  = test[test[year_col] < years[min((fold+2)*fold_size, len(years)-1)]]
        
        if len(test) < 10:
            continue
        
        X_train = train[feature_cols].fillna(0)
        y_train = train[target_col]
        X_test  = test[feature_cols].fillna(0)
        y_test  = test[target_col]
        
        model = SoftPowerXGB(n_estimators=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds['score'])
        r2  = r2_score(y_test, preds['score'])
        maes.append(mae); r2s.append(r2)
        print(f"  Fold {fold+1} | cutoff={cutoff_year} | MAE={mae:.3f} | R²={r2:.3f}")
    
    results = {'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
               'r2_mean': np.mean(r2s),   'r2_std': np.std(r2s)}
    print(f"\n[CV] MAE: {results['mae_mean']:.3f} ± {results['mae_std']:.3f}")
    print(f"[CV] R²:  {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
    return results


# ─── 4. BSTS FORECASTING (5-Year) ───────────────────────────────────────────

def bsts_forecast_country(series: pd.Series,
                           horizon: int = 5,
                           n_samples: int = 2000) -> pd.DataFrame:
    """
    Lightweight Bayesian Structural Time Series using PyMC.
    Local level + local trend model.
    Returns forecast with mean and 90% credible interval.
    
    Install: pip install pymc
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        raise ImportError("pip install pymc arviz")
    
    y = series.dropna().values.astype(float)
    T = len(y)
    
    with pm.Model() as model:
        # Priors for noise components
        sigma_obs   = pm.HalfNormal('sigma_obs', sigma=1.0)
        sigma_level = pm.HalfNormal('sigma_level', sigma=0.5)
        sigma_trend = pm.HalfNormal('sigma_trend', sigma=0.1)
        
        # Local level + local trend (Kalman-like via GaussianRandomWalk)
        trend_innov = pm.GaussianRandomWalk('trend_innov', sigma=sigma_trend, shape=T)
        level_innov = pm.GaussianRandomWalk('level_innov', sigma=sigma_level, shape=T)
        
        mu = pm.Deterministic('mu', pm.math.cumsum(trend_innov) + level_innov)
        
        # Likelihood
        obs = pm.Normal('obs', mu=mu, sigma=sigma_obs, observed=y)
        
        # Sample
        trace = pm.sample(n_samples, tune=500, chains=2,
                          progressbar=False, return_inferencedata=True)
    
    # Extrapolate last trend
    last_trend = float(trace.posterior['trend_innov'].values[:, :, -1].mean())
    last_level = float(trace.posterior['mu'].values[:, :, -1].mean())
    trend_std   = float(trace.posterior['trend_innov'].values[:, :, -5:].std())
    level_std   = float(trace.posterior['mu'].values.std())
    
    forecast_rows = []
    for h in range(1, horizon + 1):
        mean_h = last_level + last_trend * h
        ci_lo  = mean_h - 1.65 * np.sqrt(trend_std**2 * h + level_std**2)
        ci_hi  = mean_h + 1.65 * np.sqrt(trend_std**2 * h + level_std**2)
        forecast_rows.append({
            'horizon': h,
            'forecast_mean': float(np.clip(mean_h, 0, 100)),
            'forecast_ci_lo': float(np.clip(ci_lo, 0, 100)),
            'forecast_ci_hi': float(np.clip(ci_hi, 0, 100)),
        })
    
    return pd.DataFrame(forecast_rows)


def forecast_all_countries(master_df, score_col='pca_score',
                            country_col='country_iso3',
                            year_col='year', horizon=5) -> pd.DataFrame:
    """Run BSTS forecast for every country."""
    all_forecasts = []
    countries = master_df[country_col].unique()
    
    for c in countries:
        series = (master_df[master_df[country_col]==c]
                  .set_index(year_col)[score_col]
                  .sort_index())
        if len(series.dropna()) < 8:
            continue
        try:
            fc = bsts_forecast_country(series, horizon=horizon)
            fc[country_col] = c
            all_forecasts.append(fc)
        except Exception as e:
            print(f"[BSTS] {c}: {e}")
    
    return pd.concat(all_forecasts, ignore_index=True)


# ─── 5. RANKING + VOLATILITY CLASSIFICATION ─────────────────────────────────

def compute_output_table(predictions: pd.DataFrame,
                          forecast_df: pd.DataFrame,
                          trend_df: pd.DataFrame,
                          country_col='country_iso3') -> pd.DataFrame:
    """
    Assemble the final output table per country with all required fields.
    """
    out = predictions.copy()
    
    # Global rank
    out['global_rank'] = out['score'].rank(ascending=False, method='min').astype(int)
    
    # Regional rank (requires a region lookup — stub here)
    # out['regional_rank'] = out.groupby('region')['score'].rank(ascending=False)
    
    # Volatility classification
    if trend_df is not None and 'score_volatility' in trend_df.columns:
        out = out.merge(trend_df[['score_volatility', 'regime', 'influence_growth']],
                        on=country_col, how='left')
        vol_q33 = out['score_volatility'].quantile(0.33)
        vol_q66 = out['score_volatility'].quantile(0.66)
        out['volatility_class'] = pd.cut(
            out['score_volatility'],
            bins=[-np.inf, vol_q33, vol_q66, np.inf],
            labels=['low', 'medium', 'high']
        )
    
    # Attach 1-year-ahead forecast
    fc_1y = forecast_df[forecast_df['horizon']==1][[country_col, 'forecast_mean', 
                                                     'forecast_ci_lo', 'forecast_ci_hi']]
    out = out.merge(fc_1y, on=country_col, how='left')
    
    return out.sort_values('global_rank')


# ─── USAGE ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import pickle

    master_df  = pd.read_parquet('data/master_panel.parquet')
    trend_df   = pd.read_parquet('artifacts/trend_features.parquet').reset_index()
    
    INDICATORS = [...]  # your 23 KPI column names
    
    # Assemble features
    model_df = assemble_model_features(master_df, trend_df, INDICATORS)
    feature_cols = get_feature_cols(model_df, INDICATORS)
    
    # Cross-validate
    cv_results = temporal_cross_validate(model_df, feature_cols)
    
    # Train final model on all data
    X = model_df[feature_cols].fillna(0)
    y = model_df['target_score']
    
    final_model = SoftPowerXGB(n_estimators=600)
    final_model.fit(X, y)
    
    # Predict latest year per country
    latest_year = model_df['year'].max()
    latest = model_df[model_df['year']==latest_year].copy()
    preds = final_model.predict(latest[feature_cols].fillna(0))
    preds['country_iso3'] = latest['country_iso3'].values
    
    # 5-year BSTS forecast
    forecast_df = forecast_all_countries(master_df)
    
    # Final output table
    output = compute_output_table(preds, forecast_df, trend_df)
    output.to_parquet('artifacts/soft_power_output.parquet')
    print(output[['country_iso3', 'score', 'ci_lower', 'ci_upper', 
                  'global_rank', 'regime', 'volatility_class']].head(20))
    
    # Save model
    with open('artifacts/xgb_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
