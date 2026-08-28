"""
Phase 4: Continuous Learning Interface
- Detects new annual data
- Validates schema / quality
- Retrains / fine-tunes XGBoost incrementally
- Updates BSTS priors with new observations
- Logs model performance drift over time
- Exposes a simple agent interface for querying results
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path


# Copied (not imported) from archive/phase_c_model_ready.py's module-level
# config -- that module runs a full retrain as a side effect of being
# imported, so its constants are duplicated here rather than pulled in live.
# Keep these in sync if the feature set there ever changes.
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


# ─── 1. DATA INGESTION + VALIDATION ─────────────────────────────────────────

EXPECTED_SCHEMA = {
    'country_iso3': 'object',
    'year':         'int64',
    # add all 23 KPI columns here with expected dtypes
}

def validate_new_data(new_df: pd.DataFrame, indicators: list) -> dict:
    """
    Quality gate for new annual data before retraining.
    Returns a report dict; raises if critical issues found.
    """
    report = {'passed': True, 'warnings': [], 'errors': []}
    
    # Check required columns
    for col in ['country_iso3', 'year'] + indicators:
        if col not in new_df.columns:
            report['errors'].append(f"Missing column: {col}")
    
    # Check country coverage
    n_countries = new_df['country_iso3'].nunique()
    if n_countries < 100:
        report['warnings'].append(f"Only {n_countries} countries — low coverage")
    
    # Check missingness per column
    for col in indicators:
        miss_pct = new_df[col].isna().mean() * 100
        if miss_pct > 50:
            report['warnings'].append(f"{col}: {miss_pct:.0f}% missing")
        if miss_pct > 80:
            report['errors'].append(f"{col}: {miss_pct:.0f}% missing — unacceptable")
    
    # Check for duplicate (country, year) pairs
    dups = new_df.duplicated(['country_iso3', 'year']).sum()
    if dups > 0:
        report['errors'].append(f"{dups} duplicate (country, year) pairs")
    
    if report['errors']:
        report['passed'] = False
    
    return report


# ─── 2. INCREMENTAL RETRAINING ───────────────────────────────────────────────

class ContinuousLearner:
    """
    Wraps the XGBoost model with:
    - Incremental retraining on new data
    - Performance drift monitoring
    - Model versioning
    """
    
    def __init__(self, artifacts_dir='artifacts', model_registry='model_registry'):
        self.artifacts_dir = Path(artifacts_dir)
        self.registry_dir  = Path(model_registry)
        self.registry_dir.mkdir(exist_ok=True)
        self.log_path = self.artifacts_dir / 'training_log.jsonl'
    
    def load_current_model(self):
        with open(self.artifacts_dir / 'xgb_model.pkl', 'rb') as f:
            return pickle.load(f)
    
    def retrain(self, full_df: pd.DataFrame,
                trend_df: pd.DataFrame,
                indicators: list,
                feature_cols: list) -> dict:
        """
        Full retrain on all available historical data + new year.
        XGBoost trains from scratch (preferred over incremental for tree ensembles).
        """
        from archive.phase2_predictive_model import (
            assemble_model_features, SoftPowerXGB, temporal_cross_validate
        )
        
        model_df = assemble_model_features(full_df, trend_df, indicators)
        
        # Validate model_df
        assert len(model_df) > 500, "Insufficient training rows"
        
        # CV before training final model
        cv = temporal_cross_validate(model_df, feature_cols, n_splits=3)
        
        # Check for degradation vs last run
        prev_log = self._load_last_log()
        if prev_log:
            if cv['mae_mean'] > prev_log['mae_mean'] * 1.15:
                print(f"[WARNING] MAE degraded: {prev_log['mae_mean']:.3f} → {cv['mae_mean']:.3f}")
        
        X = model_df[feature_cols].fillna(0)
        y = model_df['target_score']
        
        new_model = SoftPowerXGB(n_estimators=600)
        new_model.fit(X, y)
        
        # Version and save
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.registry_dir / f'xgb_{version}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(new_model, f)
        
        # Overwrite active model
        with open(self.artifacts_dir / 'xgb_model.pkl', 'wb') as f:
            pickle.dump(new_model, f)
        
        log_entry = {
            'version':    version,
            'timestamp':  datetime.now().isoformat(),
            'n_rows':     len(model_df),
            'n_countries': model_df['country_iso3'].nunique(),
            'year_range': [int(model_df['year'].min()), int(model_df['year'].max())],
            **cv
        }
        self._append_log(log_entry)
        print(f"[ContinuousLearner] Retrained model v{version} | MAE={cv['mae_mean']:.3f}")
        return log_entry
    
    def update_bsts_priors(self, new_df: pd.DataFrame, 
                            existing_forecast: pd.DataFrame,
                            indicators: list,
                            score_col='pca_score') -> pd.DataFrame:
        """
        For BSTS: adding new observations and re-forecasting is the update step.
        The Bayesian model naturally incorporates new data by extending the series.
        """
        from archive.phase2_predictive_model import forecast_all_countries
        
        # Re-run BSTS with extended series (includes the new year)
        updated_forecast = forecast_all_countries(new_df, score_col=score_col)
        updated_forecast.to_parquet(self.artifacts_dir / 'soft_power_forecast.parquet')
        print(f"[BSTS] Updated forecasts for {updated_forecast['country_iso3'].nunique()} countries")
        return updated_forecast
    
    def _load_last_log(self):
        if not self.log_path.exists():
            return None
        with open(self.log_path) as f:
            lines = f.readlines()
        if lines:
            return json.loads(lines[-1])
        return None
    
    def _append_log(self, entry: dict):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')


# ─── 3. SOFT POWER AGENT INTERFACE ──────────────────────────────────────────

# NOTE: this class originally pointed at a root-level artifacts/ snapshot
# (soft_power_output.parquet, soft_power_forecast.parquet, ...) that predates
# the current output/ pipeline and no longer exists in that form -- it would
# raise FileNotFoundError on load. Repointed at the files the pipeline
# actually produces today (output/soft_power_predictions.csv,
# output/kalman_forecast_5yr.csv, output/trend_features.parquet,
# output/artifacts/*) and the column names those files actually use
# (iso3, not country_iso3; volatility_tier, not volatility_class; ...).

class SoftPowerAgent:
    """
    High-level query interface for the Soft Power Agent.
    Wraps all pipeline artifacts into simple method calls.
    """

    def __init__(self, output_dir='output'):
        self.dir = Path(output_dir)
        self._load_artifacts()

    def _load_artifacts(self):
        self.output_df = pd.read_csv(self.dir / 'soft_power_predictions.csv')
        self.output_df['iso3'] = self.output_df['iso3'].str.upper()

        self.forecast_df = pd.read_csv(self.dir / 'kalman_forecast_5yr.csv')
        self.forecast_df['iso3'] = self.forecast_df['iso3'].str.upper()

        # trend_features.parquet is indexed by iso3, not a plain column.
        self.trend_df = pd.read_parquet(self.dir / 'trend_features.parquet')
        self.trend_df.index = self.trend_df.index.str.upper()

        with open(self.dir / 'artifacts' / 'xgb_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        import faiss
        self.faiss_index = faiss.read_index(str(self.dir / 'artifacts' / 'faiss_index.bin'))
        self.embed_df = pd.read_parquet(self.dir / 'country_embeddings.parquet')

        with open(self.dir / 'artifacts' / 'embedding_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.embed_matrix = np.load(self.dir / 'artifacts' / 'embedding_matrix.npy')
        import faiss as _f
        _f.normalize_L2(self.embed_matrix.astype('float32'))

        # master_phase_b.parquet is needed to reconstruct model-ready features
        # for what_if() counterfactuals -- loaded lazily, not at init, since
        # get_country_report/global_rankings/regime_clusters don't need it.
        self._master = None

    def get_country_report(self, iso3: str) -> dict:
        """Full soft power report for a single country."""
        iso3 = iso3.upper()
        row = self.output_df[self.output_df['iso3'] == iso3]
        if len(row) == 0:
            return {'error': f'{iso3} not found'}
        row = row.iloc[0]

        forecast = self.forecast_df[self.forecast_df['iso3'] == iso3].sort_values('horizon')

        from archive.phase1_trend_and_similarity import find_peer_nations
        peers = find_peer_nations(iso3, self.embed_df, self.embed_matrix,
                                  self.faiss_index, k=5)

        trend_row = self.trend_df.loc[iso3] if iso3 in self.trend_df.index else None

        return {
            'country':          iso3,
            'latent_score':     round(float(row['score']), 2),
            'ci_lower':         round(float(row['ci_lower']), 2),
            'ci_upper':         round(float(row['ci_upper']), 2),
            'global_rank':      int(row['global_rank']),
            'regime':           str(row.get('regime', 'unknown')),
            'volatility':       str(row.get('volatility_tier', 'unknown')),
            'influence_growth': round(float(row.get('influence_growth', 0)), 4),
            'momentum_5y':      round(float(trend_row['momentum_5y']), 4) if trend_row is not None else None,
            '5y_forecast':      forecast[['horizon', 'forecast_year', 'forecast_score',
                                          'ci_lower_95', 'ci_upper_95']].to_dict('records'),
            'peer_nations':     peers.to_dict('records'),
        }

    def global_rankings(self, top_n=20) -> pd.DataFrame:
        return self.output_df.sort_values('global_rank').head(top_n)[
            ['iso3', 'score', 'ci_lower', 'ci_upper',
             'global_rank', 'regime', 'volatility_tier']
        ]

    def regime_clusters(self) -> pd.DataFrame:
        return self.output_df.groupby('regime').agg(
            n_countries=('iso3', 'count'),
            avg_score=('score', 'mean')
        ).reset_index()

    def _model_ready_row(self, iso3: str) -> pd.DataFrame:
        """
        Rebuild the single most-recent-year, model-ready feature row for one
        country -- the lag1/lag2/rolling/year_norm/trend-slope features the
        trained model was fit on. Mirrors build_model_features() in
        archive/phase_c_model_ready.py, but only for one country's latest row
        and without importing that module (which retrains and overwrites
        output/ as a side effect of being imported -- not safe to trigger
        from a query call).
        """
        if self._master is None:
            self._master = pd.read_parquet(self.dir / 'master_phase_b.parquet')
            if 'country_iso3' in self._master.columns and 'iso3' not in self._master.columns:
                self._master = self._master.rename(columns={'country_iso3': 'iso3'})
            self._master['iso3'] = self._master['iso3'].str.upper()

        cdf = self._master[self._master['iso3'] == iso3].sort_values('year').copy()
        if len(cdf) == 0:
            raise ValueError(f"{iso3} not found in master_phase_b.parquet")

        target_col = 'soft_power_composite_raw'
        indicators = [c for c in TEMPORAL_INDICATORS if c in cdf.columns]
        dim_cols = [c for c in DIM_SCORE_COLS if c in cdf.columns]

        for ind in indicators:
            cdf[f'{ind}_lag1'] = cdf[ind].shift(1)
            cdf[f'{ind}_lag2'] = cdf[ind].shift(2)
        for dim in dim_cols:
            cdf[f'{dim}_lag1'] = cdf[dim].shift(1)
        cdf['score_roll3_mean'] = cdf[target_col].rolling(3, min_periods=2).mean()
        cdf['score_roll3_std'] = cdf[target_col].rolling(3, min_periods=2).std()
        cdf['year_norm'] = (cdf['year'] - 2000) / 24.0

        latest = cdf.iloc[[-1]].copy()

        if iso3 in self.trend_df.index:
            slope_cols = [c for c in self.trend_df.columns
                          if c.endswith('_slope') or c in
                          ('score_volatility', 'influence_growth', 'momentum_5y', 'score_r2')]
            for c in slope_cols:
                latest[c] = self.trend_df.loc[iso3, c]

        return latest.reindex(columns=self.model.feature_cols, fill_value=0).fillna(0)

    def what_if(self, iso3: str, treatment: str, delta: float) -> dict:
        """
        Counterfactual query: what would iso3's score be if `treatment`
        moved by `delta`? Uses the trained XGBoost model directly (see the
        caveats on causal claims in MODEL_TRUST_GUIDE.md section 6 -- this
        is a model-based sensitivity check, not a validated causal estimate).
        """
        iso3 = iso3.upper()
        X_base = self._model_ready_row(iso3)
        X_counter = X_base.copy()

        if treatment in X_counter.columns:
            X_counter[treatment] = X_counter[treatment] + delta
        elif f'{treatment}_lag1' in X_counter.columns:
            X_counter[f'{treatment}_lag1'] = X_counter[f'{treatment}_lag1'] + delta
        else:
            return {'error': f"'{treatment}' is not a feature this model uses"}

        baseline = self.model.predict(X_base)
        counterfactual = self.model.predict(X_counter)

        return {
            'country':       iso3,
            'treatment':     treatment,
            'delta':         delta,
            'score_base':    float(baseline['score'].iloc[0]),
            'score_counter': float(counterfactual['score'].iloc[0]),
            'effect':        float(counterfactual['score'].iloc[0] - baseline['score'].iloc[0]),
        }


# ─── 4. ANNUAL UPDATE PIPELINE (entry point) ────────────────────────────────

def run_annual_update(new_year_data_path: str,
                       master_data_path: str,
                       indicators: list,
                       artifacts_dir='artifacts'):
    """
    Called once per year when new data becomes available.
    1. Validate new data
    2. Append to master panel
    3. Rerun Phase 1 (trends + embeddings)
    4. Retrain Phase 2 model
    5. Update forecasts
    6. Regenerate output table
    """
    print(f"\n{'='*50}")
    print(f"[Annual Update] {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")
    
    # Load new year data
    new_df = pd.read_csv(new_year_data_path)
    
    # Validate
    report = validate_new_data(new_df, indicators)
    print(f"[Validation] {'PASSED' if report['passed'] else 'FAILED'}")
    for w in report['warnings']: print(f"  ⚠ {w}")
    for e in report['errors']:   print(f"  ✗ {e}")
    if not report['passed']:
        raise ValueError("Data validation failed — aborting update")
    
    # Append to master panel
    master_df = pd.read_parquet(master_data_path)
    master_df = pd.concat([master_df, new_df], ignore_index=True)
    master_df = master_df.drop_duplicates(['country_iso3', 'year'])
    master_df.to_parquet(master_data_path)
    print(f"[Master panel] Now {len(master_df)} rows")
    
    # Phase 1
    from archive.phase1_trend_and_similarity import (
        compute_trend_features, build_country_embeddings,
        build_faiss_index, save_phase1_artifacts
    )
    trend_df = compute_trend_features(master_df, indicators)
    embed_df, matrix, scaler = build_country_embeddings(master_df, indicators)
    index = build_faiss_index(matrix)
    save_phase1_artifacts(trend_df, embed_df, matrix, index, scaler, artifacts_dir)
    
    # Phase 2 retrain
    from archive.phase2_predictive_model import assemble_model_features, get_feature_cols
    model_df = assemble_model_features(master_df, trend_df.reset_index(), indicators)
    feature_cols = get_feature_cols(model_df, indicators)
    
    learner = ContinuousLearner(artifacts_dir)
    log = learner.retrain(master_df, trend_df.reset_index(), indicators, feature_cols)
    
    # Update forecasts
    learner.update_bsts_priors(master_df, None, indicators)
    
    print(f"\n[Annual Update] Complete. Model version: {log['version']}")
    return log


# ─── USAGE ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Query the agent
    agent = SoftPowerAgent(artifacts_dir='artifacts')
    
    report = agent.get_country_report('IND')
    print(f"\n{'='*40}")
    print(f"India Soft Power Report")
    print(f"  Score:       {report['latent_score']} [{report['ci_lower']}, {report['ci_upper']}]")
    print(f"  Global Rank: #{report['global_rank']}")
    print(f"  Regime:      {report['regime']}")
    print(f"  Volatility:  {report['volatility']}")
    print(f"\n  5-Year Forecast:")
    for fc in report['5y_forecast']:
        print(f"    +{fc['horizon']}y: {fc['forecast_mean']:.1f} "
              f"[{fc['forecast_ci_lo']:.1f}, {fc['forecast_ci_hi']:.1f}]")
    print(f"\n  Peer Nations: {[p['country'] for p in report['peer_nations']]}")
    
    # Top 20 rankings
    print(f"\n{agent.global_rankings()}")
    
    # Annual update (when 2025 data is available)
    # run_annual_update('data/new_2025.csv', 'data/master_panel.parquet', INDICATORS)
