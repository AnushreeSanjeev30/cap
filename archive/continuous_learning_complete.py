"""
continuous_learning_complete.py
================================
Replaces phase4_continuous_learning.py prototype with a working system.

What this does:
  When new yearly data arrives (e.g., 2024 data drops in Jan 2025),
  instead of retraining from scratch (expensive), this module:

  1. VALIDATES new data quality before touching the model
  2. DETECTS concept drift (has the relationship between KPIs and
     soft power changed significantly?)
  3. INCREMENTALLY UPDATES the XGBoost model using warm-starting
  4. UPDATES the Kalman filter state with the new observation
  5. LOGS the update with a version tag for reproducibility

Usage:
    # One-shot update with new year's data
    python continuous_learning_complete.py --new-data output/new_year_data.parquet

    # Or import and call from your agent:
    from continuous_learning_complete import ContinuousLearner
    learner = ContinuousLearner()
    learner.update(new_df)
"""

import os
import json
import pickle
import warnings
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
MASTER_PARQUET   = 'output/master_phase_b.parquet'
MODEL_PARQUET    = 'output/model_df.parquet'
XGB_PICKLE       = 'output/xgb_model.pkl'
KALMAN_CSV       = 'output/kalman_results.csv'
VERSION_LOG      = 'output/model_versions.json'
MODEL_BACKUP_DIR = 'output/model_backups'

COUNTRY_COL = 'iso3'
YEAR_COL    = 'year'
SCORE_COL   = 'soft_power_composite_raw'

INDICATORS = [
    'tourist_arrivals', 'unesco_total_sites', 'unesco_cultural_sites',
    'internet_pct_sp', 'rnd_pct_gdp_sp', 'sci_journal_articles',
    'ai_publications', 'hightech_exports_pct', 'fh_combined_score',
    'trade_pct_gdp', 'investment_freedom', 'govt_integrity',
    'judicial_effectiveness', 'property_rights', 'business_freedom',
    'life_expectancy', 'tertiary_enroll_pct', 'physicians_per_1k',
    'infant_mortality'
]

# Drift detection thresholds
DRIFT_PSI_THRESHOLD = 0.2       # Population Stability Index > 0.2 = major drift
DRIFT_CORR_THRESHOLD = 0.15     # Correlation shift > 0.15 = drift detected
MIN_COUNTRIES_FOR_UPDATE = 50   # Don't update if fewer countries in new data
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class DataValidator:
    """Validates new data before model update."""

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df
        self.feature_cols = [c for c in INDICATORS if c in reference_df.columns]

    def validate(self, new_df: pd.DataFrame) -> dict:
        """
        Run all validation checks. Returns a report dict with
        'passed' (bool) and 'issues' (list of strings).
        """
        issues = []
        warnings_list = []

        # 1. Country coverage
        ref_countries  = set(self.reference_df[COUNTRY_COL].unique())
        new_countries  = set(new_df[COUNTRY_COL].unique()) if COUNTRY_COL in new_df.columns else set()
        n_new          = len(new_countries)
        coverage_ratio = len(new_countries & ref_countries) / max(len(ref_countries), 1)

        if n_new < MIN_COUNTRIES_FOR_UPDATE:
            issues.append(f"Too few countries: {n_new} (need ≥{MIN_COUNTRIES_FOR_UPDATE})")
        if coverage_ratio < 0.7:
            warnings_list.append(f"Low country overlap: {coverage_ratio:.0%} vs reference")

        # 2. Column presence
        missing_cols = [c for c in self.feature_cols if c not in new_df.columns]
        if len(missing_cols) > 5:
            issues.append(f"Too many missing columns: {missing_cols}")
        elif missing_cols:
            warnings_list.append(f"Missing columns (will impute): {missing_cols}")

        # 3. Missing rate per column
        present_cols = [c for c in self.feature_cols if c in new_df.columns]
        for col in present_cols:
            miss_rate = new_df[col].isna().mean()
            if miss_rate > 0.5:
                warnings_list.append(f"{col}: {miss_rate:.0%} missing in new data")

        # 4. Range sanity (new data shouldn't be wildly outside historical range)
        for col in present_cols[:5]:   # spot check 5 columns
            ref_min = self.reference_df[col].quantile(0.01)
            ref_max = self.reference_df[col].quantile(0.99)
            new_min = new_df[col].min()
            new_max = new_df[col].max()
            if new_min < ref_min * 0.1 or new_max > ref_max * 10:
                warnings_list.append(
                    f"{col}: extreme values detected "
                    f"(ref=[{ref_min:.2f},{ref_max:.2f}], new=[{new_min:.2f},{new_max:.2f}])"
                )

        passed = len(issues) == 0
        return {
            'passed':        passed,
            'issues':        issues,
            'warnings':      warnings_list,
            'n_countries':   n_new,
            'coverage_ratio': coverage_ratio,
            'missing_cols':  missing_cols,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1 → no change
    PSI 0.1–0.2 → slight change (monitor)
    PSI > 0.2 → major drift (retrain)
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    ref_pct = np.histogram(ref_clean, bins=breakpoints)[0] / len(ref_clean)
    cur_pct = np.histogram(cur_clean, bins=breakpoints)[0] / len(cur_clean)

    # Avoid division by zero
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(reference_df: pd.DataFrame,
                 new_df: pd.DataFrame,
                 feature_cols: list) -> dict:
    """
    Detect concept/data drift between reference and new data.
    Returns per-feature PSI and overall drift verdict.
    """
    psi_scores   = {}
    corr_shifts  = {}
    drifted_kpis = []

    for col in feature_cols:
        if col not in new_df.columns or col not in reference_df.columns:
            continue

        ref_vals = reference_df[col].dropna().values
        new_vals = new_df[col].dropna().values

        psi = compute_psi(ref_vals, new_vals)
        psi_scores[col] = round(psi, 4)
        if psi > DRIFT_PSI_THRESHOLD:
            drifted_kpis.append(col)

    # Correlation shift: check if KPI→score correlations have changed
    if SCORE_COL in reference_df.columns and SCORE_COL in new_df.columns:
        for col in feature_cols[:10]:   # spot check
            if col not in new_df.columns:
                continue
            try:
                ref_corr = reference_df[[col, SCORE_COL]].dropna().corr().iloc[0, 1]
                new_corr = new_df[[col, SCORE_COL]].dropna().corr().iloc[0, 1]
                shift = abs(ref_corr - new_corr)
                corr_shifts[col] = round(float(shift), 4)
            except Exception:
                pass

    drift_detected = len(drifted_kpis) > len(feature_cols) * 0.3  # >30% drifted

    print(f"\n  DRIFT DETECTION:")
    print(f"  Drifted KPIs (PSI > {DRIFT_PSI_THRESHOLD}): {len(drifted_kpis)} / {len(feature_cols)}")
    if drifted_kpis:
        print(f"  Drifted: {drifted_kpis[:5]}{'...' if len(drifted_kpis)>5 else ''}")
    print(f"  Overall drift: {'⚠️  YES — full retrain recommended' if drift_detected else '✅ No significant drift'}")

    return {
        'drift_detected':   drift_detected,
        'drifted_kpis':     drifted_kpis,
        'n_drifted':        len(drifted_kpis),
        'psi_scores':       psi_scores,
        'corr_shifts':      corr_shifts,
        'recommendation':   'full_retrain' if drift_detected else 'incremental_update',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL MODEL UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

def incremental_update_xgb(xgb_model,
                            new_df: pd.DataFrame,
                            additional_estimators: int = 50) -> object:
    """
    Warm-start update: adds `additional_estimators` new trees to each
    quantile model using only the new data. Much faster than full retrain.

    XGBoost supports this via xgb_model parameter in fit().
    """
    from xgboost import XGBRegressor

    feature_cols = xgb_model.feature_cols
    present_cols = [c for c in feature_cols if c in new_df.columns]

    X_new = new_df[present_cols].copy()
    # Impute missing with medians
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0.0
        else:
            X_new[col] = X_new[col].fillna(X_new[col].median())
    X_new = X_new[feature_cols]

    if SCORE_COL not in new_df.columns:
        print("  ⚠️  No target column in new data — skipping model update.")
        return xgb_model

    y_new = new_df[SCORE_COL].fillna(new_df[SCORE_COL].median())

    print(f"  Warm-starting with {len(X_new)} new samples, "
          f"+{additional_estimators} trees per quantile...")

    for q in xgb_model.QUANTILES:
        existing = xgb_model.models[q]
        params   = xgb_model.params.copy()
        params['n_estimators'] = additional_estimators

        updater = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            **params
        )
        updater.fit(
            X_new, y_new,
            xgb_model=existing.get_booster(),   # warm start
            verbose=False
        )
        xgb_model.models[q] = updater

    print(f"  ✅ Model updated with {additional_estimators} additional trees.")
    return xgb_model


def update_kalman_state(kalman_results_path: str,
                        new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update Kalman filter with new year's observations.
    Appends new rows to kalman_results.csv using the last state as prior.
    """
    from archive.kalman_softpower_complete import kalman_filter_country, optimize_kalman_params

    if not os.path.exists(kalman_results_path):
        print(f"  ⚠️  {kalman_results_path} not found — skipping Kalman update.")
        return None

    kalman_df = pd.read_csv(kalman_results_path)
    new_year  = int(new_df[YEAR_COL].max())

    new_rows = []

    for iso in new_df[COUNTRY_COL].unique():
        # Get historical state for this country
        hist = kalman_df[kalman_df[COUNTRY_COL] == iso].sort_values(YEAR_COL)
        if len(hist) == 0:
            continue

        last  = hist.iloc[-1]
        x_t   = float(last['kalman_score'])
        P_t   = float(last['kalman_std']) ** 2 if pd.notna(last['kalman_std']) else 1.0

        # New observation
        new_obs_row = new_df[new_df[COUNTRY_COL] == iso]
        if len(new_obs_row) == 0 or SCORE_COL not in new_obs_row.columns:
            continue

        y_new = float(new_obs_row[SCORE_COL].iloc[0])
        if np.isnan(y_new):
            continue

        # Get Q, R from historical series
        hist_scores = hist['raw_score'].values.astype(float)
        Q, R = optimize_kalman_params(hist_scores)

        # One Kalman update step
        P_pred     = P_t + Q
        K          = P_pred / (P_pred + R)
        innovation = y_new - x_t
        x_new      = x_t + K * innovation
        P_new      = (1 - K) * P_pred
        std_new    = np.sqrt(P_new)

        new_rows.append({
            COUNTRY_COL:     iso,
            YEAR_COL:        new_year,
            'raw_score':     y_new,
            'kalman_score':  round(x_new, 4),
            'kalman_std':    round(std_new, 4),
            'ci_lower_95':   round(x_new - 1.96 * std_new, 4),
            'ci_upper_95':   round(x_new + 1.96 * std_new, 4),
            'ci_width':      round(2 * 1.96 * std_new, 4),
            'innovation':    round(float(innovation), 4),
        })

    if new_rows:
        updated = pd.concat([kalman_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(kalman_results_path, index=False)
        print(f"  ✅ Kalman updated: +{len(new_rows)} rows for year {new_year}")
        return updated
    else:
        print(f"  ⚠️  No Kalman rows added (check SCORE_COL in new data)")
        return kalman_df


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_version(event: str, metadata: dict):
    """Append an entry to model_versions.json for reproducibility."""
    os.makedirs('output', exist_ok=True)

    if os.path.exists(VERSION_LOG):
        with open(VERSION_LOG) as f:
            log = json.load(f)
    else:
        log = []

    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'event':     event,
        **metadata
    }
    log.append(entry)

    with open(VERSION_LOG, 'w') as f:
        json.dump(log, f, indent=2, default=str)

    print(f"  📝 Version logged: {event}")


def backup_model(xgb_model):
    """Save a timestamped model backup before updating."""
    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    ts   = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(MODEL_BACKUP_DIR, f'xgb_model_{ts}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"  💾 Model backed up to {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS — use this from your agent
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousLearner:
    """
    Main interface for continuous learning updates.

    from continuous_learning_complete import ContinuousLearner
    learner = ContinuousLearner()
    report  = learner.update(new_year_df)
    """

    def __init__(self):
        self.reference_df = self._load_reference()
        self.xgb_model    = self._load_model()
        self.feature_cols = [c for c in INDICATORS if c in self.reference_df.columns]

    def _load_reference(self):
        for path in [MODEL_PARQUET, MASTER_PARQUET]:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if 'country_iso3' in df.columns and COUNTRY_COL not in df.columns:
                    df = df.rename(columns={'country_iso3': COUNTRY_COL})
                print(f"  Reference data loaded: {len(df)} rows from {path}")
                return df
        raise FileNotFoundError("No reference data found. Run phase2 first.")

    def _load_model(self):
        if os.path.exists(XGB_PICKLE):
            with open(XGB_PICKLE, 'rb') as f:
                model = pickle.load(f)
            print(f"  XGB model loaded from {XGB_PICKLE}")
            return model
        print(f"  ⚠️  No model at {XGB_PICKLE}")
        return None

    def update(self, new_df: pd.DataFrame,
               force_full_retrain: bool = False) -> dict:
        """
        Full update pipeline. Returns a report dict.

        Args:
            new_df:              DataFrame with new year's data (same schema as reference)
            force_full_retrain:  If True, skip incremental update, do full retrain
        """
        print("\n" + "=" * 60)
        print("CONTINUOUS LEARNING UPDATE")
        print("=" * 60)

        if 'country_iso3' in new_df.columns and COUNTRY_COL not in new_df.columns:
            new_df = new_df.rename(columns={'country_iso3': COUNTRY_COL})

        report = {'timestamp': datetime.utcnow().isoformat()}

        # Step 1: Validate
        print("\n[1/4] Data Validation")
        validator = DataValidator(self.reference_df)
        validation = validator.validate(new_df)
        report['validation'] = validation

        if not validation['passed']:
            print(f"  ❌ Validation FAILED: {validation['issues']}")
            report['status'] = 'failed_validation'
            return report

        print(f"  ✅ Validation passed ({validation['n_countries']} countries, "
              f"{validation['coverage_ratio']:.0%} coverage)")
        if validation['warnings']:
            for w in validation['warnings']:
                print(f"  ⚠️  {w}")

        # Step 2: Drift detection
        print("\n[2/4] Drift Detection")
        drift = detect_drift(self.reference_df, new_df, self.feature_cols)
        report['drift'] = drift

        # Step 3: Model update
        print("\n[3/4] Model Update")
        if self.xgb_model is None:
            print("  Skipped — no model loaded.")
            report['model_updated'] = False
        elif force_full_retrain or drift['drift_detected']:
            print("  Full retrain recommended. Run phase2_predictive_model.py manually.")
            print("  (Full retrain not done here to protect your existing model.)")
            report['model_updated'] = False
            report['retrain_needed'] = True
        else:
            backup_path = backup_model(self.xgb_model)
            self.xgb_model = incremental_update_xgb(self.xgb_model, new_df)

            # Save updated model
            with open(XGB_PICKLE, 'wb') as f:
                pickle.dump(self.xgb_model, f)
            print(f"  ✅ Updated model saved to {XGB_PICKLE}")
            report['model_updated'] = True
            report['backup_path']   = backup_path

        # Step 4: Kalman update
        print("\n[4/4] Kalman State Update")
        update_kalman_state(KALMAN_CSV, new_df)

        # Append new data to reference
        updated_ref = pd.concat([self.reference_df, new_df], ignore_index=True)
        updated_ref.drop_duplicates(
            subset=[COUNTRY_COL, YEAR_COL], keep='last'
        ).to_parquet(MASTER_PARQUET)
        print(f"  ✅ Reference data updated (+{len(new_df)} rows)")

        # Log version
        log_version('incremental_update', {
            'n_new_rows':     len(new_df),
            'n_countries':    validation['n_countries'],
            'drift_detected': drift['drift_detected'],
            'n_drifted_kpis': drift['n_drifted'],
            'model_updated':  report.get('model_updated', False),
        })

        report['status'] = 'success'
        print("\n✅ Update complete.")
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (CLI usage)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Continuous learning update')
    parser.add_argument('--new-data', required=True,
                        help='Path to new year parquet file')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force full retrain instead of incremental update')
    args = parser.parse_args()

    if not os.path.exists(args.new_data):
        raise FileNotFoundError(f"New data file not found: {args.new_data}")

    new_df = pd.read_parquet(args.new_data)
    print(f"New data: {len(new_df)} rows, year={new_df[YEAR_COL].max() if YEAR_COL in new_df else '?'}")

    learner = ContinuousLearner()
    report  = learner.update(new_df, force_full_retrain=args.force_retrain)

    print("\nFINAL REPORT:")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('drift',)}, indent=2, default=str))
