"""
kalman_softpower_complete.py
============================
Completes the Kalman Filter pipeline with:
  1. Existing 1-step Kalman filter (your code, kept intact)
  2. Multi-step ahead forecasting (5-year horizon per country)
  3. Adaptive Q/R parameter optimization via log-likelihood
  4. Rauch-Tung-Striebel (RTS) smoother for better historical estimates
  5. Regime detection (rising / stable / declining / volatile)

Outputs:
  output/kalman_results.csv          ← per country-year: smoothed score + CI  (unchanged)
  output/kalman_summary.csv          ← per country: stability, trend            (unchanged)
  output/delta_forecast_fixed.csv    ← 1-step forecast                          (unchanged)
  output/kalman_forecast_5yr.csv     ← NEW: 5-year forecast per country
  output/kalman_regimes.csv          ← NEW: regime classification per country
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Windows consoles often default to cp1252, which chokes on characters like
# 'Δ' and '→' used in this script's console output below.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# ── CONFIG (same as your original) ───────────────────────────────────────────
MASTER_PARQUET = 'output/master_phase_b.parquet'
COUNTRY_COL    = 'iso3'
YEAR_COL       = 'year'
SCORE_COL      = 'soft_power_composite_raw'

INDICATORS = [
    'tourist_arrivals', 'unesco_total_sites', 'unesco_cultural_sites',
    'internet_pct_sp', 'rnd_pct_gdp_sp', 'sci_journal_articles',
    'ai_publications', 'hightech_exports_pct', 'fh_combined_score',
    'trade_pct_gdp', 'investment_freedom', 'govt_integrity',
    'judicial_effectiveness', 'property_rights', 'business_freedom',
    'life_expectancy', 'tertiary_enroll_pct', 'physicians_per_1k',
    'infant_mortality'
]
FORECAST_HORIZON = 5   # years ahead to forecast
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: YOUR ORIGINAL KALMAN FILTER (unchanged, just wrapped cleanly)
# ═══════════════════════════════════════════════════════════════════════════════

def kalman_filter_country(scores: np.ndarray, Q: float = None, R: float = None):
    """
    1D Kalman Filter. Returns (filtered_means, filtered_vars, innovations).

    Q/R default to the fixed 30%/10%-of-variance heuristic if not given.
    Callers that already have per-country fitted values (see
    optimize_kalman_params) should pass them in explicitly -- the fixed
    heuristic leaves innovations autocorrelated for ~a third of countries
    (see trust_report.md section 6), because 0.3/0.1 isn't the right
    noise split for every country's dynamics.
    """
    n = len(scores)
    valid = scores[~np.isnan(scores)]
    if len(valid) < 3:
        return None

    diffs = np.diff(valid)
    if Q is None:
        Q = np.var(diffs) * 0.3
    if R is None:
        R = np.var(valid) * 0.1

    x = valid[0]
    P = np.var(valid)

    filtered_means = np.full(n, np.nan)
    filtered_vars  = np.full(n, np.nan)
    innovations    = np.full(n, np.nan)

    for t in range(n):
        y = scores[t]
        if np.isnan(y):
            x = x
            P = P + Q
        else:
            x_pred     = x
            P_pred     = P + Q
            K          = P_pred / (P_pred + R)
            innovation = y - x_pred
            x          = x_pred + K * innovation
            P          = (1 - K) * P_pred
            innovations[t] = innovation

        filtered_means[t] = x
        filtered_vars[t]  = P

    return filtered_means, filtered_vars, innovations


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: OPTIMIZED Q/R PARAMETERS (new)
# ═══════════════════════════════════════════════════════════════════════════════

def _negative_log_likelihood(log_q_ratio: float, scores: np.ndarray) -> float:
    """
    Compute negative log-likelihood of the Kalman model.
    We optimize over the ratio Q/R (in log space for numerical stability).
    Used internally by optimize_kalman_params.
    """
    valid = scores[~np.isnan(scores)]
    if len(valid) < 3:
        return 1e9

    q_ratio = np.exp(log_q_ratio)
    R = np.var(valid) * 0.1
    Q = R * q_ratio

    x = valid[0]
    P = np.var(valid)
    nll = 0.0

    for y in scores:
        P_pred = P + Q
        if not np.isnan(y):
            S = P_pred + R          # innovation variance
            innovation = y - x
            nll += 0.5 * (np.log(2 * np.pi * S) + innovation**2 / S)
            K  = P_pred / S
            x  = x + K * innovation
            P  = (1 - K) * P_pred
        else:
            x = x
            P = P_pred

    return nll


def optimize_kalman_params(scores: np.ndarray):
    """
    Find optimal Q (process noise) via scalar minimization of log-likelihood.
    Returns (Q_opt, R_opt).
    Falls back to heuristic if optimization fails.
    """
    valid = scores[~np.isnan(scores)]
    if len(valid) < 4:
        diffs = np.diff(valid)
        return np.var(diffs) * 0.3, np.var(valid) * 0.1

    try:
        result = minimize_scalar(
            _negative_log_likelihood,
            args=(scores,),
            bounds=(-6, 4),
            method='bounded'
        )
        q_ratio = np.exp(result.x)
        R = np.var(valid) * 0.1
        Q = R * q_ratio
        return Q, R
    except Exception:
        diffs = np.diff(valid)
        return np.var(diffs) * 0.3, np.var(valid) * 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: RTS SMOOTHER (new — better historical estimates)
# ═══════════════════════════════════════════════════════════════════════════════

def rts_smoother(filtered_means: np.ndarray,
                 filtered_vars: np.ndarray,
                 Q: float) -> tuple:
    """
    Rauch-Tung-Striebel backward smoother.
    Takes the Kalman forward-pass outputs and runs a backward pass,
    producing smoother estimates (especially for early years).

    This is what makes historical soft power scores look clean rather
    than jagged — important for your trend visualizations.
    """
    n = len(filtered_means)
    smoothed_means = filtered_means.copy()
    smoothed_vars  = filtered_vars.copy()

    for t in range(n - 2, -1, -1):
        if np.isnan(filtered_vars[t]) or np.isnan(filtered_vars[t + 1]):
            continue
        P_pred = filtered_vars[t] + Q
        if P_pred < 1e-10:
            continue
        G = filtered_vars[t] / P_pred          # smoother gain
        smoothed_means[t] = (filtered_means[t]
                             + G * (smoothed_means[t + 1] - filtered_means[t]))
        smoothed_vars[t]  = (filtered_vars[t]
                             + G**2 * (smoothed_vars[t + 1] - P_pred))

    return smoothed_means, smoothed_vars


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: MULTI-STEP AHEAD FORECASTING (new)
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_country(iso: str,
                     scores: np.ndarray,
                     years: np.ndarray,
                     horizon: int = FORECAST_HORIZON) -> list:
    """
    Produce H-step ahead forecasts for a single country.

    In a random-walk state-space model:
        E[x_{t+h} | data] = x_t   (mean doesn't drift)
        Var[x_{t+h} | data] = P_t + h * Q   (uncertainty grows with horizon)

    Returns a list of dicts (one per forecast year).
    """
    Q, R = optimize_kalman_params(scores)
    out = kalman_filter_country(scores, Q=Q, R=R)
    if out is None:
        return []

    filtered_means, filtered_vars, _ = out

    # Use RTS smoother for better state estimate
    smoothed_means, smoothed_vars = rts_smoother(filtered_means, filtered_vars, Q)

    # Final state = last non-nan smoothed estimate
    valid_idx = np.where(~np.isnan(smoothed_means))[0]
    if len(valid_idx) == 0:
        return []

    last_idx   = valid_idx[-1]
    x_last     = smoothed_means[last_idx]
    P_last     = smoothed_vars[last_idx]
    last_year  = years[last_idx]

    # Trend from last 5 years of smoothed scores
    trend_window = min(5, len(valid_idx))
    recent_idx   = valid_idx[-trend_window:]
    recent_yrs   = years[recent_idx].astype(float)
    recent_sc    = smoothed_means[recent_idx]
    if len(recent_yrs) >= 2:
        slope = np.polyfit(recent_yrs, recent_sc, 1)[0]
    else:
        slope = 0.0

    rows = []
    for h in range(1, horizon + 1):
        # Mean: carry last state forward + trend
        x_h   = x_last + slope * h
        # Variance grows with horizon
        P_h   = P_last + h * Q
        std_h = np.sqrt(max(P_h, 0))

        rows.append({
            COUNTRY_COL:         iso,
            'base_year':         int(last_year),
            'forecast_year':     int(last_year + h),
            'horizon':           h,
            'forecast_score':    round(float(x_h), 4),
            'ci_lower_80':       round(float(x_h - 1.28 * std_h), 4),
            'ci_upper_80':       round(float(x_h + 1.28 * std_h), 4),
            'ci_lower_95':       round(float(x_h - 1.96 * std_h), 4),
            'ci_upper_95':       round(float(x_h + 1.96 * std_h), 4),
            'trend_slope':       round(float(slope), 5),
            'forecast_std':      round(float(std_h), 4),
        })

    return rows


def run_5yr_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Run multi-step forecast for all countries. Returns long-format DataFrame."""
    all_rows = []
    countries = df[COUNTRY_COL].unique()
    print(f"  Forecasting {len(countries)} countries × {FORECAST_HORIZON} years...")

    for iso in countries:
        cdf    = df[df[COUNTRY_COL] == iso].sort_values(YEAR_COL)
        scores = cdf[SCORE_COL].values
        years  = cdf[YEAR_COL].values
        rows   = forecast_country(iso, scores, years)
        all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    print(f"  Done. {len(result)} forecast rows ({len(countries)} countries × {FORECAST_HORIZON} years).")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: REGIME DETECTION (new)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regimes(df: pd.DataFrame, kalman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each country into a trajectory regime using:
      - Trend slope (rising / declining)
      - Volatility (innovation std)
      - Score level (high / mid / low)
      - Acceleration (is the trend speeding up or slowing?)

    Regime labels:
      'Ascendant'       — rising + stable
      'Volatile Riser'  — rising + volatile
      'Stable Power'    — flat + high score + stable
      'Coasting'        — flat + mid score
      'Declining'       — falling slope + low volatility
      'Fragile'         — falling + volatile

    NOTE: Thresholds are computed from the actual data distribution
    (percentiles), not hardcoded — so they work regardless of score scale.
    """
    summary_rows = []

    for iso, grp in kalman_df.groupby(COUNTRY_COL):
        grp = grp.sort_values(YEAR_COL)
        ks  = grp['kalman_score'].dropna().values
        yrs = grp[YEAR_COL].values[:len(ks)]

        if len(ks) < 3:
            continue

        slope = np.polyfit(yrs.astype(float), ks, 1)[0]

        if len(ks) >= 5:
            coeffs = np.polyfit(yrs.astype(float), ks, 2)
            accel  = coeffs[0] * 2
        else:
            accel = 0.0

        innov = grp['innovation'].dropna()
        vol   = innov.std() if len(innov) > 1 else 0.0

        latest_score = ks[-1]

        summary_rows.append({
            COUNTRY_COL:      iso,
            'latest_score':   round(float(latest_score), 3),
            'trend_slope':    round(float(slope), 5),
            'acceleration':   round(float(accel), 6),
            'volatility':     round(float(vol), 4),
            'n_years':        len(ks),
        })

    regime_df = pd.DataFrame(summary_rows)

    # ── Data-relative thresholds (percentile-based, scale-invariant) ──────────
    slope_p33   = regime_df['trend_slope'].quantile(0.33)
    slope_p66   = regime_df['trend_slope'].quantile(0.66)
    vol_median  = regime_df['volatility'].median()
    score_p66   = regime_df['latest_score'].quantile(0.66)

    print(f"\n  Regime thresholds (data-driven):")
    print(f"    Slope  — low < {slope_p33:+.3f} | flat [{slope_p33:+.3f}, {slope_p66:+.3f}] | rising > {slope_p66:+.3f}")
    print(f"    Vol    — low/high split at median = {vol_median:.3f}")
    print(f"    Score  — high > {score_p66:.1f} (top 33%)")

    def assign_regime(row):
        s   = row['trend_slope']
        v   = row['volatility']
        sc  = row['latest_score']
        rising   = s > slope_p66
        falling  = s < slope_p33
        flat     = not rising and not falling
        volatile = v > vol_median
        high_sc  = sc >= score_p66

        if rising and not volatile:
            return 'Ascendant'
        elif rising and volatile:
            return 'Volatile Riser'
        elif flat and high_sc:
            return 'Stable Power'
        elif flat and not high_sc:
            return 'Coasting'
        elif falling and volatile:
            return 'Fragile'
        else:
            return 'Declining'

    regime_df['regime'] = regime_df.apply(assign_regime, axis=1)

    print("\nREGIME DISTRIBUTION:")
    dist = regime_df['regime'].value_counts()
    for regime, count in dist.items():
        pct = count / len(regime_df) * 100
        bar = '█' * int(pct / 2)
        print(f"  {regime:<18} {bar:<25} {count:>4} ({pct:.0f}%)")

    return regime_df.sort_values('latest_score', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR ORIGINAL FUNCTIONS (unchanged — keeping backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def run_kalman_all_countries(df: pd.DataFrame):
    """Your original function — now uses RTS smoother on top."""
    results = []
    countries = df[COUNTRY_COL].unique()
    print(f"  Running Kalman Filter + RTS Smoother on {len(countries)} countries...")

    for iso in countries:
        cdf    = df[df[COUNTRY_COL] == iso].sort_values(YEAR_COL)
        scores = cdf[SCORE_COL].values
        years  = cdf[YEAR_COL].values

        Q, R = optimize_kalman_params(scores)
        out = kalman_filter_country(scores, Q=Q, R=R)
        if out is None:
            continue

        filtered_means, filtered_vars, innovations = out
        smoothed_means, smoothed_vars = rts_smoother(filtered_means, filtered_vars, Q)

        for i, yr in enumerate(years):
            std = np.sqrt(smoothed_vars[i]) if not np.isnan(smoothed_vars[i]) else np.nan
            sm  = smoothed_means[i]
            results.append({
                COUNTRY_COL:     iso,
                YEAR_COL:        yr,
                'raw_score':     scores[i],
                'kalman_score':  sm,
                'kalman_std':    std,
                'ci_lower_95':   sm - 1.96 * std if not np.isnan(std) else np.nan,
                'ci_upper_95':   sm + 1.96 * std if not np.isnan(std) else np.nan,
                'ci_width':      2 * 1.96 * std  if not np.isnan(std) else np.nan,
                'innovation':    innovations[i],
            })

    return pd.DataFrame(results)


def kalman_summary(kalman_df: pd.DataFrame):
    """Your original summary function — unchanged."""
    summary = []
    for iso, grp in kalman_df.groupby(COUNTRY_COL):
        mean_ci      = grp['ci_width'].mean()
        innov_std    = grp['innovation'].std()
        mean_score   = grp['kalman_score'].mean()
        latest_score = grp.loc[grp[YEAR_COL].idxmax(), 'kalman_score']
        latest_ci    = grp.loc[grp[YEAR_COL].idxmax(), 'ci_width']

        yrs = grp[YEAR_COL].values
        ksc = grp['kalman_score'].values
        slope = np.polyfit(yrs, ksc, 1)[0] if len(yrs) > 2 else 0.0

        summary.append({
            COUNTRY_COL:      iso,
            'mean_score':     mean_score,
            'latest_score':   latest_score,
            'latest_ci':      latest_ci,
            'mean_ci_width':  mean_ci,
            'innovation_std': innov_std,
            'trend_slope':    slope,
        })

    sdf = pd.DataFrame(summary)
    q33 = sdf['innovation_std'].quantile(0.33)
    q66 = sdf['innovation_std'].quantile(0.66)
    sdf['stability_class'] = pd.cut(
        sdf['innovation_std'],
        bins=[-np.inf, q33, q66, np.inf],
        labels=['Stable', 'Moderate', 'Volatile']
    )
    return sdf.sort_values('latest_score', ascending=False)


def fixed_forecast(df: pd.DataFrame, kalman_df: pd.DataFrame):
    """Your original 1-step forecast — unchanged."""
    latest = (kalman_df.sort_values(YEAR_COL)
              .groupby(COUNTRY_COL).last().reset_index())
    latest_raw = (df.sort_values(YEAR_COL)
                  .groupby(COUNTRY_COL).last()[[SCORE_COL]].reset_index())
    forecast = latest.merge(latest_raw, on=COUNTRY_COL, suffixes=('', '_raw'))
    forecast['predicted_next_score'] = forecast['kalman_score']
    forecast['predicted_delta']      = forecast['kalman_score'] - forecast['raw_score']
    forecast['forecast_year']        = forecast[YEAR_COL] + 1
    return forecast.sort_values('predicted_delta', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(MASTER_PARQUET)

    if 'country_iso3' in df.columns and COUNTRY_COL not in df.columns:
        df = df.rename(columns={'country_iso3': COUNTRY_COL})

    print(f"Dataset: {len(df)} rows, {df[COUNTRY_COL].nunique()} countries")
    print(f"Years:   {df[YEAR_COL].min()} – {df[YEAR_COL].max()}\n")

    # ── Step 1: Kalman + RTS Smoother ────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: KALMAN FILTER + RTS SMOOTHER")
    print("=" * 60)
    kalman_df = run_kalman_all_countries(df)
    print(f"  Done. {len(kalman_df)} country-year estimates.\n")

    # ── Step 2: Summary + Stability ──────────────────────────────────────────
    summary_df = kalman_summary(kalman_df)

    stable_n   = (summary_df['stability_class'] == 'Stable').sum()
    moderate_n = (summary_df['stability_class'] == 'Moderate').sum()
    volatile_n = (summary_df['stability_class'] == 'Volatile').sum()
    print(f"STABILITY: Stable={stable_n}  Moderate={moderate_n}  Volatile={volatile_n}\n")

    print(f"{'Country':<8} {'Score':>8} {'CI±':>7} {'Trend/yr':>10} {'Class':<12}")
    print("-" * 50)
    for _, row in summary_df.head(10).iterrows():
        print(f"{row[COUNTRY_COL]:<8} {row['latest_score']:>8.2f} "
              f"{row['latest_ci']/2:>7.2f} {row['trend_slope']:>+10.3f} "
              f"{str(row['stability_class']):<12}")

    # ── Step 3: 1-step forecast (your original) ───────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: 1-STEP FORECAST (Kalman carry-forward)")
    print(f"{'='*60}")
    forecast_df = fixed_forecast(df, kalman_df)
    print(f"  {len(forecast_df)} countries forecasted.\n")

    print("Top 10 Rising:")
    for _, row in forecast_df.head(10).iterrows():
        print(f"  {row[COUNTRY_COL]:<8} Δ={row['predicted_delta']:>+.4f}  "
              f"→ {row['predicted_next_score']:.2f}")

    # ── Step 4: 5-year multi-step forecast (NEW) ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 4: {FORECAST_HORIZON}-YEAR MULTI-STEP FORECAST (NEW)")
    print(f"{'='*60}")
    forecast_5yr = run_5yr_forecast(df)

    # Show a sample for a major country
    sample_iso = 'IND' if 'IND' in forecast_5yr[COUNTRY_COL].values else \
                 forecast_5yr[COUNTRY_COL].iloc[0]
    print(f"\nSample — {sample_iso} forecast:")
    sample = forecast_5yr[forecast_5yr[COUNTRY_COL] == sample_iso]
    print(f"  {'Year':<6} {'Score':>8} {'80% CI':>20} {'95% CI':>20}")
    for _, r in sample.iterrows():
        ci80 = f"[{r['ci_lower_80']:.2f}, {r['ci_upper_80']:.2f}]"
        ci95 = f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]"
        print(f"  {int(r['forecast_year']):<6} {r['forecast_score']:>8.3f} {ci80:>20} {ci95:>20}")

    # ── Step 5: Regime detection (NEW) ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5: REGIME DETECTION (NEW)")
    print(f"{'='*60}")
    regime_df = detect_regimes(df, kalman_df)

    # ── Save all outputs ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")

    kalman_df.to_csv('output/kalman_results.csv', index=False)
    print("  ✅ output/kalman_results.csv")

    summary_df.to_csv('output/kalman_summary.csv', index=False)
    print("  ✅ output/kalman_summary.csv")

    out_cols = [COUNTRY_COL, 'raw_score', 'predicted_delta',
                'predicted_next_score', 'forecast_year', 'ci_lower_95', 'ci_upper_95']
    forecast_df[out_cols].to_csv('output/delta_forecast_fixed.csv', index=False)
    print("  ✅ output/delta_forecast_fixed.csv")

    forecast_5yr.to_csv('output/kalman_forecast_5yr.csv', index=False)
    print("  ✅ output/kalman_forecast_5yr.csv  ← NEW: 5-year horizon per country")

    regime_df.to_csv('output/kalman_regimes.csv', index=False)
    print("  ✅ output/kalman_regimes.csv        ← NEW: regime labels per country")

    print("\nDone. Next: run phase3_causal_complete.py")
    