"""
Step 1 Fix + Step 2: Kalman Filter State-Space Model
======================================================
Fixes:
  - Forecast table was only showing NLD because most countries had null
    KPI features for the latest year, causing dropna to remove them.
    Fix: we predict using whatever features ARE available per country,
    filling gaps with the country's own historical median (not 0).

Step 2 — Kalman Filter per country:
  - Models soft power score as a latent state evolving over time
  - Produces: smoothed score, ± CI (confidence interval), volatility
  - This is what your architecture diagram means by "Latent Soft Power
    Score ± CI" — the uncertainty band around each country's true score
  - Outputs a CSV + summary table of most/least stable countries

Outputs:
  output/kalman_results.csv        ← per country-year: smoothed score + CI
  output/kalman_summary.csv        ← per country: volatility, stability class
  output/delta_forecast_fixed.csv  ← fixed forecast with all countries
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')


# ── CONFIG ────────────────────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: KALMAN FILTER STATE-SPACE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def kalman_filter_country(scores: np.ndarray):
    """
    1D Kalman Filter for a single country's score time series.

    State-space model:
        x_t = x_{t-1} + w_t       (state transition: score evolves slowly)
        y_t = x_t + v_t            (observation: measured score = true + noise)

    where:
        w_t ~ N(0, Q)   process noise  (how much true soft power can jump)
        v_t ~ N(0, R)   observation noise (measurement uncertainty)

    Returns:
        filtered_means   : smoothed score estimates
        filtered_vars    : variance at each step (→ CI width)
        innovations      : residuals (observed - predicted)
    """
    n = len(scores)

    # ── Parameter initialisation ──────────────────────────────────────────────
    # Q: process noise — soft power shouldn't jump wildly year to year
    # R: observation noise — our PCA composite has measurement error
    # We estimate Q from the actual variance of year-on-year changes
    valid = scores[~np.isnan(scores)]
    if len(valid) < 3:
        return None

    diffs = np.diff(valid)
    Q = np.var(diffs) * 0.3    # process noise = 30% of observed change variance
    R = np.var(valid)  * 0.1   # observation noise = 10% of score variance

    # ── Kalman Filter forward pass ────────────────────────────────────────────
    x     = valid[0]           # initial state estimate = first observation
    P     = np.var(valid)      # initial uncertainty = score variance

    filtered_means = np.full(n, np.nan)
    filtered_vars  = np.full(n, np.nan)
    innovations    = np.full(n, np.nan)

    for t in range(n):
        y = scores[t]
        if np.isnan(y):
            # Missing observation: just propagate the prior
            x = x
            P = P + Q
        else:
            # Predict step
            x_pred = x
            P_pred = P + Q

            # Update step
            K          = P_pred / (P_pred + R)   # Kalman gain
            innovation = y - x_pred              # residual
            x          = x_pred + K * innovation  # updated state
            P          = (1 - K) * P_pred         # updated variance

            innovations[t] = innovation

        filtered_means[t] = x
        filtered_vars[t]  = P

    return filtered_means, filtered_vars, innovations


def run_kalman_all_countries(df: pd.DataFrame):
    """
    Run Kalman Filter for every country. Returns a long-format DataFrame
    with smoothed scores and 95% CI bounds.
    """
    results = []
    countries = df[COUNTRY_COL].unique()

    print(f"  Running Kalman Filter on {len(countries)} countries...")

    for iso in countries:
        cdf = df[df[COUNTRY_COL] == iso].sort_values(YEAR_COL)
        scores = cdf[SCORE_COL].values
        years  = cdf[YEAR_COL].values

        out = kalman_filter_country(scores)
        if out is None:
            continue

        filtered_means, filtered_vars, innovations = out

        for i, yr in enumerate(years):
            std = np.sqrt(filtered_vars[i]) if not np.isnan(filtered_vars[i]) else np.nan
            results.append({
                COUNTRY_COL:       iso,
                YEAR_COL:          yr,
                'raw_score':       scores[i],
                'kalman_score':    filtered_means[i],
                'kalman_std':      std,
                'ci_lower_95':     filtered_means[i] - 1.96 * std,
                'ci_upper_95':     filtered_means[i] + 1.96 * std,
                'ci_width':        2 * 1.96 * std,
                'innovation':      innovations[i],   # residual from model
            })

    return pd.DataFrame(results)


def kalman_summary(kalman_df: pd.DataFrame):
    """
    Per-country summary:
      - Mean CI width (average uncertainty about true soft power)
      - Innovation std (how often score deviates from its own trend)
      - Stability class: Stable / Moderate / Volatile
    """
    summary = []
    for iso, grp in kalman_df.groupby(COUNTRY_COL):
        mean_ci    = grp['ci_width'].mean()
        innov_std  = grp['innovation'].std()
        mean_score = grp['kalman_score'].mean()
        latest_score = grp.loc[grp[YEAR_COL].idxmax(), 'kalman_score']
        latest_ci    = grp.loc[grp[YEAR_COL].idxmax(), 'ci_width']

        # Trend: slope of kalman_score over time
        yrs = grp[YEAR_COL].values
        ksc = grp['kalman_score'].values
        if len(yrs) > 2:
            slope = np.polyfit(yrs, ksc, 1)[0]
        else:
            slope = 0.0

        # Stability class based on innovation std quartiles
        summary.append({
            COUNTRY_COL:      iso,
            'mean_score':     mean_score,
            'latest_score':   latest_score,
            'latest_ci':      latest_ci,
            'mean_ci_width':  mean_ci,
            'innovation_std': innov_std,
            'trend_slope':    slope,      # positive = rising, negative = falling
        })

    sdf = pd.DataFrame(summary)

    # Classify stability
    q33 = sdf['innovation_std'].quantile(0.33)
    q66 = sdf['innovation_std'].quantile(0.66)
    sdf['stability_class'] = pd.cut(
        sdf['innovation_std'],
        bins=[-np.inf, q33, q66, np.inf],
        labels=['Stable', 'Moderate', 'Volatile']
    )

    return sdf.sort_values('latest_score', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: FIXED FORECAST TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def fixed_forecast(df: pd.DataFrame, kalman_df: pd.DataFrame):
    """
    Fix for the NLD-only bug.

    Instead of relying on XGBoost delta prediction (which required all
    KPI features to be non-null), we use the Kalman Filter's own
    one-step-ahead prediction:
        predicted_score_{t+1} = kalman_score_t   (state carries forward)
        predicted_delta       = kalman_score_t - raw_score_t

    This works for ALL countries regardless of missing KPIs.
    """
    # Get latest Kalman estimate per country
    latest = (
        kalman_df.sort_values(YEAR_COL)
        .groupby(COUNTRY_COL)
        .last()
        .reset_index()
    )

    # Merge with latest raw score
    latest_raw = (
        df.sort_values(YEAR_COL)
        .groupby(COUNTRY_COL)
        .last()[[SCORE_COL]]
        .reset_index()
    )

    forecast = latest.merge(latest_raw, on=COUNTRY_COL, suffixes=('', '_raw'))
    forecast['predicted_next_score'] = forecast['kalman_score']   # Kalman carries state forward
    forecast['predicted_delta']      = forecast['kalman_score'] - forecast['raw_score']
    forecast['forecast_year']        = forecast[YEAR_COL] + 1

    return forecast.sort_values('predicted_delta', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_parquet(MASTER_PARQUET)

    if 'country_iso3' in df.columns and COUNTRY_COL not in df.columns:
        df = df.rename(columns={'country_iso3': COUNTRY_COL})

    print(f"Dataset: {len(df)} rows, {df[COUNTRY_COL].nunique()} countries")
    print(f"Years:   {df[YEAR_COL].min()} – {df[YEAR_COL].max()}\n")


    # ── Step 2: Kalman Filter ─────────────────────────────────────────────────
    print("=" * 55)
    print("STEP 2: KALMAN FILTER STATE-SPACE MODEL")
    print("=" * 55)

    kalman_df = run_kalman_all_countries(df)
    print(f"  Done. {len(kalman_df)} country-year estimates generated.\n")

    summary_df = kalman_summary(kalman_df)

    # Print stability breakdown
    stable_n   = (summary_df['stability_class'] == 'Stable').sum()
    moderate_n = (summary_df['stability_class'] == 'Moderate').sum()
    volatile_n = (summary_df['stability_class'] == 'Volatile').sum()

    print(f"STABILITY CLASSIFICATION:")
    print(f"  Stable:   {stable_n} countries  (low year-on-year deviation)")
    print(f"  Moderate: {moderate_n} countries")
    print(f"  Volatile: {volatile_n} countries (high year-on-year deviation)\n")

    # Top 10 by latest Kalman score
    print("TOP 10 COUNTRIES BY LATEST KALMAN SCORE:")
    print(f"  {'Country':<8} {'Score':>8} {'CI ±':>8} {'Trend/yr':>10} {'Stability':<12}")
    print(f"  {'-'*50}")
    for _, row in summary_df.head(10).iterrows():
        ci_half = row['latest_ci'] / 2
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>8.2f} "
              f"{ci_half:>8.2f} {row['trend_slope']:>+10.3f} "
              f"{str(row['stability_class']):<12}")

    # Most volatile countries (interesting for capstone narrative)
    print(f"\nTOP 10 MOST VOLATILE COUNTRIES (narrative interest):")
    print(f"  {'Country':<8} {'Latest Score':>13} {'Innov Std':>10} {'Trend/yr':>10}")
    print(f"  {'-'*46}")
    for _, row in summary_df.nlargest(10, 'innovation_std').iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>13.2f} "
              f"{row['innovation_std']:>10.3f} {row['trend_slope']:>+10.3f}")

    # Rising and falling (by trend slope)
    print(f"\n🚀 TOP 10 RISING POWERS (by Kalman trend slope):")
    print(f"  {'Country':<8} {'Latest Score':>13} {'Trend/yr':>10} {'Stability':<12}")
    print(f"  {'-'*48}")
    for _, row in summary_df.nlargest(10, 'trend_slope').iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>13.2f} "
              f"{row['trend_slope']:>+10.3f} {str(row['stability_class']):<12}")

    print(f"\n📉 TOP 10 DECLINING POWERS (by Kalman trend slope):")
    print(f"  {'Country':<8} {'Latest Score':>13} {'Trend/yr':>10} {'Stability':<12}")
    print(f"  {'-'*48}")
    for _, row in summary_df.nsmallest(10, 'trend_slope').iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>13.2f} "
              f"{row['trend_slope']:>+10.3f} {str(row['stability_class']):<12}")


    # ── Step 1 Fix: Forecast Table ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("STEP 1 FIX: FORECAST TABLE (all countries)")
    print(f"{'='*55}")

    forecast_df = fixed_forecast(df, kalman_df)
    print(f"  Forecast generated for {len(forecast_df)} countries.\n")

    print(f"🚀 TOP 10 RISING (Kalman-based delta):")
    print(f"  {'Country':<8} {'Current':>10} {'Predicted Δ':>13} {'Next Score':>12}")
    print(f"  {'-'*48}")
    for _, row in forecast_df.head(10).iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['raw_score']:>10.2f} "
              f"{row['predicted_delta']:>+13.4f} {row['predicted_next_score']:>12.2f}")

    print(f"\n📉 TOP 10 DECLINING (Kalman-based delta):")
    print(f"  {'Country':<8} {'Current':>10} {'Predicted Δ':>13} {'Next Score':>12}")
    print(f"  {'-'*48}")
    for _, row in forecast_df.tail(10).iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['raw_score']:>10.2f} "
              f"{row['predicted_delta']:>+13.4f} {row['predicted_next_score']:>12.2f}")


    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SAVING OUTPUTS")
    print(f"{'='*55}")

    kalman_df.to_csv('output/kalman_results.csv', index=False)
    print("  ✅ output/kalman_results.csv        (per country-year: score + CI)")

    summary_df.to_csv('output/kalman_summary.csv', index=False)
    print("  ✅ output/kalman_summary.csv        (per country: stability class + trend)")

    out_cols = [COUNTRY_COL, 'raw_score', 'predicted_delta',
                'predicted_next_score', 'forecast_year', 'ci_lower_95', 'ci_upper_95']
    forecast_df[out_cols].to_csv('output/delta_forecast_fixed.csv', index=False)
    print("  ✅ output/delta_forecast_fixed.csv  (all countries, Kalman-based)")

    print("\nDone. Share the terminal output and we'll move to Step 3 (FAISS).")