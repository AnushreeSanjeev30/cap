"""
Patch: Population Filter for Kalman + Forecast outputs
========================================================
Micro-states (population < 1 million) inflate per-capita KPIs and
dominate volatility rankings with statistical noise, not real soft
power shifts. This patch filters them out and re-runs the summary.

Add this BEFORE the kalman_softpower.py main block, or run standalone
after generating kalman_results.csv and kalman_summary.csv.

Two options:
  A) Hard exclusion list (fast, no extra data needed)
  B) World Bank population data join (robust, recommended if you have it)

We use Option A here with a known micro-state ISO3 list.
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
KALMAN_SUMMARY  = 'output/kalman_summary.csv'
KALMAN_RESULTS  = 'output/kalman_results.csv'
FORECAST        = 'output/delta_forecast_fixed.csv'
COUNTRY_COL     = 'iso3'

# ISO3 codes for micro-states / territories to exclude
# (population < ~1 million OR dependent territories)
MICRO_STATES = {
    'MCO', 'SMR', 'LIE', 'AND', 'VAT',          # European micro-states
    'NRU', 'TUV', 'PLW', 'MHL', 'FSM', 'KIR',   # Pacific micro-states
    'TON', 'WSM', 'VUT', 'SLB', 'COK', 'NIU',
    'TLS',                                         # Timor-Leste (data sparse)
    'STP', 'CPV', 'COM', 'DJI', 'GNB',           # Small African island states
    'LCA', 'VCT', 'GRD', 'DMA', 'ATG', 'KNA',   # Caribbean micro-states
    'BLZ', 'SUR',
    'BRN',                                         # Brunei — tiny, oil-skewed
    'MDV', 'BTN',                                  # South Asian micro-states
    'MLT', 'CYP',                                  # Mediterranean small states
}

# ─────────────────────────────────────────────────────────────────────────────

def filter_and_reprint(summary_df, kalman_df, forecast_df):
    # Filter
    s = summary_df[~summary_df[COUNTRY_COL].isin(MICRO_STATES)].copy()
    k = kalman_df[~kalman_df[COUNTRY_COL].isin(MICRO_STATES)].copy()
    f = forecast_df[~forecast_df[COUNTRY_COL].isin(MICRO_STATES)].copy()

    print(f"Countries after micro-state filter: {len(s)} (removed {len(summary_df)-len(s)})\n")

    # Reclassify stability on filtered set
    q33 = s['innovation_std'].quantile(0.33)
    q66 = s['innovation_std'].quantile(0.66)
    s['stability_class'] = pd.cut(
        s['innovation_std'],
        bins=[-np.inf, q33, q66, np.inf],
        labels=['Stable', 'Moderate', 'Volatile']
    )

    print("TOP 15 COUNTRIES BY LATEST KALMAN SCORE:")
    print(f"  {'Country':<8} {'Score':>8} {'CI ±':>8} {'Trend/yr':>10} {'Stability':<12}")
    print(f"  {'-'*52}")
    for _, row in s.sort_values('latest_score', ascending=False).head(15).iterrows():
        ci_half = row['latest_ci'] / 2
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>8.2f} "
              f"{ci_half:>8.2f} {row['trend_slope']:>+10.3f} "
              f"{str(row['stability_class']):<12}")

    print(f"\nSTABILITY BREAKDOWN:")
    for cls in ['Stable', 'Moderate', 'Volatile']:
        n = (s['stability_class'] == cls).sum()
        print(f"  {cls:<10}: {n} countries")

    print(f"\n🚀 TOP 10 RISING POWERS:")
    print(f"  {'Country':<8} {'Score':>8} {'Trend/yr':>10} {'Stability':<12}")
    print(f"  {'-'*42}")
    for _, row in s.nlargest(10, 'trend_slope').iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>8.2f} "
              f"{row['trend_slope']:>+10.3f} {str(row['stability_class']):<12}")

    print(f"\n📉 TOP 10 DECLINING POWERS:")
    print(f"  {'Country':<8} {'Score':>8} {'Trend/yr':>10} {'Stability':<12}")
    print(f"  {'-'*42}")
    for _, row in s.nsmallest(10, 'trend_slope').iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['latest_score']:>8.2f} "
              f"{row['trend_slope']:>+10.3f} {str(row['stability_class']):<12}")

    print(f"\n🚀 TOP 10 FORECAST RISING (Kalman delta):")
    print(f"  {'Country':<8} {'Current':>10} {'Δ':>10} {'Next':>10}")
    print(f"  {'-'*42}")
    for _, row in f.head(10).iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['raw_score']:>10.2f} "
              f"{row['predicted_delta']:>+10.4f} {row['predicted_next_score']:>10.2f}")

    print(f"\n📉 TOP 10 FORECAST DECLINING (Kalman delta):")
    print(f"  {'Country':<8} {'Current':>10} {'Δ':>10} {'Next':>10}")
    print(f"  {'-'*42}")
    for _, row in f.tail(10).iterrows():
        print(f"  {row[COUNTRY_COL]:<8} {row['raw_score']:>10.2f} "
              f"{row['predicted_delta']:>+10.4f} {row['predicted_next_score']:>10.2f}")

    # Save filtered versions
    s.to_csv('output/kalman_summary_filtered.csv', index=False)
    k.to_csv('output/kalman_results_filtered.csv', index=False)
    f.to_csv('output/delta_forecast_filtered.csv', index=False)

    print(f"\n✅ output/kalman_summary_filtered.csv")
    print(f"✅ output/kalman_results_filtered.csv")
    print(f"✅ output/delta_forecast_filtered.csv")

    return s, k, f


if __name__ == '__main__':
    summary_df  = pd.read_csv(KALMAN_SUMMARY)
    kalman_df   = pd.read_csv(KALMAN_RESULTS)
    forecast_df = pd.read_csv(FORECAST).sort_values('predicted_delta', ascending=False)

    print(f"Loaded: {len(summary_df)} countries in summary\n")
    filter_and_reprint(summary_df, kalman_df, forecast_df)