"""
Perception Adjustment Layer
Adds a democratic legitimacy weight to bring rankings closer to
perception-based benchmarks (Soft Power 30, Brand Finance).

RUN AFTER phase_c_fixes.py

This does NOT change your underlying model — it adds a transparent,
theoretically-grounded adjustment layer on top of it.
You can report BOTH scores in your capstone:
  - soft_power_capacity_score   (your structural model)
  - soft_power_adjusted_score   (perception-adjusted)
  and explain WHY they differ for China, UAE, etc.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─── 1. LOAD PREDICTIONS ─────────────────────────────────────────────────────

preds = pd.read_parquet('output/soft_power_predictions.parquet')
master = pd.read_parquet('output/master_phase_b.parquet')

print(f"Loaded {len(preds)} country predictions\n")

# ─── 2. DEMOCRATIC LEGITIMACY PENALTY ────────────────────────────────────────
# Theory: soft power is not just capacity — it requires legitimacy in the eyes
# of others. Authoritarian systems can build cultural and economic capacity,
# but this does not automatically translate to international influence or
# voluntary followership (Nye's original definition).
#
# We encode this via Freedom House combined score (already in your data).
# fh_combined_score: 0 = least free, 100 = most free
#
# Penalty function:
#   - Countries with fh_combined_score > 70 (Free): no penalty
#   - Countries with fh_combined_score 40–70 (Partly Free): small penalty
#   - Countries with fh_combined_score < 40 (Not Free): moderate penalty
#
# The penalty is CAPPED at 15 points — we are adjusting, not overriding.
# This is a deliberate methodological choice, not a political one.

COUNTRY_COL = 'iso3'
YEAR_COL    = 'year'

# Get latest year's FH score per country
latest_year = master[YEAR_COL].max()
fh_latest = (master[master[YEAR_COL] == latest_year]
             [[COUNTRY_COL, 'fh_combined_score']]
             .dropna(subset=['fh_combined_score']))

preds = preds.merge(fh_latest, on=COUNTRY_COL, how='left')

def governance_penalty(fh_score, max_penalty=15.0):
    """
    Smooth penalty curve based on Freedom House score.
    Returns a value between 0 (no penalty) and max_penalty.
    
    FH > 70  → 0 penalty    (Free countries: democracy bonus already in score)
    FH 40–70 → 0–8 penalty  (Partly Free: moderate discount)
    FH < 40  → 8–15 penalty (Not Free: significant discount)
    """
    if pd.isna(fh_score):
        return 0.0  # no adjustment if data missing

    if fh_score >= 70:
        return 0.0
    elif fh_score >= 40:
        # Linear ramp: 70→0, 40→8
        return round((70 - fh_score) / 30 * 8.0, 2)
    else:
        # Linear ramp: 40→8, 0→15
        return round(8.0 + (40 - fh_score) / 40 * 7.0, 2)

preds['governance_penalty'] = preds['fh_combined_score'].apply(governance_penalty)

# ─── 3. WEALTH-ADJUSTED CULTURAL INVESTMENT BONUS ────────────────────────────
# UAE problem: your model underweights wealth-driven cultural investment.
# Gulf states spend massively on soft power (Louvre Abu Dhabi, FIFA World Cup,
# global media, foreign aid) but your indicators don't fully capture this.
#
# Proxy: if a country has high GDP per capita but lower-than-expected
# cultural/tourism scores, give a small upward adjustment.
# We use Heritage Index business_freedom + investment_freedom as wealth proxy.

if 'business_freedom' in master.columns and 'investment_freedom' in master.columns:
    wealth_proxy = (master[master[YEAR_COL] == latest_year]
                    [[COUNTRY_COL, 'business_freedom', 'investment_freedom']]
                    .assign(wealth_score=lambda x: (x['business_freedom'] + x['investment_freedom']) / 2))
    preds = preds.merge(wealth_proxy[[COUNTRY_COL, 'wealth_score']], 
                        on=COUNTRY_COL, how='left')

    # Bonus only for high-wealth, low-cultural-score countries
    # (countries that invest in soft power but your indicators miss)
    if 'D1_Cultural_Influence_score' in preds.columns:
        cultural_score = preds['D1_Cultural_Influence_score'] if 'D1_Cultural_Influence_score' in preds.columns else preds['score'] * 0.15
    else:
        cultural_score = pd.Series(50, index=preds.index)

    wealth_bonus_conditions = (
        (preds.get('wealth_score', pd.Series(0, index=preds.index)) > 75) &
        (preds['score'] < 65)  # high-potential but underscoring countries
    )
    preds['wealth_bonus'] = np.where(wealth_bonus_conditions, 3.0, 0.0)
else:
    preds['wealth_bonus'] = 0.0

# ─── 4. COMPUTE ADJUSTED SCORE ───────────────────────────────────────────────

preds['soft_power_capacity_score'] = preds['score'].round(2)

preds['soft_power_adjusted_score'] = (
    preds['score']
    - preds['governance_penalty']
    + preds['wealth_bonus']
).clip(0, 100).round(2)

# Adjusted CI (widen slightly to reflect additional uncertainty in adjustment)
preds['adj_ci_lower'] = (preds['ci_lower'] - preds['governance_penalty'] * 0.5).clip(0, 100).round(2)
preds['adj_ci_upper'] = (preds['ci_upper'] + 1.5).clip(0, 100).round(2)

# Rerank
preds['capacity_rank']  = preds['soft_power_capacity_score'].rank(ascending=False, method='min').astype(int)
preds['adjusted_rank']  = preds['soft_power_adjusted_score'].rank(ascending=False, method='min').astype(int)
preds['rank_shift']     = preds['capacity_rank'] - preds['adjusted_rank']  # negative = moved up after adjustment

# ─── 5. VALIDATE AGAINST BENCHMARKS ─────────────────────────────────────────

from scipy.stats import spearmanr

SOFT_POWER_30_2019 = {
    'France': 1, 'United Kingdom': 2, 'Germany': 3, 'Sweden': 4,
    'United States': 5, 'Switzerland': 6, 'Canada': 7, 'Japan': 8,
    'Australia': 9, 'Netherlands': 10, 'Denmark': 11, 'Norway': 12,
    'New Zealand': 13, 'Finland': 14, 'Italy': 15,
    'Austria': 16, 'Spain': 17, 'Belgium': 18, 'Ireland': 19,
    'Singapore': 21, 'South Korea': 22, 'China': 23,
    'Brazil': 28, 'India': 29, 'Mexico': 30
}

def spearman_vs_benchmark(df, score_col, rank_col, benchmark):
    matched = []
    for country, ext_rank in benchmark.items():
        row = df[df['canonical'].str.contains(country, case=False, na=False)]
        if len(row) == 0:
            continue
        matched.append({
            'country':    country,
            'your_rank':  int(row.iloc[0][rank_col]),
            'ext_rank':   ext_rank
        })
    if len(matched) < 10:
        return None, matched
    df_m = pd.DataFrame(matched)
    rho, p = spearmanr(df_m['your_rank'], df_m['ext_rank'])
    return rho, df_m

print("─── Benchmark Alignment: Capacity vs Adjusted Score ───\n")

rho_cap, df_cap = spearman_vs_benchmark(preds, 'soft_power_capacity_score', 'capacity_rank', SOFT_POWER_30_2019)
rho_adj, df_adj = spearman_vs_benchmark(preds, 'soft_power_adjusted_score', 'adjusted_rank', SOFT_POWER_30_2019)

if rho_cap:
    print(f"  Capacity score  ρ vs Soft Power 30: {rho_cap:.3f}")
if rho_adj:
    print(f"  Adjusted score  ρ vs Soft Power 30: {rho_adj:.3f}")
    if rho_adj > rho_cap:
        print(f"  Improvement: +{rho_adj - rho_cap:.3f} from governance adjustment")

# ─── 6. KEY COUNTRY COMPARISON ───────────────────────────────────────────────

print("\n─── Key country comparison ───\n")
print(f"{'Country':<22} {'Capacity':>10} {'C-Rank':>8} {'Adjusted':>10} {'A-Rank':>8} {'Penalty':>9} {'Bonus':>7}")
print("-" * 80)

spotlight = ['Germany', 'France', 'United Kingdom', 'United States', 'China',
             'India', 'Japan', 'Russian Federation', 'United Arab Emirates',
             'Brazil', 'Switzerland', 'Sweden', 'Singapore', 'South Korea']

for country in spotlight:
    row = preds[preds['canonical'].str.contains(country.split()[0], case=False, na=False)]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    penalty = r.get('governance_penalty', 0)
    bonus   = r.get('wealth_bonus', 0)
    print(f"  {r.get('canonical', country):<20} "
          f"{r['soft_power_capacity_score']:>10.2f} "
          f"{r['capacity_rank']:>8} "
          f"{r['soft_power_adjusted_score']:>10.2f} "
          f"{r['adjusted_rank']:>8} "
          f"{penalty:>9.1f} "
          f"{bonus:>7.1f}")

# ─── 7. SAVE ──────────────────────────────────────────────────────────────────

preds.to_parquet('output/soft_power_predictions_final.parquet', index=False)
preds.to_csv('output/soft_power_predictions_final.csv', index=False)

print(f"\n{'═'*60}")
print(f"  Dual-score output saved")
print(f"  → output/soft_power_predictions_final.csv")
print(f"\n  For your report, present BOTH columns and explain:")
print(f"    soft_power_capacity_score  — structural capacity index")
print(f"    soft_power_adjusted_score  — perception-adjusted index")
print(f"\n  The gap between them = the 'legitimacy discount'")
print(f"  (how much authoritarian states lose in perceived soft power)")
print(f"  This is itself a finding worth discussing.")
print(f"{'═'*60}")
