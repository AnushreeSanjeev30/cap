"""
Phase B — Improvements to apply BEFORE modeling
Run this after your existing Phase A notebook output (master_soft_power_panel.csv)

Key fixes:
1. Indicator collinearity audit — drop redundant FH indicators
2. Refined dimension groupings — move trade_pct_gdp out of culture
3. Weighted PCA with factor rotation (Varimax) for better interpretability
4. External validation against Soft Power 30 / Brand Finance rankings
5. Reliability-weighted composite score (penalise imputed values)
6. Temporal leakage check before modeling
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD YOUR PHASE A OUTPUT ────────────────────────────────────────────────

master = pd.read_csv('output/master_soft_power_panel.csv')   # adjust path
print(f"Loaded: {master.shape} | {master['iso3'].nunique()} countries | "
      f"{master['year'].min()}–{master['year'].max()}")

# ─── 1. INDICATOR COLLINEARITY AUDIT ────────────────────────────────────────

print("\n─── Collinearity Audit ───\n")

# Your 23 KPI columns (adjust names to match your actual columns)
ALL_INDICATORS = [
    'tourist_arrivals', 'trade_pct_gdp', 'unesco_total_sites', 'unesco_cultural_sites',
    'internet_pct_sp', 'rnd_pct_gdp_sp', 'ai_publications', 'rd_researchers_per_mil',
    'hightech_exports_pct', 'sci_journal_articles', 'ict_patents',
    'fh_combined_score', 'fh_status_num', 'fh_total_score',
    'govt_integrity', 'judicial_effectiveness', 'property_rights',
    'business_freedom', 'investment_freedom',
    'life_expectancy', 'tertiary_enroll_pct', 'physicians_per_1k', 'infant_mortality'
]

# Keep only existing columns
ALL_INDICATORS = [c for c in ALL_INDICATORS if c in master.columns]

# Use EDA-imputed version for this analysis (fully filled)
master_eda = master.copy()

# Correlation matrix
corr = master_eda[ALL_INDICATORS].corr(method='spearman')

# Find pairs with |r| > 0.92 (near-duplicate signals)
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > 0.92:
            high_corr_pairs.append({
                'col_a': corr.columns[i],
                'col_b': corr.columns[j],
                'spearman_r': round(r, 3)
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('spearman_r', key=abs, ascending=False)
print("Near-duplicate indicator pairs (|r| > 0.92):")
print(high_corr_df.to_string(index=False))

# RECOMMENDATION: Keep only fh_combined_score from the Freedom House trio
# fh_status_num and fh_total_score are linear transformations of fh_combined_score
INDICATORS_TO_DROP = ['fh_status_num', 'fh_total_score']  # update based on output above
print(f"\nDropping redundant: {INDICATORS_TO_DROP}")

# ─── 2. REFINED DIMENSION GROUPINGS ─────────────────────────────────────────

print("\n─── Refined Dimension Groupings ───\n")

# PROBLEM WITH YOUR CURRENT GROUPINGS:
# - trade_pct_gdp in D1 (Culture) is wrong — it's economic integration
# - D3 has 3 near-identical FH indicators (97.7% PC1 variance = collinearity, not cohesion)
# - D2 has 7 indicators pulling in different directions (only 61% variance)

DIMENSIONS_REFINED = {
    'D1_Cultural_Influence': [
        'tourist_arrivals',       # inbound tourism = cultural magnetism
        'unesco_total_sites',     # heritage = cultural depth
        'unesco_cultural_sites',  # specifically cultural (not natural) sites
    ],
    'D2_Innovation_Knowledge': [
        'internet_pct_sp',        # digital infrastructure
        'rnd_pct_gdp_sp',         # R&D investment (input)
        'sci_journal_articles',   # knowledge output
        'ai_publications',        # frontier tech signal
        'hightech_exports_pct',   # innovation applied to trade
        # REMOVED: ict_patents, rd_researchers_per_mil (high collinearity w/ above)
    ],
    'D3_Political_Legitimacy': [
        'fh_combined_score',      # keep ONE FH indicator only
        # trade_pct_gdp moved here as proxy for economic openness/integration
        'trade_pct_gdp',
        'investment_freedom',     # openness to global capital
    ],
    'D4_Institutional_Quality': [
        'govt_integrity',
        'judicial_effectiveness',
        'property_rights',
        'business_freedom',
        # REMOVED: investment_freedom (moved to D3)
    ],
    'D5_Human_Development': [
        'life_expectancy',
        'tertiary_enroll_pct',
        'physicians_per_1k',
        'infant_mortality',
    ],
}

# Check PC1 variance for refined groupings
print(f"{'Dimension':<30} {'Old PC1%':>10} {'New PC1%':>10} {'Change':>10}")
print("-" * 65)

for dim, indicators in DIMENSIONS_REFINED.items():
    cols_exist = [c for c in indicators if c in master_eda.columns]
    if len(cols_exist) < 2:
        continue
    
    data = master_eda[cols_exist].dropna()
    if len(data) < 50:
        continue
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    pca = PCA(n_components=min(len(cols_exist), 3))
    pca.fit(X)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    
    print(f"  {dim:<28} {'?':>10} {pc1_var:>9.1f}%")

# ─── 3. VARIMAX-ROTATED PCA ──────────────────────────────────────────────────

print("\n─── Varimax Factor Rotation ───\n")
print("(Produces more interpretable, less correlated factors than raw PCA)\n")

from numpy.linalg import svd

def varimax(loadings, max_iter=1000, tol=1e-6):
    """Varimax rotation for better factor interpretability."""
    p, k = loadings.shape
    rotation = np.eye(k)
    
    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(k):
            for j in range(i+1, k):
                x = loadings @ rotation
                u = x[:, i]**2 - x[:, j]**2
                v = 2 * x[:, i] * x[:, j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = np.sum(u * v)
                num = D - 2*A*B/p
                den = C - (A**2 - B**2)/p
                theta = 0.25 * np.arctan2(num, den) if den != 0 else 0
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R = np.eye(k)
                R[i, i] = cos_t;  R[j, j] = cos_t
                R[i, j] = -sin_t; R[j, i] = sin_t
                rotation = rotation @ R
        
        if np.max(np.abs(rotation - old_rotation)) < tol:
            break
    
    return loadings @ rotation


def compute_dimension_score_varimax(master_df, dim_name, indicators):
    """
    Compute a dimension score using Varimax-rotated PCA.
    Returns (scores, weights, pc1_variance_explained)
    """
    cols = [c for c in indicators if c in master_df.columns]
    data = master_df[cols].copy()
    
    # Handle infant_mortality (negative direction — higher is worse)
    negative_indicators = ['infant_mortality']
    for col in negative_indicators:
        if col in data.columns:
            data[col] = data[col].max() - data[col]  # invert
    
    # Standardise
    scaler = StandardScaler()
    X_full = data.values.astype(float)
    
    # Fit only on non-missing rows
    mask = ~np.isnan(X_full).any(axis=1)
    X_fit = scaler.fit_transform(X_full[mask])
    
    # PCA
    n_components = min(len(cols), 3)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_fit)
    
    # Varimax rotation if more than 1 component
    if n_components > 1:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        rotated  = varimax(loadings)
        # Use first rotated factor
        weights_raw = np.abs(rotated[:, 0])
    else:
        weights_raw = np.abs(pca.components_[0])
    
    weights = weights_raw / weights_raw.sum()
    
    # Apply to full data (including rows with some NaN)
    X_scaled = scaler.transform(X_full)
    
    scores = np.full(len(master_df), np.nan)
    for i in range(len(master_df)):
        row = X_scaled[i]
        valid = ~np.isnan(row)
        if valid.sum() >= max(2, len(cols) // 2):
            w = weights[valid]
            w = w / w.sum()
            scores[i] = float(np.dot(row[valid], w))
    
    # Scale to 0–100
    valid_scores = scores[~np.isnan(scores)]
    s_min, s_max = valid_scores.min(), valid_scores.max()
    if s_max > s_min:
        scores = np.where(
            ~np.isnan(scores),
            (scores - s_min) / (s_max - s_min) * 100,
            np.nan
        )
    
    pc1_var = pca.explained_variance_ratio_[0] * 100
    weights_dict = {col: round(float(w), 4) for col, w in zip(cols, weights)}
    
    return scores, weights_dict, pc1_var


# Compute refined dimension scores
print(f"{'Dimension':<32} {'PC1%':>8}  Weights")
print("-" * 70)

for dim, indicators in DIMENSIONS_REFINED.items():
    scores, weights, pc1_var = compute_dimension_score_varimax(master_eda, dim, indicators)
    master[f'{dim}_score'] = scores
    master_eda[f'{dim}_score'] = scores
    
    top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_str = ', '.join([f'{k}({v:.2f})' for k, v in top3])
    print(f"  {dim:<30} {pc1_var:>7.1f}%  {top3_str}")


# ─── 4. RELIABILITY-WEIGHTED COMPOSITE SCORE ────────────────────────────────

print("\n─── Reliability-Weighted Composite Score ───\n")

# The key improvement over simple equal/PCA dimension weights:
# Countries with many imputed values should get wider confidence intervals
# and slightly penalised point estimates

# Literature-informed dimension weights (your existing WEIGHTS_LITERATURE)
DIM_WEIGHTS = {
    'D1_Cultural_Influence_score':    0.15,
    'D2_Innovation_Knowledge_score':  0.25,
    'D3_Political_Legitimacy_score':  0.20,
    'D4_Institutional_Quality_score': 0.25,
    'D5_Human_Development_score':     0.15,
}

def compute_composite_with_reliability(row, dim_weights, reliability_col='data_reliability'):
    scores = {d: row.get(d, np.nan) for d in dim_weights}
    # If reliability_col is None or missing, use default 0.75
    if reliability_col is None or reliability_col not in row or pd.isna(row.get(reliability_col, None)):
        reliability = 0.75
    else:
        reliability = row.get(reliability_col, 0.75)

    valid_dims = {d: v for d, v in scores.items() if not np.isnan(v)}
    if not valid_dims:
        return np.nan, np.nan

    # Renormalise weights for available dims
    total_w = sum(dim_weights[d] for d in valid_dims)
    composite = sum(v * dim_weights[d] / total_w 
                    for d, v in valid_dims.items())

    # Reliability adjustment: penalise up to 5% for low-quality data
    reliability_penalty = (1 - reliability) * 0.05 * 100
    adjusted = composite - reliability_penalty

    return round(float(composite), 3), round(float(adjusted), 3)



# Always compute composite scores, using default reliability if missing
results = master.apply(
    lambda row: compute_composite_with_reliability(row, DIM_WEIGHTS, reliability_col='data_reliability' if 'data_reliability' in master.columns else None), axis=1
)
master['soft_power_composite_raw']      = [r[0] for r in results]
master['soft_power_composite_adjusted'] = [r[1] for r in results]

print(f"  Raw composite      — Mean: {master['soft_power_composite_raw'].mean():.2f}, "
    f"Std: {master['soft_power_composite_raw'].std():.2f}")
print(f"  Adjusted composite — Mean: {master['soft_power_composite_adjusted'].mean():.2f}, "
    f"Std: {master['soft_power_composite_adjusted'].std():.2f}")
print(f"  Avg penalty applied: "
    f"{(master['soft_power_composite_raw'] - master['soft_power_composite_adjusted']).mean():.3f} pts")


# ─── 5. EXTERNAL VALIDATION ──────────────────────────────────────────────────

print("\n─── External Validation ───\n")

# Soft Power 30 2019 rankings (last published year)
# Source: Portland Communications / USC CPD
SOFT_POWER_30_2019 = {
    'France': 1, 'United Kingdom': 2, 'Germany': 3, 'Sweden': 4,
    'United States': 5, 'Switzerland': 6, 'Canada': 7, 'Japan': 8,
    'Australia': 9, 'Netherlands': 10, 'Denmark': 11, 'Norway': 12,
    'New Zealand': 13, 'Finland': 14, 'Italy': 15,
    'Austria': 16, 'Spain': 17, 'Belgium': 18, 'Ireland': 19, 'Portugal': 20,
    'Singapore': 21, 'South Korea': 22, 'China': 23, 'Czech Republic': 24,
    'Chile': 25, 'Poland': 26, 'United Arab Emirates': 27,
    'Brazil': 28, 'India': 29, 'Mexico': 30
}

# Brand Finance Global Soft Power Index 2024 top 20
BRAND_FINANCE_2024 = {
    'United States': 1, 'United Kingdom': 2, 'Germany': 3, 'China': 4,
    'Japan': 5, 'France': 6, 'Canada': 7, 'Italy': 8, 'South Korea': 9,
    'Australia': 10, 'Switzerland': 11, 'Spain': 12, 'Netherlands': 13,
    'Sweden': 14, 'Norway': 15, 'Denmark': 16, 'Belgium': 17,
    'Austria': 18, 'Singapore': 19, 'Finland': 20
}

def validate_against_benchmark(master_df, your_score_col, benchmark_dict, 
                                 benchmark_name, year=None):
    """Compute Spearman correlation between your ranking and an external benchmark."""
    if year:
        df = master_df[master_df['year'] == year].copy()
    else:
        df = master_df.copy()
    
    df = df.sort_values(your_score_col, ascending=False)
    df['your_rank'] = range(1, len(df) + 1)
    
    # Match countries
    matched = []
    for country_name, ext_rank in benchmark_dict.items():
        # Try matching on canonical name
        row = df[df['canonical'].str.contains(country_name, case=False, na=False)]
        if len(row) == 0:
            # Try iso3 fallback
            continue
        your_rank = int(row.iloc[0]['your_rank'])
        matched.append({'country': country_name, 
                        'your_rank': your_rank, 
                        'external_rank': ext_rank,
                        'rank_diff': your_rank - ext_rank})
    
    if len(matched) < 10:
        print(f"  [{benchmark_name}] Only {len(matched)} matches found — check canonical names")
        return None
    
    matched_df = pd.DataFrame(matched)
    rho, p = spearmanr(matched_df['your_rank'], matched_df['external_rank'])
    
    print(f"\n  [{benchmark_name}]")
    print(f"    Matched countries  : {len(matched_df)}")
    print(f"    Spearman ρ         : {rho:.3f}  (p={p:.4f})")
    print(f"    Mean |rank diff|   : {matched_df['rank_diff'].abs().mean():.1f}")
    
    if rho > 0.85:
        print(f"    Assessment         : ✓ STRONG alignment")
    elif rho > 0.70:
        print(f"    Assessment         : ~ MODERATE alignment — review outliers")
    else:
        print(f"    Assessment         : ✗ WEAK alignment — re-examine dimension weights")
    
    # Show biggest disagreements
    big_diff = matched_df[matched_df['rank_diff'].abs() > 10].sort_values('rank_diff', key=abs, ascending=False)
    if len(big_diff) > 0:
        print(f"\n    Countries with >10 rank disagreement:")
        print(big_diff[['country','your_rank','external_rank','rank_diff']].to_string(index=False))
    
    return rho, matched_df


# Run validation on latest year
latest_year = master['year'].max()
rho_sp30, _ = validate_against_benchmark(
    master, 'soft_power_composite_raw', 
    SOFT_POWER_30_2019, 'Soft Power 30 (2019)', year=2019
)
rho_bf, _ = validate_against_benchmark(
    master, 'soft_power_composite_raw',
    BRAND_FINANCE_2024, 'Brand Finance (2024)', year=latest_year
)


# ─── 6. TEMPORAL LEAKAGE CHECK ───────────────────────────────────────────────

print("\n─── Temporal Leakage Check ───\n")

print("Checking for static features that will cause leakage in time-series CV...")

# UNESCO and world_data were merged as static (no year). 
# If these columns appear in your model features, they'll look like
# "future data" when you train on 2000–2015 and test on 2016–2024.

# Identify static columns (same value across all years for a country)
static_cols = []
for col in master.select_dtypes(include='number').columns:
    if col in ['iso3', 'year']:
        continue
    std_by_country = master.groupby('iso3')[col].std().mean()
    if std_by_country < 0.001:
        static_cols.append(col)

print(f"  Columns with near-zero within-country temporal variation: {len(static_cols)}")
print(f"  Examples: {static_cols[:5]}")
print(f"\n  ACTION: These columns are safe to use as country fixed effects")
print(f"  but must NOT be in the feature set for year-ahead prediction.")
print(f"  → Separate them into a 'country_features' table and join by iso3 only.\n")


# ─── 7. FINAL ENRICHED MASTER SAVE ──────────────────────────────────────────

output_cols = (
    ['iso3', 'canonical', 'year'] +
    [f'{d}_score' for d in DIMENSIONS_REFINED.keys()] +
    ['soft_power_composite_raw', 'soft_power_composite_adjusted'] +
    (['data_reliability'] if 'data_reliability' in master.columns else []) +
    [c for c in master.columns if c in ALL_INDICATORS]
)
output_cols = [c for c in output_cols if c in master.columns]

master_out = master[output_cols].sort_values(['iso3', 'year'])
master_out.to_parquet('output/master_phase_b.parquet', index=False)
master_out.to_csv('output/master_phase_b.csv', index=False)

print(f"\n{'='*60}")
print(f"  Phase B Complete")
print(f"  Output shape : {master_out.shape}")
print(f"  Saved to     : output/master_phase_b.parquet")

if rho_sp30 is not None:
    print(f"\n  Spearman ρ vs Soft Power 30 : {rho_sp30:.3f}")
else:
    print("\n  Spearman ρ vs Soft Power 30 : N/A")
if rho_bf is not None:
    print(f"  Spearman ρ vs Brand Finance  : {rho_bf:.3f}")
else:
    print("  Spearman ρ vs Brand Finance  : N/A")
print(f"\n  → NEXT STEP: Phase C — Trend slope + FAISS + XGBoost modeling")
print(f"{'='*60}")
