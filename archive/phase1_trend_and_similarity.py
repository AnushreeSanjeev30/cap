"""
Phase 1: Trend Slope Analysis + Country Similarity Vectors
Inputs: master_panel DataFrame (country, year, 23 KPIs, pca_score)
Outputs: trend_features DataFrame, FAISS index + country embeddings
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import faiss
import pickle
import os

# ─── 1. TREND SLOPE ANALYSIS ───────────────────────────────────────────────

def compute_trend_features(df: pd.DataFrame, 
                            indicators: list,
                            country_col='iso3',
                            year_col='year',
                            score_col='soft_power_composite_raw') -> pd.DataFrame:
    """
    For each country, compute:
    - OLS slope for each of the 23 indicators (trend direction + magnitude)
    - OLS slope for the composite PCA score
    - R² of that trend (trend reliability)
    - Volatility (std dev of residuals)
    - Influence growth proxy (slope / mean, i.e. growth rate)
    """
    records = []
    
    for country, group in df.groupby(country_col):
        group = group.sort_values(year_col)
        years = group[year_col].values
        n = len(years)
        
        if n < 4:  # skip if insufficient history
            continue
        
        row = {country_col: country}
        
        # --- per-indicator slopes ---
        for ind in indicators:
            vals = group[ind].values
            if np.isnan(vals).mean() > 0.4:  # skip if >40% missing
                row[f'{ind}_slope'] = np.nan
                row[f'{ind}_r2'] = np.nan
                continue
            
            mask = ~np.isnan(vals)
            if mask.sum() < 4:
                row[f'{ind}_slope'] = np.nan
                row[f'{ind}_r2'] = np.nan
                continue
            
            slope, intercept, r, p, se = stats.linregress(years[mask], vals[mask])
            row[f'{ind}_slope'] = slope
            row[f'{ind}_r2'] = r**2
        
        # --- composite score trend ---
        if score_col in group.columns:
            scores = group[score_col].values
            mask = ~np.isnan(scores)
            if mask.sum() >= 4:
                slope, intercept, r, p, se = stats.linregress(years[mask], scores[mask])
                fitted = intercept + slope * years[mask]
                residuals = scores[mask] - fitted
                
                row['score_slope'] = slope
                row['score_r2'] = r**2
                row['score_volatility'] = np.std(residuals)
                row['score_mean'] = np.nanmean(scores)
                # Influence growth: normalised slope
                row['influence_growth'] = slope / (row['score_mean'] + 1e-6)
                # Regime classification (rising / stable / declining)
                if slope > 0.5 and r**2 > 0.3:
                    row['regime'] = 'rising'
                elif slope < -0.5 and r**2 > 0.3:
                    row['regime'] = 'declining'
                else:
                    row['regime'] = 'stable'
        
        # --- recent momentum (last 5y vs previous 5y) ---
        if n >= 8:
            recent = group.tail(5)[score_col].mean()
            prior  = group.iloc[-10:-5][score_col].mean() if n >= 10 else group.head(5)[score_col].mean()
            row['momentum_5y'] = recent - prior
        else:
            row['momentum_5y'] = np.nan
        
        records.append(row)
    
    trend_df = pd.DataFrame(records).set_index(country_col)
    print(f"[Trend] Computed trend features for {len(trend_df)} countries.")
    return trend_df


# ─── 2. COUNTRY SIMILARITY VECTORS ─────────────────────────────────────────

def build_country_embeddings(df: pd.DataFrame,
                              indicators: list,
                              country_col='iso3',
                              year_col='year',
                              recent_years=5) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a fixed-length embedding vector per country using:
    - Mean of each indicator over last N years
    - Standard deviation (intra-country variation)
    - Trend slope of each indicator
    
    Returns (embedding_df, embedding_matrix)
    """
    latest_year = df[year_col].max()
    cutoff = latest_year - recent_years
    recent = df[df[year_col] > cutoff]
    
    embed_records = []
    
    for country, grp in recent.groupby(country_col):
        vec = {country_col: country}
        for ind in indicators:
            vec[f'{ind}_mean'] = grp[ind].mean()
            vec[f'{ind}_std']  = grp[ind].std()
        embed_records.append(vec)
    
    embed_df = pd.DataFrame(embed_records).set_index(country_col)
    
    # fill NaN with column means (so FAISS doesn't break)
    embed_df = embed_df.fillna(embed_df.mean())
    
    # L2-normalise for cosine similarity via FAISS inner product
    scaler = StandardScaler()
    matrix = scaler.fit_transform(embed_df.values).astype('float32')

    # Make memory contiguous for FAISS
    matrix = np.ascontiguousarray(matrix)

    faiss.normalize_L2(matrix)
    
    return embed_df, matrix, scaler


def build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS index (inner product = cosine on normalised vectors)."""
    d = matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(matrix)
    print(f"[FAISS] Index built: {index.ntotal} vectors, dim={d}")
    return index


def find_peer_nations(query_country: str,
                      embed_df: pd.DataFrame,
                      matrix: np.ndarray,
                      index: faiss.IndexFlatIP,
                      k: int = 10) -> pd.DataFrame:
    """Return top-k most similar countries to a given country."""
    countries = list(embed_df.index)
    if query_country not in countries:
        raise ValueError(f"{query_country} not in embedding index")
    
    q_idx = countries.index(query_country)
    q_vec = matrix[q_idx:q_idx+1]
    
    scores, indices = index.search(q_vec, k+1)  # +1 to exclude self
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        peer = countries[idx]
        if peer != query_country:
            results.append({'country': peer, 'similarity': float(score)})
    
    return pd.DataFrame(results[:k])


def save_phase1_artifacts(trend_df, embed_df, matrix, index, scaler, out_dir='artifacts'):
    os.makedirs(out_dir, exist_ok=True)
    trend_df.to_parquet(f'{out_dir}/trend_features.parquet')
    embed_df.to_parquet(f'{out_dir}/country_embeddings.parquet')
    np.save(f'{out_dir}/embedding_matrix.npy', matrix)
    faiss.write_index(index, f'{out_dir}/faiss_index.bin')
    with open(f'{out_dir}/embedding_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[Phase 1] Artifacts saved to '{out_dir}/'")


# ─── USAGE EXAMPLE ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Use the available master panel file from output directory
    df = pd.read_parquet('output/master_phase_b.parquet')
    
    INDICATORS = [
        'tourist_arrivals',
        'unesco_total_sites',
        'unesco_cultural_sites',

        'internet_pct_sp',
        'rnd_pct_gdp_sp',
        'sci_journal_articles',
        'ai_publications',
        'hightech_exports_pct',

        'fh_combined_score',
        'trade_pct_gdp',
        'investment_freedom',

        'govt_integrity',
        'judicial_effectiveness',
        'property_rights',
        'business_freedom',

        'life_expectancy',
        'tertiary_enroll_pct',
        'physicians_per_1k',
        'infant_mortality'
    ]
    
    # 1. Trend features
    trend_df = compute_trend_features(
        df,
        INDICATORS,
        score_col='soft_power_composite_raw'
    )
    
    # 2. Country embeddings + FAISS
    embed_df, matrix, scaler = build_country_embeddings(df, INDICATORS)
    index = build_faiss_index(matrix)
    
    # 3. Quick test
    peers = find_peer_nations('IND', embed_df, matrix, index, k=5)
    print("Top peers for India:\n", peers)
    peers = find_peer_nations('CHN', embed_df, matrix, index, k=5)
    print("Top peers for China:\n", peers)
    
    save_phase1_artifacts(trend_df, embed_df, matrix, index, scaler)
