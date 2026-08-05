"""
Peer Similarity Explainer
Answers: WHY is country A similar to country B?

For any pair of countries, this decomposes their similarity into:
  1. Which dimensions drive the similarity (cultural, innovation, etc.)
  2. Which specific indicators are closest / most different
  3. A plain-English narrative summary

Run after phase_c_complete.py
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

COUNTRY_COL = 'iso3'
YEAR_COL    = 'year'

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────────────────

master   = pd.read_parquet('output/master_phase_b.parquet')
preds    = pd.read_parquet('output/soft_power_predictions.parquet')
embed_df = pd.read_parquet('output/country_embeddings.parquet')

with open('output/artifacts/embedding_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

matrix = np.load('output/artifacts/embedding_matrix.npy').astype('float32')
index  = faiss.read_index('output/artifacts/faiss_index.bin')

COUNTRIES = list(embed_df.index)

# Dimension groupings — must match what you used in phase_b
DIMENSIONS = {
    'Cultural':      ['tourist_arrivals_mean', 'unesco_total_sites_mean',
                      'unesco_cultural_sites_mean'],
    'Innovation':    ['internet_pct_sp_mean', 'rnd_pct_gdp_sp_mean',
                      'sci_journal_articles_mean', 'ai_publications_mean',
                      'hightech_exports_pct_mean'],
    'Political':     ['fh_combined_score_mean', 'fh_status_num_mean',
                      'fh_total_score_mean', 'trade_pct_gdp_mean',
                      'investment_freedom_mean'],
    'Institutional': ['govt_integrity_mean', 'judicial_effectiveness_mean',
                      'property_rights_mean', 'business_freedom_mean'],
    'Human Dev':     ['life_expectancy_mean', 'tertiary_enroll_pct_mean',
                      'physicians_per_1k_mean', 'infant_mortality_mean'],
}

# Keep only columns that actually exist in embed_df
DIMENSIONS = {
    dim: [c for c in cols if c in embed_df.columns]
    for dim, cols in DIMENSIONS.items()
}

# Nice display names for indicators
INDICATOR_LABELS = {
    'tourist_arrivals_mean':        'Tourist arrivals',
    'unesco_total_sites_mean':      'UNESCO total sites',
    'unesco_cultural_sites_mean':   'UNESCO cultural sites',
    'internet_pct_sp_mean':         'Internet penetration',
    'rnd_pct_gdp_sp_mean':          'R&D spend (% GDP)',
    'sci_journal_articles_mean':    'Scientific publications',
    'ai_publications_mean':         'AI publications',
    'hightech_exports_pct_mean':    'High-tech exports',
    'ict_patents_mean':             'ICT patents',
    'rd_researchers_per_mil_mean':  'Researchers per million',
    'fh_combined_score_mean':       'Freedom House score',
    'fh_status_num_mean':           'FH status',
    'fh_total_score_mean':          'FH total score',
    'trade_pct_gdp_mean':           'Trade openness',
    'investment_freedom_mean':      'Investment freedom',
    'govt_integrity_mean':          'Government integrity',
    'judicial_effectiveness_mean':  'Judicial effectiveness',
    'property_rights_mean':         'Property rights',
    'business_freedom_mean':        'Business freedom',
    'life_expectancy_mean':         'Life expectancy',
    'tertiary_enroll_pct_mean':     'Tertiary enrollment',
    'physicians_per_1k_mean':       'Physicians per 1k',
    'infant_mortality_mean':        'Infant mortality',
}


# ─── CORE FUNCTIONS ──────────────────────────────────────────────────────────

def get_country_vector(iso3: str) -> np.ndarray:
    if iso3 not in COUNTRIES:
        raise ValueError(f"{iso3} not in embedding index")
    return matrix[COUNTRIES.index(iso3)]


def get_peers(iso3: str, k: int = 5) -> pd.DataFrame:
    q_vec = get_country_vector(iso3).reshape(1, -1)
    scores, idxs = index.search(q_vec, k + 1)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        peer = COUNTRIES[idx]
        if peer != iso3:
            results.append({COUNTRY_COL: peer,
                            'similarity': round(float(score), 4)})
    return pd.DataFrame(results[:k])


def explain_similarity(iso3_a: str, iso3_b: str) -> dict:
    """
    Decompose WHY two countries are similar.
    Returns a dict with dimension-level and indicator-level breakdown.
    """
    if iso3_a not in COUNTRIES or iso3_b not in COUNTRIES:
        raise ValueError(f"One of {iso3_a}, {iso3_b} not in index")

    # Raw (unscaled) values for interpretability
    vals_a = embed_df.loc[iso3_a]
    vals_b = embed_df.loc[iso3_b]

    # ── Overall cosine similarity ─────────────────────────────────────
    vec_a = get_country_vector(iso3_a)
    vec_b = get_country_vector(iso3_b)
    overall_sim = float(np.dot(vec_a, vec_b))  # already L2-normalised

    # ── Dimension-level similarity ────────────────────────────────────
    dim_similarities = {}
    for dim, cols in DIMENSIONS.items():
        cols_exist = [c for c in cols if c in embed_df.columns]
        if not cols_exist:
            continue
        # Scale these columns the same way the full scaler did
        # (use mean values from embed_df as proxy for what scaler learned)
        sub_a = vals_a[cols_exist].values.astype(float)
        sub_b = vals_b[cols_exist].values.astype(float)

        # Cosine similarity on raw values (directional agreement)
        norm_a = np.linalg.norm(sub_a)
        norm_b = np.linalg.norm(sub_b)
        if norm_a > 0 and norm_b > 0:
            cos_sim = np.dot(sub_a, sub_b) / (norm_a * norm_b)
        else:
            cos_sim = 0.0

        # Also compute absolute difference per indicator
        diffs = {c: abs(float(vals_a[c]) - float(vals_b[c]))
                 if not (np.isnan(vals_a[c]) or np.isnan(vals_b[c]))
                 else np.nan
                 for c in cols_exist}

        dim_similarities[dim] = {
            'cosine_sim': round(float(cos_sim), 3),
            'indicator_diffs': diffs,
        }

    # ── Most similar indicators (smallest standardised difference) ────
    all_mean_cols = [c for c in embed_df.columns if c.endswith('_mean')]
    diffs_all = {}
    for col in all_mean_cols:
        a_val = vals_a.get(col, np.nan)
        b_val = vals_b.get(col, np.nan)
        if pd.isna(a_val) or pd.isna(b_val):
            continue
        # Normalise by column std across all countries for comparability
        col_std = embed_df[col].std()
        if col_std > 0:
            diffs_all[col] = abs(float(a_val) - float(b_val)) / col_std

    sorted_diffs = sorted(diffs_all.items(), key=lambda x: x[1])
    most_similar  = sorted_diffs[:5]   # closest in normalised space
    most_different = sorted_diffs[-5:] # furthest apart

    # ── Actual scores for context ─────────────────────────────────────
    score_a = preds[preds[COUNTRY_COL]==iso3_a]['score'].values
    score_b = preds[preds[COUNTRY_COL]==iso3_b]['score'].values
    rank_a  = preds[preds[COUNTRY_COL]==iso3_a]['global_rank'].values
    rank_b  = preds[preds[COUNTRY_COL]==iso3_b]['global_rank'].values

    return {
        'country_a':      iso3_a,
        'country_b':      iso3_b,
        'overall_sim':    round(overall_sim, 4),
        'score_a':        round(float(score_a[0]), 2) if len(score_a) else None,
        'score_b':        round(float(score_b[0]), 2) if len(score_b) else None,
        'rank_a':         int(rank_a[0]) if len(rank_a) else None,
        'rank_b':         int(rank_b[0]) if len(rank_b) else None,
        'dim_similarities': dim_similarities,
        'most_similar_indicators':   most_similar,
        'most_different_indicators': most_different,
    }


def print_similarity_report(iso3_a: str, iso3_b: str):
    """Pretty-print a full similarity explanation."""
    result = explain_similarity(iso3_a, iso3_b)

    name_a = preds[preds[COUNTRY_COL]==iso3_a]['canonical'].values
    name_b = preds[preds[COUNTRY_COL]==iso3_b]['canonical'].values
    name_a = name_a[0] if len(name_a) else iso3_a
    name_b = name_b[0] if len(name_b) else iso3_b

    print(f"\n{'═'*65}")
    print(f"  SIMILARITY REPORT: {name_a} vs {name_b}")
    print(f"{'═'*65}")
    print(f"  Overall similarity : {result['overall_sim']:.3f}  (1.0 = identical)")
    print(f"  {name_a:<25} score={result['score_a']}  rank=#{result['rank_a']}")
    print(f"  {name_b:<25} score={result['score_b']}  rank=#{result['rank_b']}")

    print(f"\n  ── Similarity by dimension ──────────────────────────────")
    print(f"  {'Dimension':<18} {'Cosine sim':>12}  {'Interpretation'}")
    print(f"  {'-'*60}")
    for dim, data in result['dim_similarities'].items():
        sim = data['cosine_sim']
        if sim > 0.95:   interp = "near-identical profiles"
        elif sim > 0.85: interp = "very similar"
        elif sim > 0.70: interp = "broadly similar"
        elif sim > 0.50: interp = "partly similar"
        else:            interp = "quite different"
        print(f"  {dim:<18} {sim:>12.3f}  {interp}")

    print(f"\n  ── Indicators where they are MOST alike ─────────────────")
    for col, diff in result['most_similar_indicators']:
        label = INDICATOR_LABELS.get(col, col.replace('_mean','').replace('_',' '))
        a_val = embed_df.loc[iso3_a, col] if col in embed_df.columns else np.nan
        b_val = embed_df.loc[iso3_b, col] if col in embed_df.columns else np.nan
        print(f"  {label:<35}  {name_a[:10]}: {a_val:>8.2f}  "
              f"{name_b[:10]}: {b_val:>8.2f}  (diff={diff:.3f}σ)")

    print(f"\n  ── Indicators where they are MOST different ──────────────")
    for col, diff in reversed(result['most_different_indicators']):
        label = INDICATOR_LABELS.get(col, col.replace('_mean','').replace('_',' '))
        a_val = embed_df.loc[iso3_a, col] if col in embed_df.columns else np.nan
        b_val = embed_df.loc[iso3_b, col] if col in embed_df.columns else np.nan
        print(f"  {label:<35}  {name_a[:10]}: {a_val:>8.2f}  "
              f"{name_b[:10]}: {b_val:>8.2f}  (diff={diff:.3f}σ)")

    # ── Plain-English narrative ───────────────────────────────────────
    print(f"\n  ── Why they are similar (plain English) ─────────────────")

    # Find the dimensions with highest similarity
    top_dims = sorted(result['dim_similarities'].items(),
                      key=lambda x: x[1]['cosine_sim'], reverse=True)
    top_dim_names = [d[0] for d in top_dims[:2]]

    # Find the most different dimension
    bot_dim = top_dims[-1][0]
    bot_sim = top_dims[-1][1]['cosine_sim']

    most_alike_label  = INDICATOR_LABELS.get(
        result['most_similar_indicators'][0][0],
        result['most_similar_indicators'][0][0].replace('_mean',''))
    most_differ_label = INDICATOR_LABELS.get(
        result['most_different_indicators'][-1][0],
        result['most_different_indicators'][-1][0].replace('_mean',''))

    score_diff = abs((result['score_a'] or 0) - (result['score_b'] or 0))
    rank_diff  = abs((result['rank_a'] or 0) - (result['rank_b'] or 0))

    print(f"""
  {name_a} and {name_b} are structurally similar (cosine={result['overall_sim']:.3f}),
  meaning they occupy a comparable position in the global soft power
  capacity landscape despite different geopolitical roles.

  Their similarity is strongest in {top_dim_names[0]} and {top_dim_names[1]}
  dimensions, where their indicator profiles are nearly aligned.
  The single indicator where they are most alike is {most_alike_label}.

  They diverge most on {most_differ_label} and in the {bot_dim}
  dimension (similarity={bot_sim:.3f}), which explains why their overall
  soft power scores differ by {score_diff:.1f} points (ranks #{result['rank_a']} vs #{result['rank_b']}).

  In peer-nation analysis, this means {name_b} serves as a useful
  benchmark for {name_a}: policies that improved {name_b}'s
  {bot_dim.lower()} performance may be transferable to {name_a}.
""")

    return result


def full_peer_analysis(iso3: str, k: int = 5):
    """Get peers + explain each one."""
    peers = get_peers(iso3, k=k)
    print(f"\n{'═'*65}")
    print(f"  PEER ANALYSIS FOR {iso3}")
    print(f"{'═'*65}")
    print(f"  Top {k} peers:\n")
    print(peers.to_string(index=False))

    for _, row in peers.iterrows():
        print_similarity_report(iso3, row[COUNTRY_COL])


# ─── RUN EXAMPLES ────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Example 1: Full peer analysis for India
    full_peer_analysis('IND', k=3)

    # Example 2: Specific pair — India vs China
    print_similarity_report('IND', 'CHN')

    # Example 3: USA vs Japan (top peers from your output)
    print_similarity_report('USA', 'JPN')

    # Example 4: You can query any pair interactively
    print("\n─── Interactive query ───")
    print("Edit the iso3 codes below to analyse any country pair:\n")

    # Add any country pair you want to explain
    pairs_to_explain = [
        ('IND', 'MEX'),   # India vs Mexico
        ('CHN', 'DEU'),   # China vs Germany
        ('GBR', 'FRA'),   # UK vs France
    ]
    for a, b in pairs_to_explain:
        print_similarity_report(a, b)