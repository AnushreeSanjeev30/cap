"""
Phase 3: Structural Causal Modeling (SCM)
Uses DoWhy + CausalNex to answer:
  - "What caused India's soft power to rise between 2010–2020?"
  - "If press_freedom improved by 10 points, what happens to soft power?"
  - "Which indicators have the highest causal effect on the composite score?"

Install: pip install dowhy econml causallearn
"""

import numpy as np
import pandas as pd
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ─── 1. DEFINE THE CAUSAL GRAPH ─────────────────────────────────────────────

def build_soft_power_dag(indicators: list, score_col='pca_score') -> nx.DiGraph:
    """
    Construct a Directed Acyclic Graph (DAG) encoding assumed causal structure.
    
    Theory-driven edges (from soft power literature):
    - Governance dims → Institutional score → Composite
    - Innovation dims → Tech score → Composite  
    - Cultural dims → Culture score → Composite
    - HDI dims → Human Dev score → Composite
    - All dimension scores → pca_score
    
    You can refine this with CausalLearn's PC algorithm (data-driven discovery).
    """
    G = nx.DiGraph()
    
    # Dimension groupings (adjust to your 23 KPIs)
    DIMS = {
        'culture':       ['tourism_inflow', 'unesco_sites', 'cultural_exports'],
        'innovation':    ['patent_applications', 'internet_users', 'rd_expenditure'],
        'politics':      ['press_freedom', 'democracy_index', 'gov_effectiveness'],
        'institutions':  ['rule_of_law', 'corruption_control', 'political_stability'],
        'human_dev':     ['hdi', 'tertiary_education', 'life_expectancy'],
    }
    
    DIM_NODES = {
        'culture':      'dim_culture',
        'innovation':   'dim_innovation',
        'politics':     'dim_politics',
        'institutions': 'dim_institutions',
        'human_dev':    'dim_human_dev',
    }
    
    # Indicator → dimension
    for dim, inds in DIMS.items():
        dim_node = DIM_NODES[dim]
        for ind in inds:
            if ind in indicators:
                G.add_edge(ind, dim_node)
    
    # Dimension → composite
    for dim_node in DIM_NODES.values():
        G.add_edge(dim_node, score_col)
    
    # Cross-dimension edges (theory-driven)
    G.add_edge('dim_institutions', 'dim_politics')     # institutions enable political freedom
    G.add_edge('dim_human_dev',    'dim_innovation')   # educated population drives innovation
    G.add_edge('dim_politics',     'dim_culture')      # open societies export more culture
    
    return G


# ─── 2. AUTOMATED CAUSAL DISCOVERY (PC ALGORITHM) ───────────────────────────

def discover_causal_structure(model_df: pd.DataFrame,
                               feature_cols: list,
                               alpha: float = 0.05) -> nx.DiGraph:
    """
    Data-driven DAG discovery using the PC algorithm (constraint-based).
    Use this to validate or augment your theory-driven DAG.
    
    Install: pip install causallearn
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except ImportError:
        raise ImportError("pip install causallearn")
    
    data = model_df[feature_cols].dropna().values.astype(float)
    
    cg = pc(data, alpha=alpha, ci_test=fisherz)
    
    # Convert to networkx DiGraph
    G = nx.DiGraph()
    nodes = feature_cols
    adj = cg.G.graph
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adj[i, j] == -1 and adj[j, i] == 1:  # directed i→j
                G.add_edge(nodes[i], nodes[j])
    
    print(f"[PC algorithm] Discovered {G.number_of_edges()} edges among {G.number_of_nodes()} nodes")
    return G


# ─── 3. CAUSAL EFFECT ESTIMATION ────────────────────────────────────────────

def estimate_causal_effects(model_df: pd.DataFrame,
                              dag: nx.DiGraph,
                              treatment_vars: list,
                              outcome_var: str = 'pca_score') -> pd.DataFrame:
    """
    Use DoWhy to estimate ATE (Average Treatment Effect) of each indicator
    on the soft power composite score.
    
    Returns a DataFrame with estimated causal effects and p-values.
    """
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        raise ImportError("pip install dowhy econml")
    
    results = []
    data = model_df[list(dag.nodes) + [outcome_var]].dropna()
    
    # Convert nx DAG to DOT string for DoWhy
    dot_str = 'digraph { '
    for u, v in dag.edges():
        dot_str += f'"{u}" -> "{v}"; '
    dot_str += '}'
    
    for treatment in treatment_vars:
        if treatment not in data.columns:
            continue
        try:
            causal_model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome_var,
                graph=dot_str
            )
            
            identified_estimand = causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )
            
            # Linear regression estimator (fast; swap for DML for non-linear)
            estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                confidence_intervals=True,
                test_significance=True
            )
            
            results.append({
                'treatment':    treatment,
                'ate':          float(estimate.value),
                'ci_lower':     float(estimate.get_confidence_intervals()[0]),
                'ci_upper':     float(estimate.get_confidence_intervals()[1]),
                'p_value':      float(estimate.test_stat_significance()['p_value']),
            })
            
        except Exception as e:
            print(f"[SCM] {treatment}: {e}")
    
    effects_df = pd.DataFrame(results).sort_values('ate', key=abs, ascending=False)
    return effects_df


# ─── 4. INTERVENTION SIMULATION ─────────────────────────────────────────────

def simulate_intervention(country: str,
                           treatment_var: str,
                           delta: float,
                           model_df: pd.DataFrame,
                           xgb_model,
                           feature_cols: list,
                           country_col='country_iso3',
                           year_col='year') -> dict:
    """
    Counterfactual: what would the soft power score be if 
    `treatment_var` increased by `delta` for this country?
    
    Uses the XGBoost model for estimation (DoWhy's refutation would be ideal but slow).
    """
    latest = (model_df[model_df[country_col]==country]
              .sort_values(year_col).iloc[[-1]])
    
    if len(latest) == 0:
        raise ValueError(f"Country {country} not found")
    
    X_base = latest[feature_cols].fillna(0).copy()
    X_counter = X_base.copy()
    
    # Apply intervention
    if treatment_var in X_counter.columns:
        X_counter[treatment_var] = X_counter[treatment_var] + delta
    elif f'{treatment_var}_lag1' in X_counter.columns:
        X_counter[f'{treatment_var}_lag1'] = X_counter[f'{treatment_var}_lag1'] + delta
    
    baseline    = xgb_model.predict(X_base)
    counterfact = xgb_model.predict(X_counter)
    
    return {
        'country':       country,
        'treatment':     treatment_var,
        'delta':         delta,
        'score_base':    float(baseline['score'].iloc[0]),
        'score_counter': float(counterfact['score'].iloc[0]),
        'effect':        float(counterfact['score'].iloc[0] - baseline['score'].iloc[0]),
    }


# ─── 5. REFUTATION TESTS ────────────────────────────────────────────────────

def run_refutations(causal_model, identified_estimand, estimate) -> dict:
    """
    Robustness checks for a causal estimate:
    1. Placebo treatment (should give ~0 effect)
    2. Random common cause (effect should be stable)
    3. Data subset (effect should be consistent)
    """
    results = {}
    
    # Placebo
    ref_placebo = causal_model.refute_estimate(
        identified_estimand, estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute"
    )
    results['placebo_effect'] = float(ref_placebo.new_effect)
    results['placebo_p']      = float(ref_placebo.refutation_result['p_value'])
    
    # Random common cause
    ref_rcc = causal_model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause"
    )
    results['rcc_new_effect'] = float(ref_rcc.new_effect)
    
    return results


# ─── USAGE ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import pickle
    
    master_df  = pd.read_parquet('data/master_panel.parquet')
    trend_df   = pd.read_parquet('artifacts/trend_features.parquet').reset_index()
    
    INDICATORS = [...]  # your 23 KPI names
    
    # Build causal graph
    dag = build_soft_power_dag(INDICATORS)
    print(f"[DAG] {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    
    # Estimate causal effects of all indicators on composite score
    model_df = pd.read_parquet('artifacts/model_df.parquet')  # from phase 2
    effects  = estimate_causal_effects(model_df, dag, INDICATORS)
    print("\nTop causal drivers of soft power:")
    print(effects.head(10))
    effects.to_csv('artifacts/causal_effects.csv', index=False)
    
    # Counterfactual: India, 10-point improvement in press_freedom
    with open('artifacts/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    from archive.phase2_predictive_model import get_feature_cols
    feature_cols = get_feature_cols(model_df, INDICATORS)
    
    result = simulate_intervention(
        country='IND',
        treatment_var='press_freedom',
        delta=10.0,
        model_df=model_df,
        xgb_model=xgb_model,
        feature_cols=feature_cols
    )
    print(f"\nIntervention result:\n{result}")
