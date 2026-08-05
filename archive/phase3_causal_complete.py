import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, json

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GIAS · Soft Power Index",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── COLOUR SYSTEM ─────────────────────────────────────────────────────────────
BG       = "#080B14"
BG2      = "#0D1220"
BG3      = "#111827"
PANEL    = "#0F1829"
BORDER   = "#1C2940"
GREEN    = "#00FF9C"
GREEN2   = "#00CC7A"
BLUE     = "#4B9FFF"
RED      = "#FF4B6E"
AMBER    = "#FFB800"
PURPLE   = "#A855F7"
GREY     = "#4A5568"
GREY2    = "#718096"
WHITE    = "#E2E8F0"

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, .stApp {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif;
    color: {WHITE};
}}

/* Hide streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none; }}
div[data-testid="stToolbar"] {{ display: none; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG2}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {GREEN}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {BG2} !important;
    border-bottom: 1px solid {BORDER} !important;
    gap: 0 !important;
    padding: 0 24px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {GREY2} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    padding: 12px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase !important;
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: {GREEN} !important;
    border-bottom: 2px solid {GREEN} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: {BG} !important;
    padding: 0 !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    color: {WHITE} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 2px !important;
}}

/* Text input */
.stTextInput > div > div > input {{
    background: {BG2} !important;
    border: 1px solid {BORDER} !important;
    color: {GREEN} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 2px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {GREEN} !important;
    box-shadow: 0 0 0 1px {GREEN}40 !important;
}}

/* Button */
.stButton > button {{
    background: transparent !important;
    border: 1px solid {GREEN} !important;
    color: {GREEN} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 8px 20px !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    background: {GREEN}15 !important;
    box-shadow: 0 0 12px {GREEN}30 !important;
}}

/* Spinner */
.stSpinner > div {{ border-top-color: {GREEN} !important; }}

/* Metric */
[data-testid="stMetric"] {{
    background: {PANEL} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 2px !important;
    padding: 12px 16px !important;
}}
[data-testid="stMetricLabel"] {{
    color: {GREY2} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] {{
    color: {GREEN} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
}}

.panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 16px;
    margin-bottom: 12px;
    position: relative;
}}
.panel-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: {GREY};
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.panel-eyebrow::before {{
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: {GREEN};
    border-radius: 50%;
    box-shadow: 0 0 6px {GREEN};
    flex-shrink: 0;
}}
.panel-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: {WHITE};
    letter-spacing: 0.04em;
    margin-bottom: 12px;
}}

/* Top bar */
.topbar {{
    background: {BG2};
    border-bottom: 1px solid {BORDER};
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 48px;
    position: sticky;
    top: 0;
    z-index: 999;
}}
.topbar-logo {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: {GREEN};
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.topbar-status {{
    display: flex;
    align-items: center;
    gap: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {GREY2};
    letter-spacing: 0.08em;
}}
.live-dot {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: {GREEN};
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
}}
.live-dot::before {{
    content: '';
    display: inline-block;
    width: 7px; height: 7px;
    background: {GREEN};
    border-radius: 50%;
    animation: pulse 1.8s ease-in-out infinite;
    box-shadow: 0 0 8px {GREEN};
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.85); }}
}}
.score-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    color: {GREEN};
    line-height: 1;
    text-shadow: 0 0 20px {GREEN}60;
}}
.rank-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: {AMBER};
    letter-spacing: 0.08em;
}}
.regime-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    margin-top: 6px;
}}
.kpi-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid {BORDER};
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}}
.kpi-row:last-child {{ border-bottom: none; }}
.kpi-name {{ color: {GREY2}; text-transform: uppercase; letter-spacing: 0.08em; font-size: 10px; }}
.kpi-bar-wrap {{ flex: 1; margin: 0 12px; height: 4px; background: {BORDER}; border-radius: 2px; }}
.kpi-bar {{ height: 4px; border-radius: 2px; }}
.kpi-val {{ color: {WHITE}; min-width: 40px; text-align: right; }}

.chat-msg-user {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 2px 12px 12px 12px;
    padding: 10px 14px;
    margin: 8px 0;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: {BLUE};
    max-width: 80%;
}}
.chat-msg-agent {{
    background: {PANEL};
    border: 1px solid {GREEN}30;
    border-radius: 12px 2px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px auto;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: {WHITE};
    line-height: 1.6;
    max-width: 85%;
}}
.chat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.signal-bar {{
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 20px;
}}
.signal-bar span {{
    display: inline-block;
    width: 3px;
    background: {GREEN};
    border-radius: 1px;
    opacity: 0.3;
}}
.signal-bar span.active {{ opacity: 1; box-shadow: 0 0 4px {GREEN}; }}
</style>

<!-- TOP BAR -->
<div class="topbar">
  <div class="topbar-logo">⬡ GIAS · Geopolitical Intelligence Analysis System</div>
  <div class="topbar-status">
    <span class="live-dot">LIVE</span>
    <span>195 NATIONS · 19 KPIs · 2000–2024</span>
    <span>ENSEMBLE MODEL · LightGBM PRIMARY</span>
    <span style="color:{AMBER}">CLASSIFICATION: UNCLASSIFIED // OPEN SOURCE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "output"
    d = {}
    loaders = {
        "preds":       (f"{base}/ensemble_predictions.parquet", "parquet"),
        "forecast":    (f"{base}/kalman_forecast_5yr.csv",      "csv"),
        "regimes":     (f"{base}/kalman_regimes.csv",           "csv"),
        "kalman":      (f"{base}/kalman_results.csv",           "csv"),
        "shap_global": (f"{base}/shap_global.csv",              "csv"),
        "shap_country":(f"{base}/shap_country.csv",             "csv"),
        "counter":     (f"{base}/counterfactual_results.csv",   "csv"),
        "model_cmp":   (f"{base}/model_comparison.csv",         "csv"),
        "summary":     (f"{base}/kalman_summary.csv",           "csv"),
    }
    for key, (path, fmt) in loaders.items():
        if os.path.exists(path):
            d[key] = pd.read_parquet(path) if fmt=="parquet" else pd.read_csv(path)
        else:
            d[key] = pd.DataFrame()
    return d

D = load_data()

def iso3_to_name(preds):
    if "country_name" in preds.columns:
        m = dict(zip(preds["iso3"], preds["country_name"]))
    else:
        m = dict(zip(preds["iso3"], preds["iso3"]))
    return m

ISOMAP = iso3_to_name(D["preds"]) if not D["preds"].empty else {}

REGIME_COLORS = {
    "Ascendant":     GREEN,
    "Volatile Riser":AMBER,
    "Stable Power":  BLUE,
    "Coasting":      GREY2,
    "Declining":     RED,
    "Fragile":       "#FF6B35",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=WHITE, size=10),
    margin=dict(l=8, r=8, t=32, b=8),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1),
)

def styled_fig(**kw):
    fig = go.Figure(**kw)
    fig.update_layout(**PLOT_LAYOUT)
    return fig

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "◈  GLOBAL OVERVIEW",
    "◉  COUNTRY INTEL",
    "△  REGIME MAP",
    "◎  MODEL SIGNALS",
    "⌘  AGENT INTERFACE",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div style='padding:16px 24px 0'>", unsafe_allow_html=True)

    preds = D["preds"]
    if preds.empty:
        st.warning("No prediction data found. Run phase_d_multi_model.py first.")
    else:
        # ── KPI strip ────────────────────────────────────────────────────────
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            st.metric("NATIONS TRACKED", "195")
        with c2:
            st.metric("ACTIVE KPIs", "19")
        with c3:
            top = preds.loc[preds["ensemble_score"].idxmax()]
            st.metric("TOP RANKED", f"{ISOMAP.get(top['iso3'],top['iso3'])}", f"Score {top['ensemble_score']:.1f}")
        with c4:
            med = preds["ensemble_score"].median()
            st.metric("GLOBAL MEDIAN", f"{med:.1f}", "/ 100")
        with c5:
            if not D["regimes"].empty:
                top_regime = D["regimes"]["regime"].value_counts().index[0]
                st.metric("DOMINANT REGIME", top_regime)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── World choropleth ──────────────────────────────────────────────────
        st.markdown("""<div class='panel'>
            <div class='panel-eyebrow'>Global Soft Power Index · Ensemble Score</div>
            <div class='panel-title'>WORLD SOFT POWER MAP — 2024</div>""",
            unsafe_allow_html=True)

        fig_map = go.Figure(go.Choropleth(
            locations=preds["iso3"],
            z=preds["ensemble_score"],
            colorscale=[
                [0.0,  "#1a0a0a"],
                [0.2,  "#3d1515"],
                [0.4,  "#7a2d2d"],
                [0.55, "#c45c1a"],
                [0.7,  "#e8a020"],
                [0.85, "#60d080"],
                [1.0,  "#00FF9C"],
            ],
            zmin=0, zmax=100,
            marker_line_color=BORDER,
            marker_line_width=0.4,
            colorbar=dict(
                title=dict(text="SCORE", font=dict(color=GREY2, size=9,
                           family="JetBrains Mono, monospace")),
                thickness=8, len=0.6,
                tickfont=dict(color=GREY2, size=9, family="JetBrains Mono, monospace"),
                tickformat=".0f",
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER,
            ),
            customdata=preds[["iso3","ensemble_score","ensemble_rank"]].values if "ensemble_rank" in preds.columns else preds[["iso3","ensemble_score"]].values,
            hovertemplate="<b>%{location}</b><br>Score: %{z:.1f}<extra></extra>",
        ))
        fig_map.update_geos(
            showframe=False, showcoastlines=True,
            coastlinecolor=BORDER, coastlinewidth=0.5,
            showland=True, landcolor="#0D1220",
            showocean=True, oceancolor=BG,
            showlakes=False, showrivers=False,
            showcountries=True, countrycolor=BORDER,
            bgcolor="rgba(0,0,0,0)",
            projection_type="natural earth",
        )
        fig_map.update_layout(**PLOT_LAYOUT, height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Top/Bottom tables + Distribution ─────────────────────────────────
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("""<div class='panel'>
                <div class='panel-eyebrow'>Rankings · Ensemble Score</div>
                <div class='panel-title'>TOP 15 · SOFT POWER LEADERS</div>""",
                unsafe_allow_html=True)

            top15 = preds.nlargest(15, "ensemble_score").reset_index(drop=True)
            fig_top = go.Figure()
            colors_bar = [GREEN if i == 0 else BLUE if i < 3 else GREY2
                          for i in range(len(top15))]
            fig_top.add_trace(go.Bar(
                y=top15["iso3"][::-1],
                x=top15["ensemble_score"][::-1],
                orientation='h',
                marker=dict(
                    color=list(reversed([
                        GREEN if i==0 else BLUE if i<3 else GREY2
                        for i in range(len(top15))])),
                    line=dict(width=0),
                ),
                text=[f"{s:.1f}" for s in top15["ensemble_score"][::-1]],
                textposition="outside",
                textfont=dict(color=GREY2, size=9, family="JetBrains Mono, monospace"),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            fig_top.update_layout(**PLOT_LAYOUT, height=360,
                xaxis=dict(range=[60,100], gridcolor=BORDER, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=10, color=WHITE)),
                bargap=0.25,
            )
            st.plotly_chart(fig_top, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("""<div class='panel'>
                <div class='panel-eyebrow'>Score Distribution · All Nations</div>
                <div class='panel-title'>GLOBAL SCORE DISTRIBUTION</div>""",
                unsafe_allow_html=True)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=preds["ensemble_score"],
                nbinsx=30,
                marker=dict(
                    color=preds["ensemble_score"],
                    colorscale=[[0,"#3d1515"],[0.5,BLUE],[1.0,GREEN]],
                    line=dict(color=BG, width=0.5),
                ),
                hovertemplate="Score: %{x:.0f}–%{x:.0f}<br>Nations: %{y}<extra></extra>",
            ))
            # Add median line
            med_val = preds["ensemble_score"].median()
            fig_hist.add_vline(x=med_val, line_color=AMBER, line_width=1.5,
                               line_dash="dash",
                               annotation_text=f"MEDIAN {med_val:.1f}",
                               annotation_font=dict(color=AMBER, size=9,
                                                    family="JetBrains Mono, monospace"),
                               annotation_position="top right")
            fig_hist.update_layout(**PLOT_LAYOUT, height=200,
                bargap=0.05,
                xaxis=dict(title="Score", tickfont=dict(size=9)),
                yaxis=dict(title="Nations", tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Model agreement scatter
            st.markdown("<div class='panel-eyebrow' style='margin-top:8px'>Model Consensus · Agreement Across Ensemble</div>", unsafe_allow_html=True)
            if "xgboost_score" in preds.columns and "lightgbm_score" in preds.columns:
                preds2 = preds.copy()
                preds2["disagreement"] = (
                    preds2[["xgboost_score","randomforest_score","lightgbm_score"]].std(axis=1)
                )
                fig_ag = go.Figure()
                fig_ag.add_trace(go.Scatter(
                    x=preds2["ensemble_score"],
                    y=preds2["disagreement"],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=preds2["disagreement"],
                        colorscale=[[0,GREEN],[0.5,AMBER],[1.0,RED]],
                        opacity=0.7,
                        line=dict(width=0),
                    ),
                    text=preds2["iso3"],
                    hovertemplate="<b>%{text}</b><br>Score: %{x:.1f}<br>Model spread: ±%{y:.2f}<extra></extra>",
                ))
                fig_ag.update_layout(**PLOT_LAYOUT, height=140,
                    xaxis=dict(title="Ensemble Score", tickfont=dict(size=9)),
                    yaxis=dict(title="Model Spread (σ)", tickfont=dict(size=9)),
                    margin=dict(l=8,r=8,t=8,b=8),
                )
                st.plotly_chart(fig_ag, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COUNTRY INTEL
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div style='padding:16px 24px 0'>", unsafe_allow_html=True)

    preds = D["preds"]
    if preds.empty:
        st.warning("No data.")
    else:
        # Country selector
        country_options = sorted([
            f"{ISOMAP.get(iso, iso)} ({iso})"
            for iso in preds["iso3"].tolist()
        ])
        sel_raw = st.selectbox("SELECT NATION", country_options,
                               index=country_options.index("India (IND)")
                               if "India (IND)" in country_options else 0)
        sel_iso = sel_raw.split("(")[-1].replace(")", "").strip()

        row = preds[preds["iso3"] == sel_iso]
        if row.empty:
            st.warning(f"No data for {sel_iso}")
        else:
            row = row.iloc[0]

            # ── Score card ────────────────────────────────────────────────────
            reg_row = D["regimes"][D["regimes"]["iso3"]==sel_iso] if not D["regimes"].empty else pd.DataFrame()
            regime  = reg_row.iloc[0]["regime"] if not reg_row.empty else "Unknown"
            slope   = reg_row.iloc[0]["trend_slope"] if not reg_row.empty else 0
            rc      = REGIME_COLORS.get(regime, GREY2)

            c1,c2,c3,c4 = st.columns([1.2,1,1,1])
            with c1:
                st.markdown(f"""
                <div class='panel' style='border-color:{GREEN}30'>
                  <div class='panel-eyebrow'>Soft Power Score · 2024</div>
                  <div class='score-badge'>{row['ensemble_score']:.1f}</div>
                  <div class='rank-badge'>RANK #{int(row['ensemble_rank']) if 'ensemble_rank' in row else '—'} OF 195</div>
                  <div class='regime-tag' style='background:{rc}20;color:{rc};border:1px solid {rc}40'>
                    {regime}
                  </div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.metric("XGBoost", f"{row.get('xgboost_score',0):.2f}")
            with c3:
                st.metric("Random Forest", f"{row.get('randomforest_score',0):.2f}")
            with c4:
                st.metric("LightGBM", f"{row.get('lightgbm_score',0):.2f}")

            # ── Kalman history + 5yr forecast ─────────────────────────────────
            col_l, col_r = st.columns([1.6, 1])
            with col_l:
                st.markdown("""<div class='panel'>
                    <div class='panel-eyebrow'>Historical Trajectory + 5-Year Forecast · Kalman Smoothed</div>
                    <div class='panel-title'>SOFT POWER TRAJECTORY</div>""",
                    unsafe_allow_html=True)

                fig_traj = go.Figure()

                # Historical
                hist = D["kalman"][D["kalman"]["iso3"]==sel_iso].sort_values("year") if not D["kalman"].empty else pd.DataFrame()
                if not hist.empty:
                    fig_traj.add_trace(go.Scatter(
                        x=hist["year"], y=hist["kalman_score"],
                        mode="lines",
                        name="HISTORICAL (SMOOTHED)",
                        line=dict(color=BLUE, width=2),
                        hovertemplate="Year: %{x}<br>Score: %{y:.2f}<extra></extra>",
                    ))
                    # CI band
                    if "ci_upper_95" in hist.columns:
                        fig_traj.add_trace(go.Scatter(
                            x=pd.concat([hist["year"], hist["year"][::-1]]),
                            y=pd.concat([hist["ci_upper_95"], hist["ci_lower_95"][::-1]]),
                            fill="toself",
                            fillcolor=f"{BLUE}15",
                            line=dict(width=0),
                            name="95% CI",
                            hoverinfo="skip",
                        ))
                    # Raw dots
                    fig_traj.add_trace(go.Scatter(
                        x=hist["year"], y=hist["raw_score"],
                        mode="markers",
                        name="RAW OBSERVATIONS",
                        marker=dict(size=3, color=GREY2, opacity=0.6),
                        hovertemplate="Year: %{x}<br>Raw: %{y:.2f}<extra></extra>",
                    ))

                # Forecast
                fc = D["forecast"][D["forecast"]["iso3"]==sel_iso] if not D["forecast"].empty else pd.DataFrame()
                if not fc.empty:
                    # Connect line
                    if not hist.empty:
                        last_hist_yr  = hist["year"].max()
                        last_hist_sc  = hist[hist["year"]==last_hist_yr]["kalman_score"].values[0]
                        first_fc_yr   = fc["forecast_year"].min()
                        first_fc_sc   = fc[fc["forecast_year"]==first_fc_yr]["forecast_score"].values[0]
                        fig_traj.add_trace(go.Scatter(
                            x=[last_hist_yr, first_fc_yr],
                            y=[last_hist_sc, first_fc_sc],
                            mode="lines",
                            line=dict(color=GREEN, width=1.5, dash="dot"),
                            showlegend=False, hoverinfo="skip",
                        ))
                    # 95% CI band
                    fig_traj.add_trace(go.Scatter(
                        x=pd.concat([fc["forecast_year"], fc["forecast_year"][::-1]]),
                        y=pd.concat([fc["ci_upper_95"], fc["ci_lower_95"][::-1]]),
                        fill="toself",
                        fillcolor=f"{GREEN}12",
                        line=dict(width=0),
                        name="95% FORECAST CI",
                        hoverinfo="skip",
                    ))
                    # 80% CI band
                    fig_traj.add_trace(go.Scatter(
                        x=pd.concat([fc["forecast_year"], fc["forecast_year"][::-1]]),
                        y=pd.concat([fc["ci_upper_80"], fc["ci_lower_80"][::-1]]),
                        fill="toself",
                        fillcolor=f"{GREEN}20",
                        line=dict(width=0),
                        name="80% FORECAST CI",
                        hoverinfo="skip",
                    ))
                    fig_traj.add_trace(go.Scatter(
                        x=fc["forecast_year"], y=fc["forecast_score"],
                        mode="lines+markers",
                        name="5-YR FORECAST",
                        line=dict(color=GREEN, width=2.5, dash="dash"),
                        marker=dict(size=6, color=GREEN,
                                    line=dict(color=BG, width=2)),
                        hovertemplate="Year: %{x}<br>Forecast: %{y:.2f}<extra></extra>",
                    ))
                    # Add vertical "NOW" line
                    fig_traj.add_vline(
                        x=2024, line_color=AMBER,
                        line_width=1, line_dash="dot",
                        annotation_text="NOW",
                        annotation_font=dict(color=AMBER, size=9,
                                             family="JetBrains Mono, monospace"),
                        annotation_position="top",
                    )

                fig_traj.update_layout(**PLOT_LAYOUT, height=320,
                    legend=dict(orientation="h", y=-0.15, font=dict(size=9)),
                    xaxis=dict(tickformat="d", dtick=4),
                )
                st.plotly_chart(fig_traj, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r:
                # Forecast table
                st.markdown("""<div class='panel'>
                    <div class='panel-eyebrow'>Kalman · Forward Projection</div>
                    <div class='panel-title'>5-YEAR FORECAST</div>""",
                    unsafe_allow_html=True)
                if not fc.empty:
                    for _, r in fc.iterrows():
                        yr   = int(r["forecast_year"])
                        sc   = r["forecast_score"]
                        lo80 = r["ci_lower_80"]
                        hi80 = r["ci_upper_80"]
                        delta = sc - row["ensemble_score"]
                        dcol = GREEN if delta >= 0 else RED
                        st.markdown(f"""
                        <div style='display:flex;justify-content:space-between;
                             align-items:center;padding:7px 0;
                             border-bottom:1px solid {BORDER};
                             font-family:JetBrains Mono,monospace;font-size:11px'>
                          <span style='color:{GREY2}'>{yr}</span>
                          <span style='color:{WHITE};font-size:14px;font-weight:700'>{sc:.1f}</span>
                          <span style='color:{GREY};font-size:10px'>[{lo80:.1f}–{hi80:.1f}]</span>
                          <span style='color:{dcol}'>{delta:+.2f}</span>
                        </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Regime card
                if not reg_row.empty:
                    rr = reg_row.iloc[0]
                    vol = rr.get("volatility", 0)
                    st.markdown(f"""<div class='panel' style='margin-top:12px;border-color:{rc}30'>
                        <div class='panel-eyebrow'>Trajectory Classification</div>
                        <div class='panel-title' style='color:{rc}'>{regime}</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:{GREY2};margin-top:8px'>
                          <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                            <span>TREND SLOPE</span>
                            <span style='color:{GREEN if slope>0 else RED}'>{slope:+.4f}/yr</span>
                          </div>
                          <div style='display:flex;justify-content:space-between'>
                            <span>VOLATILITY</span>
                            <span style='color:{AMBER}'>{vol:.4f}</span>
                          </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # ── SHAP KPI drivers ─────────────────────────────────────────────
            col_l2, col_r2 = st.columns([1, 1])
            with col_l2:
                st.markdown("""<div class='panel'>
                    <div class='panel-eyebrow'>SHAP Attribution · KPI Contribution</div>
                    <div class='panel-title'>WHAT DRIVES THIS NATION'S SCORE</div>""",
                    unsafe_allow_html=True)

                shap_g = D["shap_global"]
                if not shap_g.empty:
                    top10 = shap_g.head(10).copy()
                    top10["pct"] = top10["mean_abs_shap"] / top10["mean_abs_shap"].sum() * 100
                    DIM_C = {"culture":GREEN,"innovation":BLUE,
                              "governance":AMBER,"trade":PURPLE,"human_dev":RED,"other":GREY2}

                    fig_shap = go.Figure()
                    colors_shap = [DIM_C.get(d, GREY2) for d in top10.get("dimension", ["other"]*len(top10))]
                    fig_shap.add_trace(go.Bar(
                        x=top10["pct"],
                        y=top10["kpi"],
                        orientation='h',
                        marker=dict(color=colors_shap, line=dict(width=0)),
                        text=[f"{p:.1f}%" for p in top10["pct"]],
                        textposition="outside",
                        textfont=dict(size=9, color=GREY2,
                                      family="JetBrains Mono, monospace"),
                        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                    ))
                    fig_shap.update_layout(**PLOT_LAYOUT, height=300,
                        yaxis=dict(autorange="reversed",
                                   tickfont=dict(size=9, color=WHITE)),
                        xaxis=dict(title="% of Attribution",
                                   tickfont=dict(size=9)),
                        bargap=0.3,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r2:
                st.markdown("""<div class='panel'>
                    <div class='panel-eyebrow'>Counterfactual Simulation · +10% Per KPI</div>
                    <div class='panel-title'>BEST POLICY LEVERS</div>""",
                    unsafe_allow_html=True)

                ct = D["counter"]
                if not ct.empty and "country" in ct.columns:
                    ct_c = ct[ct["country"]==sel_iso].copy()
                    if not ct_c.empty and "effect" in ct_c.columns:
                        ct_c["abs_effect"] = ct_c["effect"].abs()
                        ct_top = ct_c.nlargest(10, "abs_effect")
                        fig_ct = go.Figure()
                        fig_ct.add_trace(go.Bar(
                            x=ct_top["effect"],
                            y=ct_top["treatment_var"],
                            orientation='h',
                            marker=dict(
                                color=[GREEN if v>0 else RED for v in ct_top["effect"]],
                                line=dict(width=0),
                            ),
                            text=[f"{v:+.4f}" for v in ct_top["effect"]],
                            textposition="outside",
                            textfont=dict(size=9, color=GREY2,
                                          family="JetBrains Mono, monospace"),
                            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
                        ))
                        fig_ct.update_layout(**PLOT_LAYOUT, height=300,
                            yaxis=dict(autorange="reversed",
                                       tickfont=dict(size=9)),
                            xaxis=dict(title="Score Δ if +10%",
                                       tickfont=dict(size=9)),
                            bargap=0.3,
                        )
                        st.plotly_chart(fig_ct, use_container_width=True)
                    else:
                        st.markdown(f"<p style='color:{GREY2};font-size:11px;font-family:JetBrains Mono,monospace'>No counterfactual data for {sel_iso}.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIME MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div style='padding:16px 24px 0'>", unsafe_allow_html=True)

    regimes = D["regimes"]
    if regimes.empty:
        st.warning("No regime data found.")
    else:
        col_l, col_r = st.columns([1.8, 1])
        with col_l:
            st.markdown("""<div class='panel'>
                <div class='panel-eyebrow'>Trajectory Classification · All Nations</div>
                <div class='panel-title'>SLOPE vs VOLATILITY · REGIME SPACE</div>""",
                unsafe_allow_html=True)

            fig_scat = go.Figure()
            for regime, grp in regimes.groupby("regime"):
                rc = REGIME_COLORS.get(regime, GREY2)
                fig_scat.add_trace(go.Scatter(
                    x=grp["trend_slope"],
                    y=grp["volatility"],
                    mode="markers+text",
                    name=regime,
                    text=grp["iso3"],
                    textposition="top center",
                    textfont=dict(size=7, color=rc,
                                  family="JetBrains Mono, monospace"),
                    marker=dict(
                        size=grp["latest_score"]/8,
                        color=rc,
                        opacity=0.75,
                        line=dict(color=BG, width=1),
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Slope: %{x:+.4f}/yr<br>"
                        "Volatility: %{y:.4f}<br>"
                        f"Regime: {regime}<extra></extra>"
                    ),
                ))

            # Quadrant lines
            fig_scat.add_hline(y=regimes["volatility"].median(),
                               line_color=BORDER, line_width=1, line_dash="dot")
            fig_scat.add_vline(x=0, line_color=BORDER,
                               line_width=1, line_dash="dot")

            fig_scat.update_layout(**PLOT_LAYOUT, height=480,
                xaxis=dict(title="TREND SLOPE (pts/yr)", zeroline=True,
                           zerolinecolor=GREY, tickfont=dict(size=9)),
                yaxis=dict(title="VOLATILITY (σ innovations)",
                           tickfont=dict(size=9)),
                legend=dict(orientation="h", y=-0.12, font=dict(size=9)),
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            # Regime donut
            st.markdown("""<div class='panel'>
                <div class='panel-eyebrow'>Distribution · 195 Nations</div>
                <div class='panel-title'>REGIME BREAKDOWN</div>""",
                unsafe_allow_html=True)

            rc_counts = regimes["regime"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=rc_counts.index,
                values=rc_counts.values,
                hole=0.62,
                marker=dict(
                    colors=[REGIME_COLORS.get(r, GREY2) for r in rc_counts.index],
                    line=dict(color=BG, width=2),
                ),
                textfont=dict(family="JetBrains Mono, monospace", size=9),
                hovertemplate="<b>%{label}</b><br>%{value} nations (%{percent})<extra></extra>",
            ))
            fig_donut.update_layout(**PLOT_LAYOUT, height=240,
                showlegend=True,
                legend=dict(orientation="v", font=dict(size=9)),
                annotations=[dict(
                    text=f"<b>{len(regimes)}</b><br><span style='font-size:10px'>NATIONS</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color=WHITE, size=14,
                              family="JetBrains Mono, monospace"),
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Regime roster
            st.markdown("""<div class='panel'>
                <div class='panel-eyebrow'>Nations by Regime</div>
                <div class='panel-title'>REGIME ROSTER</div>""",
                unsafe_allow_html=True)

            sel_regime = st.selectbox("FILTER REGIME",
                ["ALL"] + list(REGIME_COLORS.keys()))
            filtered = regimes if sel_regime=="ALL" else regimes[regimes["regime"]==sel_regime]
            filtered = filtered.sort_values("latest_score", ascending=False)

            for _, r in filtered.head(12).iterrows():
                rc2 = REGIME_COLORS.get(r["regime"], GREY2)
                pct = r["latest_score"] / 100
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:8px;
                     padding:5px 0;border-bottom:1px solid {BORDER}'>
                  <span style='color:{GREY2};font-family:JetBrains Mono,monospace;
                        font-size:10px;min-width:32px'>{r['iso3']}</span>
                  <div style='flex:1;height:3px;background:{BORDER};border-radius:2px'>
                    <div style='width:{pct*100:.0f}%;height:3px;
                         background:{rc2};border-radius:2px'></div>
                  </div>
                  <span style='color:{WHITE};font-family:JetBrains Mono,monospace;
                        font-size:10px;min-width:36px'>{r['latest_score']:.1f}</span>
                  <span style='color:{rc2};font-family:JetBrains Mono,monospace;
                        font-size:9px;min-width:14px'>{r['trend_slope']:+.3f}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div style='padding:16px 24px 0'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        # Model comparison
        st.markdown("""<div class='panel'>
            <div class='panel-eyebrow'>Temporal Cross-Validation · 4 Folds</div>
            <div class='panel-title'>MODEL PERFORMANCE COMPARISON</div>""",
            unsafe_allow_html=True)

        mc = D["model_cmp"]
        if not mc.empty:
            fig_mc = make_subplots(rows=1, cols=2,
                subplot_titles=["MAE (lower = better)", "R² (higher = better)"])

            model_colors = [GREEN, BLUE, AMBER]
            for i, (_, row) in enumerate(mc.iterrows()):
                c = model_colors[i % len(model_colors)]
                fig_mc.add_trace(go.Bar(
                    name=row["model"], x=[row["model"]],
                    y=[row["mae_mean"]], marker_color=c,
                    error_y=dict(type="data", array=[row.get("mae_std",0)],
                                 color=GREY2, thickness=1.5),
                    showlegend=False,
                    hovertemplate=f"{row['model']}<br>MAE: {row['mae_mean']:.3f} ± {row.get('mae_std',0):.3f}<extra></extra>",
                ), row=1, col=1)
                fig_mc.add_trace(go.Bar(
                    name=row["model"], x=[row["model"]],
                    y=[row["r2_mean"]], marker_color=c,
                    error_y=dict(type="data", array=[row.get("r2_std",0)],
                                 color=GREY2, thickness=1.5),
                    showlegend=True,
                    hovertemplate=f"{row['model']}<br>R²: {row['r2_mean']:.3f} ± {row.get('r2_std',0):.3f}<extra></extra>",
                ), row=1, col=2)

            fig_mc.update_layout(**PLOT_LAYOUT, height=260,
                showlegend=True,
                legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
            )
            fig_mc.update_annotations(font=dict(size=9, color=GREY2,
                                                 family="JetBrains Mono, monospace"))
            st.plotly_chart(fig_mc, use_container_width=True)

            # Stats table
            for _, row in mc.iterrows():
                is_best = row["r2_mean"] == mc["r2_mean"].max()
                bc = GREEN if is_best else GREY
                star = "★ BEST" if is_best else ""
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                     padding:8px 12px;margin-bottom:4px;
                     background:{BG2};border:1px solid {bc}30;border-radius:2px;
                     font-family:JetBrains Mono,monospace;font-size:10px'>
                  <span style='color:{bc};font-weight:700'>{row['model']}</span>
                  <span style='color:{GREY2}'>MAE <span style='color:{WHITE}'>{row['mae_mean']:.3f}</span></span>
                  <span style='color:{GREY2}'>R² <span style='color:{WHITE}'>{row['r2_mean']:.3f}</span></span>
                  <span style='color:{GREY2}'>CI Cov <span style='color:{WHITE}'>{row.get('ci_coverage',0):.0%}</span></span>
                  <span style='color:{GREEN};font-size:9px'>{star}</span>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Feature importance
        st.markdown("""<div class='panel'>
            <div class='panel-eyebrow'>Ensemble Weighted · KPI Importance</div>
            <div class='panel-title'>COMBINED FEATURE IMPORTANCE</div>""",
            unsafe_allow_html=True)

        # Try combined importance file, fall back to shap_global
        fi_path = "output/combined_feature_importance.csv"
        fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else D["shap_global"]
        if not fi_df.empty:
            fi_show = fi_df.head(15).copy()
            fi_col = "importance" if "importance" in fi_show.columns else "mean_abs_shap"
            fi_show["pct"] = fi_show[fi_col] / fi_show[fi_col].sum() * 100

            fig_fi = go.Figure()
            bar_colors = [GREEN if i < 3 else BLUE if i < 6 else GREY2
                          for i in range(len(fi_show))]
            fig_fi.add_trace(go.Bar(
                y=fi_show["feature"][::-1],
                x=fi_show["pct"][::-1],
                orientation='h',
                marker=dict(color=list(reversed(bar_colors)), line=dict(width=0)),
                text=[f"{p:.1f}%" for p in fi_show["pct"][::-1]],
                textposition="outside",
                textfont=dict(size=8, color=GREY2,
                              family="JetBrains Mono, monospace"),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_fi.update_layout(**PLOT_LAYOUT, height=380,
                yaxis=dict(tickfont=dict(size=9, color=WHITE)),
                xaxis=dict(title="% Importance", tickfont=dict(size=9)),
                bargap=0.2,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        # Score correlation heatmap across models
        st.markdown("""<div class='panel'>
            <div class='panel-eyebrow'>Cross-Model Correlation · Score Agreement</div>
            <div class='panel-title'>ENSEMBLE CONSENSUS MATRIX</div>""",
            unsafe_allow_html=True)

        preds = D["preds"]
        model_cols = [c for c in ["xgboost_score","randomforest_score","lightgbm_score","ensemble_score"]
                      if c in preds.columns]
        if len(model_cols) >= 2:
            corr = preds[model_cols].corr()
            labels = [c.replace("_score","").upper().replace("RANDOMFOREST","RAND FOREST")
                      for c in model_cols]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=labels, y=labels,
                colorscale=[[0,BG2],[0.5,BLUE],[1.0,GREEN]],
                zmin=0.9, zmax=1.0,
                text=[[f"{v:.4f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=10, family="JetBrains Mono, monospace"),
                hovertemplate="%{y} vs %{x}: %{z:.4f}<extra></extra>",
            ))
            fig_corr.update_layout(**PLOT_LAYOUT, height=220,
                xaxis=dict(tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Stability distribution
        st.markdown("""<div class='panel'>
            <div class='panel-eyebrow'>Kalman Filter · Trajectory Stability</div>
            <div class='panel-title'>STABILITY CLASSIFICATION</div>""",
            unsafe_allow_html=True)

        summary = D["summary"]
        if not summary.empty and "stability_class" in summary.columns:
            stab = summary["stability_class"].value_counts()
            stab_colors = {
                "Stable": GREEN, "Moderate": AMBER, "Volatile": RED
            }
            fig_stab = go.Figure(go.Bar(
                x=list(stab.index),
                y=list(stab.values),
                marker=dict(
                    color=[stab_colors.get(str(s), GREY2) for s in stab.index],
                    line=dict(width=0),
                ),
                text=list(stab.values),
                textposition="outside",
                textfont=dict(size=10, color=WHITE,
                              family="JetBrains Mono, monospace"),
                hovertemplate="%{x}: %{y} nations<extra></extra>",
            ))
            fig_stab.update_layout(**PLOT_LAYOUT, height=180,
                yaxis=dict(title="Nations", tickfont=dict(size=9)),
                xaxis=dict(tickfont=dict(size=11)),
                bargap=0.3,
            )
            st.plotly_chart(fig_stab, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Score vs Trend quadrant
        st.markdown("""<div class='panel'>
            <div class='panel-eyebrow'>Score vs Trajectory · Strategic Quadrant</div>
            <div class='panel-title'>POWER × MOMENTUM MATRIX</div>""",
            unsafe_allow_html=True)

        if not regimes.empty and not preds.empty:
            quad = regimes.merge(preds[["iso3","ensemble_score"]], on="iso3", how="left")
            fig_quad = go.Figure()
            for regime, grp in quad.groupby("regime"):
                rc3 = REGIME_COLORS.get(regime, GREY2)
                fig_quad.add_trace(go.Scatter(
                    x=grp["ensemble_score"],
                    y=grp["trend_slope"],
                    mode="markers",
                    name=regime,
                    marker=dict(size=7, color=rc3, opacity=0.75,
                                line=dict(color=BG, width=1)),
                    text=grp["iso3"],
                    hovertemplate="<b>%{text}</b><br>Score: %{x:.1f}<br>Slope: %{y:+.4f}<extra></extra>",
                ))
            # Quadrant dividers
            fig_quad.add_hline(y=0, line_color=GREY, line_width=0.8, line_dash="dot")
            fig_quad.add_vline(x=quad["ensemble_score"].median(),
                               line_color=GREY, line_width=0.8, line_dash="dot")
            # Quadrant labels
            xmed = quad["ensemble_score"].median()
            fig_quad.add_annotation(x=xmed+20, y=quad["trend_slope"].max()*0.8,
                text="RISING POWERS", font=dict(size=8, color=GREEN,
                family="JetBrains Mono, monospace"), showarrow=False)
            fig_quad.add_annotation(x=xmed-20, y=quad["trend_slope"].min()*0.8,
                text="DECLINING PERIPHERY", font=dict(size=8, color=RED,
                family="JetBrains Mono, monospace"), showarrow=False)

            fig_quad.update_layout(**PLOT_LAYOUT, height=280,
                xaxis=dict(title="ENSEMBLE SCORE", tickfont=dict(size=9)),
                yaxis=dict(title="TREND SLOPE", tickfont=dict(size=9)),
                legend=dict(orientation="h", y=-0.2, font=dict(size=8)),
            )
            st.plotly_chart(fig_quad, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AGENT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div style='padding:16px 24px 0'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        st.markdown(f"""<div class='panel' style='border-color:{GREEN}30'>
            <div class='panel-eyebrow'>Claude claude-sonnet-4-6 · Soft Power Agent</div>
            <div class='panel-title'>INTELLIGENCE QUERY INTERFACE</div>
            <p style='font-family:JetBrains Mono,monospace;font-size:10px;color:{GREY2};margin-bottom:16px'>
            Natural language queries resolved against 195-country soft power database.
            SHAP attribution · Kalman forecasts · Counterfactual simulation.
            </p>""", unsafe_allow_html=True)

        # Init chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(f"""
                <div style='text-align:center;padding:32px;color:{GREY};
                     font-family:JetBrains Mono,monospace;font-size:11px'>
                  <div style='font-size:24px;margin-bottom:8px'>⬡</div>
                  SYSTEM READY · AWAITING QUERY
                </div>""", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style='margin-bottom:12px'>
                      <div class='chat-label' style='color:{BLUE}'>ANALYST</div>
                      <div class='chat-msg-user'>{msg['content']}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='margin-bottom:12px'>
                      <div class='chat-label' style='color:{GREEN};text-align:right'>SOFT POWER AGENT</div>
                      <div class='chat-msg-agent'>{msg['content']}</div>
                    </div>""", unsafe_allow_html=True)

        # Input
        query = st.text_input("", placeholder="e.g. What drives India's soft power ranking?",
                              label_visibility="collapsed", key="query_input")
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            send = st.button("▶  TRANSMIT", use_container_width=True)
        with c2:
            if st.button("↺  CLEAR", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and query:
            st.session_state.chat_history.append({"role":"user","content":query})
            with st.spinner("PROCESSING QUERY..."):
                try:
                    import anthropic
                    # Build context from pipeline outputs
                    preds = D["preds"]
                    context_parts = ["=== SOFT POWER DATABASE CONTEXT ==="]

                    # Try to identify country in query
                    q_lower = query.lower()
                    target_iso = None
                    for iso, name in ISOMAP.items():
                        if name and (name.lower() in q_lower or iso.lower() in q_lower):
                            target_iso = iso
                            break

                    if target_iso and not preds.empty:
                        row = preds[preds["iso3"]==target_iso]
                        if not row.empty:
                            r = row.iloc[0]
                            context_parts.append(f"Country: {ISOMAP.get(target_iso,target_iso)} ({target_iso})")
                            context_parts.append(f"Ensemble Score: {r.get('ensemble_score',0):.2f}/100")
                            if "ensemble_rank" in r:
                                context_parts.append(f"Global Rank: #{int(r['ensemble_rank'])} of 195")
                            for mc in ["xgboost_score","randomforest_score","lightgbm_score"]:
                                if mc in r:
                                    context_parts.append(f"{mc}: {r[mc]:.2f}")

                        reg = D["regimes"]
                        if not reg.empty:
                            rr = reg[reg["iso3"]==target_iso]
                            if not rr.empty:
                                context_parts.append(f"Regime: {rr.iloc[0]['regime']}")
                                context_parts.append(f"Trend slope: {rr.iloc[0]['trend_slope']:+.4f}/yr")

                        fc = D["forecast"]
                        if not fc.empty:
                            fcc = fc[fc["iso3"]==target_iso]
                            if not fcc.empty:
                                context_parts.append("5-Year Forecast:")
                                for _, fr in fcc.iterrows():
                                    context_parts.append(f"  {int(fr['forecast_year'])}: {fr['forecast_score']:.1f} [{fr['ci_lower_80']:.1f}–{fr['ci_upper_80']:.1f}]")

                        shap_g = D["shap_global"]
                        if not shap_g.empty:
                            context_parts.append("Top global KPI drivers (SHAP):")
                            for _, sr in shap_g.head(5).iterrows():
                                context_parts.append(f"  {sr['kpi']}: {sr['mean_abs_shap']:.4f} ({sr.get('dimension','?')})")

                        ct = D["counter"]
                        if not ct.empty and "country" in ct.columns:
                            ct_c = ct[ct["country"]==target_iso]
                            if not ct_c.empty and "effect" in ct_c.columns:
                                best = ct_c.loc[ct_c["effect"].abs().idxmax()]
                                context_parts.append(f"Best policy lever: {best['treatment_var']} (+10% → score delta {best['effect']:+.4f})")
                    else:
                        if not preds.empty:
                            context_parts.append(f"Database covers {len(preds)} countries, scores range {preds['ensemble_score'].min():.1f}–{preds['ensemble_score'].max():.1f}")
                            top5 = preds.nlargest(5,"ensemble_score")
                            context_parts.append("Top 5: " + ", ".join([
                                f"{ISOMAP.get(r['iso3'],r['iso3'])} ({r['ensemble_score']:.1f})"
                                for _,r in top5.iterrows()
                            ]))

                    context = "\n".join(context_parts)

                    client = anthropic.Anthropic()
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=800,
                        system=f"""You are the Soft Power Agent in the Geopolitical Intelligence Analysis System (GIAS).
You have access to soft power scores, 5-year Kalman forecasts, regime classifications, SHAP-based KPI attribution, and counterfactual policy simulations for 195 countries (2000-2024).

Response style:
- Be direct and analytical. No filler phrases.
- Always cite specific scores, ranks, and percentages from the data.
- Structure: Score/Rank → Key Drivers → Trajectory → Policy Levers
- Keep responses concise (3-5 paragraphs max).
- Use plain language for SHAP (say "driven by" not "SHAP value of").""",
                        messages=[
                            {"role":"user","content":f"{context}\n\nQuery: {query}"}
                        ]
                    )
                    answer = response.content[0].text
                except ImportError:
                    answer = ("Anthropic SDK not installed. Run: pip install anthropic\n\n"
                              "Once installed, set ANTHROPIC_API_KEY and re-run the dashboard.")
                except Exception as e:
                    answer = f"Agent error: {str(e)}\n\nEnsure ANTHROPIC_API_KEY is set in your environment."

            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""<div class='panel'>
            <div class='panel-eyebrow'>Quick Queries</div>
            <div class='panel-title'>EXAMPLE INTELLIGENCE REQUESTS</div>""",
            unsafe_allow_html=True)

        examples = [
            "What drives India's soft power ranking?",
            "Which country has the strongest 5-year growth forecast?",
            "Compare Germany and France soft power trajectories.",
            "What is the best policy lever for Brazil to improve its score?",
            "Which nations are in the Fragile regime and why?",
            "How does USA's soft power compare to China?",
            "What KPIs matter most globally for soft power?",
            "Which countries will overtake their current rank by 2029?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":ex})
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # System status panel
        st.markdown(f"""<div class='panel' style='margin-top:0'>
            <div class='panel-eyebrow'>System Status</div>
            <div class='panel-title'>PIPELINE INTEGRITY</div>""",
            unsafe_allow_html=True)

        checks = [
            ("Ensemble Model",       os.path.exists("output/ensemble_model.pkl")),
            ("Predictions",          not D["preds"].empty),
            ("Kalman Forecast",      not D["forecast"].empty),
            ("Regime Labels",        not D["regimes"].empty),
            ("SHAP Attribution",     not D["shap_global"].empty),
            ("Counterfactuals",      not D["counter"].empty),
            ("Model Comparison",     not D["model_cmp"].empty),
        ]
        for label, ok in checks:
            dot_col = GREEN if ok else RED
            status  = "ONLINE" if ok else "MISSING"
            st.markdown(f"""
            <div style='display:flex;align-items:center;justify-content:space-between;
                 padding:5px 0;border-bottom:1px solid {BORDER};
                 font-family:JetBrains Mono,monospace;font-size:10px'>
              <span style='color:{GREY2}'>{label}</span>
              <span style='color:{dot_col};display:flex;align-items:center;gap:5px'>
                <span style='width:5px;height:5px;background:{dot_col};border-radius:50%;
                      display:inline-block;{"box-shadow:0 0 5px "+dot_col if ok else ""}'></span>
                {status}
              </span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)