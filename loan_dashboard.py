"""
CARA PAKAI:
    1. Jalankan loan_model_trainer.py terlebih dahulu → hasilkan pipe_ensemble.pkl
    2. Letakkan pipe_ensemble.pkl + model_meta.json di folder yang sama
    3. streamlit run loan_dashboard.py
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import numpy as np
import os
from loan_rule import check_all_rules, RULES_CONFIG

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan App BI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONSTANTS — sesuaikan dengan DB
# ─────────────────────────────────────────────
DB_NAME      = "loans.db"
TARGET_TABLE = "loan_records"
MODEL_PKL    = "pipe_ensemble.pkl"
MODEL_META   = "model_meta.json"

PLOT_BG    = "#112240"
PAPER_BG   = "#112240"
FONT_COLOR = "#C8D6E5"
GRID_COLOR = "#1E3A5F"
PALETTE    = ["#F4A261", "#52D9A3", "#4A9EF4", "#F47461", "#B07FEA", "#FFD166"]

# Mapping kolom DB>>nama fitur model
DB_TO_MODEL = {
    "age":                 "Age",
    "gender":              "Gender",
    "education":           "Education",
    "person_income":       "Person Income",
    "employee_experience": "Employee Experience",
    "home_ownership":      "Home Onwership",  
    "loan_amount":         "Loan Amount",
    "loan_intent":         "Loan Intent",
    "loan_interest_rate":  "Loan interest Rate",
    "loan_percentage":     "Loan percentage",
    "credit_history":      "Credit History",
    "credit_score":        "Credit Score",
    "previous_loan":       "Previous Loan",
    "loan_status":         "Loan Status",
}
EDUCATION_ORDER = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0D1B2A; color: #E8EDF2; }
.stApp { background-color: #0D1B2A; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0A1628 0%, #112240 100%); border-right: 1px solid #1E3A5F; }
[data-testid="stSidebar"] * { color: #C8D6E5 !important; }
[data-testid="stSidebar"] label { color: #8BA7C7 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0D1B2A; border-bottom: 1px solid #1E3A5F; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #5A7A9F; border-radius: 8px 8px 0 0; font-size: 0.80rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; padding: 10px 20px; border: 1px solid transparent; }
.stTabs [aria-selected="true"] { background: #112240 !important; color: #F4A261 !important; border: 1px solid #1E3A5F !important; border-bottom: 1px solid #112240 !important; }

/* ── KPI Cards ── */
.kpi-card { background: linear-gradient(135deg, #112240 0%, #1B3A6B 100%); border: 1px solid #1E3A5F; border-radius: 12px; padding: 20px 24px; position: relative; overflow: hidden; transition: transform 0.2s; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #F4A261; }
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { font-size: 0.70rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: #8BA7C7; margin-bottom: 8px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 700; color: #F4A261; line-height: 1; }
.kpi-sub { font-size: 0.72rem; color: #5A7A9F; margin-top: 6px; }
.kpi-accent { color: #52D9A3; } .kpi-danger { color: #F47461; } .kpi-info { color: #4A9EF4; }

/* ── Section Headers ── */
.section-header { display: flex; align-items: center; gap: 10px; margin: 32px 0 16px 0; padding-bottom: 10px; border-bottom: 1px solid #1E3A5F; }
.section-header h3 { font-size: 0.80rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; color: #8BA7C7; margin: 0; }
.section-dot { width: 8px; height: 8px; border-radius: 50%; background: #F4A261; flex-shrink: 0; }

/* ── Chart Cards ── */
.chart-card { background: #112240; border: 1px solid #1E3A5F; border-radius: 12px; padding: 20px; margin-bottom: 4px; }
.chart-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #8BA7C7; margin-bottom: 4px; }
.chart-desc { font-size: 0.72rem; color: #3D5A7A; margin-bottom: 16px; }

/* ── Insight Box ── */
.insight-box { background: linear-gradient(135deg, #0F2A1E 0%, #0D1B2A 100%); border: 1px solid #1A4A35; border-left: 3px solid #52D9A3; border-radius: 8px; padding: 12px 16px; margin: 12px 0; font-size: 0.80rem; color: #A0C4A8; line-height: 1.5; }
.insight-box strong { color: #52D9A3; }
.warning-box { background: linear-gradient(135deg, #2A1A0F 0%, #0D1B2A 100%); border-left: 3px solid #F4A261; border-radius: 8px; padding: 12px 16px; margin: 12px 0; font-size: 0.80rem; color: #C4A280; }

/* ── Sidebar ── */
.counter-badge { background: linear-gradient(135deg, #1B3A6B, #0F2A45); border: 1px solid #F4A261; border-radius: 10px; padding: 14px 18px; text-align: center; margin: 16px 0; }
.counter-number { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: #F4A261; }
.counter-label { font-size: 0.68rem; color: #5A7A9F; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* ── Prediction Result ── */
.pred-approved { background: linear-gradient(135deg, #0F2A1E 0%, #112240 100%); border: 2px solid #52D9A3; border-radius: 16px; padding: 28px; text-align: center; }
.pred-rejected { background: linear-gradient(135deg, #2A0F0F 0%, #112240 100%); border: 2px solid #F47461; border-radius: 16px; padding: 28px; text-align: center; }
.pred-icon { font-size: 3rem; margin-bottom: 8px; }
.pred-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; color: #5A7A9F; margin-bottom: 4px; }
.pred-status-approved { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #52D9A3; }
.pred-status-rejected { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #F47461; }
.pred-prob { font-family: 'JetBrains Mono', monospace; font-size: 1rem; margin-top: 6px; }

/* ── Gauge/Progress bar ── */
.risk-row { display: flex; justify-content: space-between; font-size: 0.72rem; color: #8BA7C7; margin-bottom: 2px; }
.risk-bar-wrap { background: #0D1B2A; border-radius: 6px; height: 7px; margin-bottom: 10px; overflow: hidden; }
.risk-bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s; }

/* ── Buttons ── */
.stDownloadButton > button { background: linear-gradient(135deg, #F4A261, #E07B3A) !important; color: #0D1B2A !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #F4A261, #E07B3A) !important; color: #0D1B2A !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-size: 0.9rem !important; width: 100% !important; padding: 12px !important; }

/* ── Model badge ── */
.model-badge { display: inline-block; background: #0D2A45; border: 1px solid #1E3A5F; border-radius: 20px; padding: 4px 12px; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; color: #4A9EF4; margin: 2px; }

hr { border-color: #1E3A5F !important; }
.stPlotlyChart { border-radius: 8px; overflow: hidden; }
.brand-footer { text-align: center; padding: 24px 0 8px 0; font-size: 0.68rem; color: #2A4A6B; letter-spacing: 0.1em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def apply_theme(fig, height=280):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=11),
        height=height, margin=dict(t=16, b=16, l=8, r=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=FONT_COLOR)),
        coloraxis_colorbar=dict(tickfont=dict(color=FONT_COLOR)),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR))
    fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR))
    return fig


def risk_bar_html(label, value_display, pct, color):
    return f"""
    <div class='risk-row'><span>{label}</span><span style='color:{color};font-family:JetBrains Mono;'>{value_display}</span></div>
    <div class='risk-bar-wrap'><div class='risk-bar-fill' style='width:{pct:.0f}%;background:{color};'></div></div>
    """


def score_to_color(pct_good):
    """pct_good: 0–100, semakin tinggi semakin baik → hijau."""
    if pct_good >= 70: return "#52D9A3"
    elif pct_good >= 40: return "#F4A261"
    return "#F47461"


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur yang sama persis seperti di loan_model_trainer.py."""
    df = df.copy()
    df["loan_to_income_ratio"]   = df["Loan Amount"] / (df["Person Income"] + 1)
    df["annual_interest_burden"] = (df["Loan Amount"] * df["Loan interest Rate"] / 100) / (df["Person Income"] + 1)
    df["income_per_exp_year"]    = df["Person Income"] / (df["Employee Experience"] + 1)
    df["score_per_hist_year"]    = df["Credit Score"] / (df["Credit History"] + 1)
    df["affordability_index"]    = df["Loan percentage"] * df["Loan interest Rate"]
    df["credit_risk_score"]      = (df["Credit Score"] / 850) - df["Loan percentage"]
    df["financial_maturity"]     = df["Age"] * 0.4 + df["Employee Experience"] * 0.6
    df["credit_tier"] = pd.cut(
        df["Credit Score"],
        bins=[0, 579, 669, 739, 850],
        labels=["Poor", "Fair", "Good", "Excellent"]
    ).astype(str)
    return df


# ─────────────────────────────────────────────
#  DATA & MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT * FROM {TARGET_TABLE}", conn)
    conn.close()
    df.rename(columns={k: v for k, v in DB_TO_MODEL.items() if k in df.columns}, inplace=True)
    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PKL):
        return None, None
    with open(MODEL_PKL, "rb") as f:
        pipe = pickle.load(f)
    meta = {}
    if os.path.exists(MODEL_META):
        with open(MODEL_META) as f:
            meta = json.load(f)
    return pipe, meta


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Loan App")
    st.markdown("<p style='font-size:0.72rem;color:#3D5A7A;margin-top:-8px;'>Loan Approval Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")

    try:
        raw_df = load_data()
    except Exception as e:
        st.error(f"Gagal memuat database: {e}")
        st.stop()

    st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#5A7A9F;'>Filters (Analytics Tab)</p>", unsafe_allow_html=True)

    sel_gender  = st.selectbox("Gender",         ["All"] + sorted(raw_df["Gender"].dropna().unique().tolist()))
    sel_edu     = st.selectbox("Education",       ["All"] + sorted(raw_df["Education"].dropna().unique().tolist()))
    sel_intent  = st.selectbox("Loan Intent",     ["All"] + sorted(raw_df["Loan Intent"].dropna().unique().tolist()))
    sel_home    = st.selectbox("Home Ownership",  ["All"] + sorted(raw_df["Home Onwership"].dropna().unique().tolist()))
    sel_prev    = st.selectbox("Previous Loan",   ["All"] + sorted(raw_df["Previous Loan"].dropna().unique().tolist()))
    status_map  = {"All": None, "Approved ✅": 1, "Rejected ❌": 0}
    sel_status  = status_map[st.selectbox("Loan Status", list(status_map.keys()))]

    age_r   = st.slider("Age",          int(raw_df["Age"].min()),          int(raw_df["Age"].max()),          (int(raw_df["Age"].min()),          int(raw_df["Age"].max())))
    inc_r   = st.slider("Income (Rp)",  int(raw_df["Person Income"].min()), int(raw_df["Person Income"].max()), (int(raw_df["Person Income"].min()), int(raw_df["Person Income"].max())))
    scr_r   = st.slider("Credit Score", int(raw_df["Credit Score"].min()),  int(raw_df["Credit Score"].max()),  (int(raw_df["Credit Score"].min()),  int(raw_df["Credit Score"].max())))

    df = raw_df.copy()
    if sel_gender != "All":    df = df[df["Gender"]        == sel_gender]
    if sel_edu    != "All":    df = df[df["Education"]      == sel_edu]
    if sel_intent != "All":    df = df[df["Loan Intent"]    == sel_intent]
    if sel_home   != "All":    df = df[df["Home Onwership"] == sel_home]
    if sel_prev   != "All":    df = df[df["Previous Loan"]  == sel_prev]
    if sel_status is not None: df = df[df["Loan Status"]    == sel_status]
    df = df[df["Age"].between(*age_r) & df["Person Income"].between(*inc_r) & df["Credit Score"].between(*scr_r)]

    st.markdown(f"""
    <div class="counter-badge">
        <div class="counter-number">{len(df):,}</div>
        <div class="counter-label">Nasabah Aktif dalam Filter</div>
    </div>
    <p style='font-size:0.68rem;color:#3D5A7A;text-align:center;'>dari {len(raw_df):,} total data</p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.download_button("⬇ Download Filtered CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       "loanapp_filtered.csv", "text/csv")


# ─────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:16px'>
    <h1 style='font-size:1.6rem;font-weight:700;color:#E8EDF2;margin:0;letter-spacing:-0.02em;'>Loan Approval Intelligence</h1>
    <p style='font-size:0.80rem;color:#3D5A7A;margin:4px 0 0 0;'>Credit risk analytics & Ensemble ML prediction engine</p>
</div>
""", unsafe_allow_html=True)

tab_analytics, tab_predict = st.tabs(["Analytics Dashboard", "Ensemble Predictor"])


# ══════════════════════════════════════════════
#  TAB 1 — ANALYTICS
# ══════════════════════════════════════════════
with tab_analytics:
    if df.empty:
        st.warning("⚠️ Filter menghasilkan data kosong.")
        st.stop()

    df_app = df[df["Loan Status"] == 1]
    df_rej = df[df["Loan Status"] == 0]
    apr    = len(df_app) / len(df) * 100

    def cseg(s):
        if s < 580: return "Poor"
        elif s < 670: return "Fair"
        elif s < 740: return "Good"
        return "Excellent"
    df["credit_segment"] = df["Credit Score"].apply(cseg)
    seg_order = ["Poor", "Fair", "Good", "Excellent"]

    # KPIs
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Executive Summary</h3></div>""", unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Approval Rate</div><div class='kpi-value {'kpi-accent' if apr>=60 else 'kpi-danger'}'>{apr:.1f}%</div><div class='kpi-sub'>{len(df_app):,} dari {len(df):,}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Loan Amount</div><div class='kpi-value'>Rp{df['Loan Amount'].mean()/1e6:.1f}M</div><div class='kpi-sub'>Rata-rata pinjaman</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Credit Score</div><div class='kpi-value'>{df['Credit Score'].mean():.0f}</div><div class='kpi-sub'>Skor kredit rata-rata</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Interest Rate</div><div class='kpi-value'>{df['Loan interest Rate'].mean():.2f}%</div><div class='kpi-sub'>Suku bunga rata-rata</div></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg DTI Ratio</div><div class='kpi-value {'kpi-danger' if df['Loan percentage'].mean()>0.3 else 'kpi-accent'}'>{df['Loan percentage'].mean():.1%}</div><div class='kpi-sub'>Debt-to-income</div></div>", unsafe_allow_html=True)

    # Portfolio Overview
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Portfolio Overview</h3></div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        sc = df["Loan Status"].value_counts().reset_index()
        sc.columns = ["status","count"]
        sc["label"] = sc["status"].map({1:"Approved",0:"Rejected"})
        fig = go.Figure(go.Pie(labels=sc["label"],values=sc["count"],hole=0.65,
            marker=dict(colors=["#52D9A3","#F47461"],line=dict(color=PLOT_BG,width=3)),
            textfont=dict(color=FONT_COLOR,size=11)))
        fig.add_annotation(text=f"<b>{apr:.0f}%</b><br><span style='font-size:9px'>Approved</span>",x=0.5,y=0.5,showarrow=False,font=dict(size=18,color="#52D9A3",family="JetBrains Mono"))
        st.markdown("<div class='chart-card'><div class='chart-title'>Loan Status Distribution</div><div class='chart-desc'>Proporsi approval vs rejection</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,280),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        ig = df.groupby("Loan Intent")["Loan Status"].agg(["sum","count"]).reset_index()
        ig["rate"] = ig["sum"]/ig["count"]*100
        fig = px.bar(ig.sort_values("rate"),x="rate",y="Loan Intent",orientation="h",
            color="rate",color_continuous_scale=["#F47461","#F4A261","#52D9A3"],range_color=[0,100],
            labels={"rate":"Approval Rate (%)","Loan Intent":""},text=ig.sort_values("rate")["rate"].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition="outside",textfont_size=10)
        fig.update_coloraxes(showscale=False)
        st.markdown("<div class='chart-card'><div class='chart-title'>Approval Rate by Loan Intent</div><div class='chart-desc'>Tujuan pinjaman vs peluang disetujui</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,280),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c3:
        hg = df.groupby("Home Onwership")["Loan Status"].agg(["sum","count"]).reset_index()
        hg["rate"] = hg["sum"]/hg["count"]*100
        fig = px.bar(hg,x="Home Onwership",y="rate",color="Home Onwership",color_discrete_sequence=PALETTE,
            labels={"rate":"Approval Rate (%)","Home Onwership":""},text=hg["rate"].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition="outside",showlegend=False)
        st.markdown("<div class='chart-card'><div class='chart-title'>Approval Rate by Home Ownership</div><div class='chart-desc'>Status kepemilikan vs persetujuan</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,280),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # Risk Profile
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Risk & Credit Profile</h3></div>""", unsafe_allow_html=True)
    c4,c5 = st.columns([1.4,1])
    with c4:
        fig = go.Figure()
        for s,l,c in [(1,"Approved","#52D9A3"),(0,"Rejected","#F47461")]:
            fig.add_trace(go.Histogram(x=df[df["Loan Status"]==s]["Credit Score"],name=l,marker_color=c,opacity=0.7,nbinsx=30))
        fig.update_layout(barmode="overlay")
        st.markdown("<div class='chart-card'><div class='chart-title'>Credit Score Distribution</div><div class='chart-desc'>Persebaran skor kredit — approved vs rejected</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,300),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c5:
        sg = df.groupby("credit_segment")["Loan Status"].agg(["sum","count"]).reset_index()
        sg["rate"] = sg["sum"]/sg["count"]*100
        sg["credit_segment"] = pd.Categorical(sg["credit_segment"],categories=seg_order,ordered=True)
        sg = sg.sort_values("credit_segment")
        fig = px.bar(sg,x="credit_segment",y="rate",color="rate",
            color_continuous_scale=["#F47461","#F4A261","#52D9A3"],range_color=[0,100],
            labels={"rate":"Approval Rate (%)","credit_segment":""},text=sg["rate"].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.markdown("<div class='chart-card'><div class='chart-title'>Approval by Credit Segment</div><div class='chart-desc'>Poor / Fair / Good / Excellent</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,300),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    c6,c7 = st.columns(2)
    with c6:
        sp = df.sample(min(1000,len(df)),random_state=42).copy()
        sp["Status"] = sp["Loan Status"].map({1:"Approved",0:"Rejected"})
        fig = px.scatter(sp,x="Person Income",y="Loan Amount",color="Status",
            color_discrete_map={"Approved":"#52D9A3","Rejected":"#F47461"},opacity=0.55,
            hover_data=["Age","Credit Score","Loan Intent"])
        st.markdown("<div class='chart-card'><div class='chart-title'>Income vs Loan Amount</div><div class='chart-desc'>Affordability — titik hijau = disetujui</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,300),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c7:
        df["dti_bin"] = pd.cut(df["Loan percentage"],bins=[0,0.1,0.2,0.3,0.4,0.5,1.0],labels=["0–10%","10–20%","20–30%","30–40%","40–50%",">50%"])
        dg = df.groupby("dti_bin",observed=True)["Loan Status"].agg(["sum","count"]).reset_index()
        dg["rate"] = dg["sum"]/dg["count"]*100
        fig = px.line(dg,x="dti_bin",y="rate",markers=True,color_discrete_sequence=["#F4A261"],
            labels={"dti_bin":"Debt-to-Income Ratio","rate":"Approval Rate (%)"},
            text=dg["rate"].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(line=dict(width=2.5),marker=dict(size=8),textposition="top center")
        fig.add_hline(y=50,line_dash="dot",line_color="#3D5A7A",annotation_text="50% threshold",annotation_font_color="#3D5A7A")
        st.markdown("<div class='chart-card'><div class='chart-title'>DTI Ratio vs Approval Rate</div><div class='chart-desc'>Semakin tinggi beban, semakin rendah peluang approve</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,300),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # Demographics
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Customer Demographics</h3></div>""", unsafe_allow_html=True)
    c8,c9,c10 = st.columns(3)
    with c8:
        fig = go.Figure()
        for s,l,c in [(1,"Approved","#52D9A3"),(0,"Rejected","#F47461")]:
            fig.add_trace(go.Box(y=df[df["Loan Status"]==s]["Age"],name=l,marker_color=c,boxmean=True,line_width=1.5))
        st.markdown("<div class='chart-card'><div class='chart-title'>Age Distribution</div><div class='chart-desc'>Distribusi usia per status pinjaman</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,280),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c9:
        eg = df.groupby("Education")["Loan Status"].agg(["sum","count"]).reset_index()
        eg["rate"] = eg["sum"]/eg["count"]*100
        fig = px.bar(eg.sort_values("rate",ascending=False),x="Education",y="rate",
            color="Education",color_discrete_sequence=PALETTE,
            labels={"rate":"Approval Rate (%)","Education":""},
            text=eg.sort_values("rate",ascending=False)["rate"].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition="outside",showlegend=False)
        st.markdown("<div class='chart-card'><div class='chart-title'>Approval by Education Level</div><div class='chart-desc'>Tingkat pendidikan vs approval</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,280),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c10:
        df["exp_bin"] = pd.cut(df["Employee Experience"],bins=[0,2,5,10,20,100],labels=["0–2yr","2–5yr","5–10yr","10–20yr","20+yr"])
        xg = df.groupby("exp_bin",observed=True)["Loan Status"].agg(["sum","count"]).reset_index()
        xg["rate"] = xg["sum"]/xg["count"]*100
        fig = px.bar(xg,x="exp_bin",y="rate",color="rate",color_continuous_scale=["#F47461","#F4A261","#52D9A3"],
            range_color=[0,100],labels={"rate":"Approval Rate (%)","exp_bin":"Pengalaman"},
            text=xg["rate"].apply(lambda x:f"{x:.0f}%"))
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.markdown("<div class='chart-card'><div class='chart-title'>Employee Experience vs Approval</div><div class='chart-desc'>Stabilitas masa kerja pemohon</div>",unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig,280),use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # Insights
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Key Insights</h3></div>""", unsafe_allow_html=True)
    best_intent = df.groupby("Loan Intent")["Loan Status"].mean().idxmax()
    best_rate   = df.groupby("Loan Intent")["Loan Status"].mean().max()*100
    dti_min_row = dg[dg["rate"]==dg["rate"].min()]["dti_bin"].values[0]
    top_seg     = sg.sort_values("rate",ascending=False).iloc[0]
    i1,i2,i3 = st.columns(3)
    i1.markdown(f"<div class='insight-box'>💡 Pinjaman tujuan <strong>{best_intent}</strong> punya approval rate tertinggi: <strong>{best_rate:.0f}%</strong>.</div>", unsafe_allow_html=True)
    i2.markdown(f"<div class='insight-box'>⚠️ Approval rate drop paling drastis di DTI bucket <strong>{dti_min_row}</strong> — jadikan ini batas kebijakan.</div>", unsafe_allow_html=True)
    i3.markdown(f"<div class='insight-box'>🏆 Segmen kredit <strong>{top_seg['credit_segment']}</strong> punya approval rate <strong>{top_seg['rate']:.0f}%</strong>.</div>", unsafe_allow_html=True)

    with st.expander("🗂️  Data Mentah (Filtered)", expanded=False):
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=360)


# ══════════════════════════════════════════════
#  TAB 2 — ENSEMBLE PREDICTOR
# ══════════════════════════════════════════════
with tab_predict:
    pipe, meta = load_model()

    if pipe is None:
        st.markdown("""
        <div style='background:#1A1A0F;border:2px dashed #F4A261;border-radius:16px;padding:36px;text-align:center;margin:32px 0;'>
            <div style='font-size:2.5rem;margin-bottom:12px;'>📦</div>
            <div style='font-size:1rem;font-weight:700;color:#F4A261;margin-bottom:8px;'>pipe_ensemble.pkl belum ditemukan</div>
            <div style='font-size:0.80rem;color:#5A7A9F;line-height:1.8;'>
                Jalankan <code style='background:#112240;padding:2px 8px;border-radius:4px;color:#4A9EF4;'>python loan_model_trainer.py</code> terlebih dahulu.<br>
                Setelah selesai, salin <code style='background:#112240;padding:2px 6px;border-radius:4px;color:#4A9EF4;'>pipe_ensemble.pkl</code> dan
                <code style='background:#112240;padding:2px 6px;border-radius:4px;color:#4A9EF4;'>model_meta.json</code> ke folder ini, lalu refresh halaman.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Model info header
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Ensemble Model Status</h3></div>""", unsafe_allow_html=True)
    mi1,mi2,mi3,mi4 = st.columns(4)
    mi1.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Model Type</div>
    <div class='kpi-value kpi-info' style='font-size:0.9rem;'>Stacking Ensemble</div>
    <div class='kpi-sub'>XGB + LGBM + CatBoost → LR</div></div>""", unsafe_allow_html=True)
    mi2.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Test Accuracy</div>
    <div class='kpi-value kpi-accent'>{meta.get('accuracy',0)*100:.2f}%</div>
    <div class='kpi-sub'>Pada hold-out test set</div></div>""", unsafe_allow_html=True)
    mi3.markdown(f"""<div class='kpi-card'><div class='kpi-label'>AUC-ROC</div>
    <div class='kpi-value kpi-accent'>{meta.get('auc_roc',0):.4f}</div>
    <div class='kpi-sub'>Area under curve</div></div>""", unsafe_allow_html=True)
    mi4.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Total Features</div>
    <div class='kpi-value'>{meta.get('n_features',21)}</div>
    <div class='kpi-sub'>13 raw + 8 engineered</div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin:8px 0 24px 0;'>
        <span class='model-badge'>XGBoost</span>
        <span class='model-badge'>LightGBM</span>
        <span class='model-badge'>CatBoost</span>
        <span class='model-badge'>→ Logistic Regression (meta)</span>
        <span class='model-badge'>Optuna Tuned</span>
        <span class='model-badge'>No Data Leakage</span>
    </div>
    """, unsafe_allow_html=True)

    # INPUT FORM
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Loan Application Simulator</h3></div>""", unsafe_allow_html=True)

    left, right, result = st.columns([1, 1, 1])

    with left:
        st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#5A7A9F;border-bottom:1px solid #1E3A5F;padding-bottom:6px;margin-bottom:12px;'>👤 Data Pribadi</p>", unsafe_allow_html=True)
        inp_age    = st.number_input("Usia (tahun)",             min_value=15, max_value=100, value=28, step=1)
        inp_gender = st.selectbox("Gender",                      ["male", "female"])
        inp_edu    = st.selectbox("Tingkat Pendidikan",          EDUCATION_ORDER)
        inp_income = st.number_input("Pendapatan Tahunan (Rp)",  min_value=0, value=72000000, step=1000000)
        inp_exp    = st.number_input("Pengalaman Kerja (tahun)", min_value=0, max_value=60, value=4, step=1)
        inp_home   = st.selectbox("Kepemilikan Rumah",           ["RENT", "OWN", "MORTGAGE", "OTHER"])

    with right:
        st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#5A7A9F;border-bottom:1px solid #1E3A5F;padding-bottom:6px;margin-bottom:12px;'>💳 Data Kredit & Pinjaman</p>", unsafe_allow_html=True)
        inp_loan   = st.number_input("Jumlah Pinjaman (Rp)",     min_value=0, value=18000000, step=500000)
        inp_intent = st.selectbox("Tujuan Pinjaman",             ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
        inp_rate   = st.number_input("Suku Bunga (%)",           min_value=0.0, max_value=50.0, value=11.5, step=0.1)
        inp_score  = st.slider("Credit Score",                   300, 850, 680)
        inp_hist   = st.number_input("Riwayat Kredit (tahun)",   min_value=0, max_value=50, value=4, step=1)
        inp_prev   = st.selectbox("Pernah Gagal Bayar?",         ["N", "Y"])

        calc_dti = inp_loan / inp_income if inp_income > 0 else 0.0
        st.markdown(f"<p style='font-size:0.78rem;color:#8BA7C7;margin-top:8px;'>DTI Otomatis: <strong style='color:#F4A261;font-family:JetBrains Mono;'>{calc_dti:.2%}</strong></p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Prediksi Kelayakan Pinjaman")

    with result:
        st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#5A7A9F;border-bottom:1px solid #1E3A5F;padding-bottom:6px;margin-bottom:12px;'>📋 Hasil Prediksi</p>", unsafe_allow_html=True)

        if predict_btn:
            try:
                prev_encoded = 1 if inp_prev == "Y" else 0

                # ══════════════════════════════════════════
                #  LAYER 1 — BUSINESS RULES CHECK
                # ══════════════════════════════════════════
                rule_result = check_all_rules(
                    income        = inp_income,
                    loan_amount   = inp_loan,
                    dti_ratio     = calc_dti,
                    credit_score  = inp_score,
                    interest_rate = inp_rate,
                    credit_hist   = inp_hist,
                    employee_exp  = inp_exp,
                    age           = inp_age,
                    prev_default  = prev_encoded,
                    loan_intent   = inp_intent,
                )

                final_dec = rule_result["final_decision"]

                # KONDISI 1 HARD REJECT (SKIP ML)
                if final_dec == "HARD_REJECT":
                    p_approve = 0.0
                    p_reject  = 100.0
                    
                    st.markdown(f"""
                    <div class='pred-rejected'>
                        <div class='pred-icon'>🚫</div>
                        <div class='pred-label'>Business Rule — Hard Reject</div>
                        <div class='pred-status-rejected'>DITOLAK</div>
                        <div class='pred-prob' style='color:#F47461;font-size:0.78rem;margin-top:8px;'>{rule_result["reject_reason"]}</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;color:#F47461;margin-bottom:6px;'>Rules yang Dilanggar</p>", unsafe_allow_html=True)
                    for r in rule_result["hard_rejects"]:
                        st.markdown(f"""
                        <div style='background:#2A0F0F;border-left:3px solid #F47461;border-radius:6px;
                             padding:8px 12px;margin-bottom:6px;font-size:0.75rem;color:#CFA0A0;'>
                            <strong style='color:#F47461;'>{r.rule_name}</strong><br>
                            {r.message}<br>
                            <span style='color:#6E4242;'>Nilai: {r.actual_value} | Batas: {r.threshold}</span>
                        </div>""", unsafe_allow_html=True)

                # KONDISI 2 PASS / WARN (Kombinasi Aturan + Model ML)
                else:
                    raw_input = pd.DataFrame({
                        "Age":                  [inp_age],
                        "Gender":               [inp_gender],
                        "Education":            [inp_edu],
                        "Person Income":        [inp_income],
                        "Employee Experience":  [inp_exp],
                        "Home Onwership":       [inp_home],
                        "Loan Amount":          [inp_loan],
                        "Loan Intent":          [inp_intent],
                        "Loan interest Rate":   [inp_rate],
                        "Loan percentage":      [calc_dti],
                        "Credit History":       [inp_hist],
                        "Credit Score":         [inp_score],
                        "Previous Loan":        [prev_encoded],
                    })

                    input_df  = add_engineered_features(raw_input)
                    feat_cols = meta.get("feature_columns", list(input_df.columns))
                    feat_cols = [c for c in feat_cols if c in input_df.columns]
                    input_df  = input_df[feat_cols]

                    # Perolehan probabilitas dasar dari model ensemble
                    prob      = pipe.predict_proba(input_df)[0]
                    raw_p_approve = prob[1]

                    # Implementasi Sistem Poin Penalti Aturan Bisnis
                    penalty   = rule_result["confidence_penalty"]
                    effective_p_approve = max(0.0, raw_p_approve - penalty)
                    
                    p_approve = effective_p_approve * 100
                    p_reject  = 100.0 - p_approve

                    # Konversi keputusan akhir berdasarkan threshold dinamis pasca-penalti (50%)
                    effective_pred = 1 if effective_p_approve >= 0.50 else 0

                    # Render Visual Output Hasil Kelayakan
                    if effective_pred == 1:
                        warn_note = f"<div style='font-size:0.70rem;color:#8BA7C7;margin-top:4px;'>Setelah pinalti {penalty*100:.0f}% dari {len(rule_result['warnings'])} indikator risiko</div>" if penalty > 0 else ""
                        st.markdown(f"""
                        <div class='pred-approved'>
                            <div class='pred-icon'>✅</div>
                            <div class='pred-label'>Business Rules ✓ + ML Ensemble</div>
                            <div class='pred-status-approved'>DISETUJUI</div>
                            <div class='pred-prob' style='color:#52D9A3;'>Confidence: {p_approve:.1f}%</div>
                            {warn_note}
                        </div>""", unsafe_allow_html=True)
                    else:
                        warn_note = f"<div style='font-size:0.70rem;color:#8BA7C7;margin-top:4px;'>Awal: {raw_p_approve*100:.1f}% → Pasca-pinalti: {p_approve:.1f}%</div>" if penalty > 0 else ""
                        st.markdown(f"""
                        <div class='pred-rejected'>
                            <div class='pred-icon'>❌</div>
                            <div class='pred-label'>Business Rules ✓ + ML Ensemble</div>
                            <div class='pred-status-rejected'>DITOLAK</div>
                            <div class='pred-prob' style='color:#F47461;'>Risiko Penolakan: {p_reject:.1f}%</div>
                            {warn_note}
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Tampilkan daftar peringatan aktif jika ada pinalti aturan bisnis
                    if rule_result["warnings"]:
                        st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;color:#F4A261;margin-bottom:4px;'>⚠ Peringatan Risiko Aktif</p>", unsafe_allow_html=True)
                        for w in rule_result["warnings"]:
                            st.markdown(f"""
                            <div style='background:#2A1A0F;border-left:3px solid #F4A261;border-radius:6px;
                                 padding:6px 10px;margin-bottom:4px;font-size:0.72rem;color:#C4A280;'>
                                <strong style='color:#F4A261;'>{w.rule_name}</strong> — {w.message}
                            </div>""", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                # ELEMEN BERSAMA

                # Render Bar Faktor Risiko Struktural
                st.markdown("<p style='font-size:0.68rem;font-weight:600;text-transform:uppercase;color:#5A7A9F;margin-bottom:6px;'>Faktor Risiko Utama</p>", unsafe_allow_html=True)
                cs_pct = max(0, min((inp_score - 300) / 550 * 100, 100))
                dti_disp = min(calc_dti, 2.0)
                st.markdown(risk_bar_html("Credit Score",   str(inp_score),         cs_pct,                     score_to_color(cs_pct)),                    unsafe_allow_html=True)
                st.markdown(risk_bar_html("DTI Ratio",      f"{calc_dti*100:.1f}%",   min(dti_disp/0.65*100,100), score_to_color(100-min(dti_disp/0.65*100,100))), unsafe_allow_html=True)
                st.markdown(risk_bar_html("Suku Bunga",     f"{inp_rate:.1f}%",     min(inp_rate/36*100,100),   score_to_color(100-min(inp_rate/36*100,100))), unsafe_allow_html=True)
                st.markdown(risk_bar_html("Riwayat Kredit", f"{inp_hist} thn",      min(inp_hist/20*100,100),   score_to_color(min(inp_hist/20*100,100))),   unsafe_allow_html=True)

                # Render Plotly Gauge Probabilitas Akhir
                st.markdown("<br>", unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_approve,
                    title={"text": "Probabilitas Kelayakan Akhir", "font": {"color": FONT_COLOR, "size": 11}},
                    number={"suffix": "%", "font": {"color": "#F4A261", "size": 24, "family": "JetBrains Mono"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
                        "bar": {"color": "#52D9A3" if p_approve >= 50 else "#F47461"},
                        "bgcolor": "#0D1B2A",
                        "bordercolor": "#1E3A5F",
                        "steps": [
                            {"range": [0, 40],  "color": "#2A0F0F"},
                            {"range": [40, 60], "color": "#2A1A0F"},
                            {"range": [60, 100],"color": "#0F2A1E"},
                        ],
                        "threshold": {"line": {"color": "#F4A261", "width": 2}, "value": 50},
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                    font=dict(color=FONT_COLOR), height=200, margin=dict(t=30,b=0,l=20,r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Pemetaan tambahan Info Box Peringatan Cepat Koperasi
                flags = []
                if inp_prev == "Y":     flags.append("Riwayat gagal bayar terdeteksi aktiva macet")
                if calc_dti > 0.35:     flags.append("Rasio cicilan bulanan (DTI) berada di zona merah (>35%)")
                if inp_score < 580:     flags.append("Peringkat skor kredit internal di bawah standar ideal")
                if inp_rate > 20:       flags.append("Suku bunga tinggi berpotensi memicu default sistemik")

                if flags:
                    st.markdown("<br>", unsafe_allow_html=True)
                    flag_html = "<div class='warning-box'><strong>Ringkasan Risiko Lapangan:</strong><br>" + "<br>".join(flags) + "</div>"
                    st.markdown(flag_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Kalkulasi Prediksi Gagal: {e}")
                st.info("Saran: Periksa apakah file loan_rule.py sudah disimpan di direktori yang sama dan fungsi check_all_rules mengembalikan kamus data yang sesuai.")

        else:
            st.markdown("""
            <div style='background:#0D1B2A;border:1px dashed #1E3A5F;border-radius:12px;padding:40px;text-align:center;margin-top:8px;'>
                <div style='font-size:2.5rem;opacity:0.3;margin-bottom:10px;'>🏦</div>
                <div style='font-size:0.78rem;color:#3D5A7A;line-height:1.7;'>
                    Isi parameter data aplikasi di panel kiri<br>
                    kemudian klik tombol <strong style='color:#5A7A9F;'>Prediksi Kelayakan</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Nasabah Serupa ──
    st.markdown("""<div class='section-header'><div class='section-dot'></div><h3>Konteks: Nasabah Serupa di Dataset</h3></div>""", unsafe_allow_html=True)
    if predict_btn:
        try:
            similar = raw_df[
                raw_df["Credit Score"].between(inp_score - 60, inp_score + 60) &
                (raw_df["Loan Intent"] == inp_intent)
            ][["Age", "Credit Score", "Loan Amount", "Loan interest Rate", "Loan percentage", "Loan Status"]].head(8)

            if not similar.empty:
                similar["Loan Status"] = similar["Loan Status"].map({1: "✅ Approved", 0: "❌ Rejected"})
                similar.columns = ["Usia", "Credit Score", "Loan Amt (Rp)", "Rate (%)", "DTI", "Status"]
                st.dataframe(similar.reset_index(drop=True), use_container_width=True, height=240)
            else:
                st.markdown("<p style='font-size:0.78rem;color:#3D5A7A;'>Tidak ada nasabah serupa di dataset.</p>", unsafe_allow_html=True)
        except Exception:
            pass