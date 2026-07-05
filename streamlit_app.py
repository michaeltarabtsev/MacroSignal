"""
MacroSignal — Streamlit Web App
================================
Deploys Section 1 (Macro Regime Analyser) as a live web app.
Section 5 results are shown as pre-computed validated backtest stats.

Deploy: streamlit run streamlit_app.py
"""

import os
import csv
import time
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MacroSignal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0f1e;
    color: #e2e8f0;
  }
  .block-container { padding: 2rem 3rem; max-width: 1200px; }

  /* Header */
  .ms-header {
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }
  .ms-logo {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
  }
  .ms-tagline {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 2px;
    letter-spacing: 0.04em;
  }

  /* Score bar */
  .score-bar-wrap {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }

  /* Stat cards */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 1.2rem 0;
  }
  .stat-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 14px 16px;
  }
  .stat-label {
    font-size: 10px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }
  .stat-value {
    font-size: 22px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }
  .stat-sub {
    font-size: 10px;
    color: #475569;
    margin-top: 3px;
  }

  /* Regime badge */
  .regime-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  /* Layer breakdown */
  .layer-card {
    background: #0a0f1e;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 12px 14px;
  }
  .layer-label {
    font-size: 10px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }
  .layer-value {
    font-size: 15px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Allocation cards */
  .alloc-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 14px;
  }
  .alloc-title {
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 4px;
  }
  .alloc-dir {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #38bdf8;
    margin-bottom: 8px;
  }
  .alloc-text {
    font-size: 11px;
    color: #94a3b8;
    line-height: 1.5;
  }

  /* Backtest section */
  .backtest-wrap {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 2rem;
  }
  .backtest-title {
    font-size: 11px;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 10px;
    margin-bottom: 14px;
  }

  /* Event table */
  .event-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
  }
  .event-table th {
    color: #475569;
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid #1e293b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: 'Inter', sans-serif;
  }
  .event-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #0f172a;
  }

  /* Streamlit overrides */
  .stTextArea textarea {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    border-radius: 6px !important;
  }
  .stButton > button {
    background: #38bdf8 !important;
    color: #0a0f1e !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
    font-size: 13px !important;
    letter-spacing: 0.03em;
  }
  .stButton > button:hover {
    background: #7dd3fc !important;
  }
  div[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 12px 14px;
  }
  .stSpinner > div { color: #38bdf8 !important; }
  footer { display: none; }
  #MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    zeroshot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert   = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return zeroshot, tokenizer, finbert


# ── Lexicons (same as notebook) ────────────────────────────────────────────────
HAWKISH_LEXICON = [
    ("rate hike", 3), ("rate increase", 3), ("raise rates", 3),
    ("raising rates", 3), ("higher rates", 3), ("contractionary", 3),
    ("tightening cycle", 3), ("policy tightening", 3),
    ("quantitative tightening", 3), ("balance sheet reduction", 3),
    ("sustained elevated rates", 3), ("elevated rates", 2),
    ("rates higher for longer", 3), ("higher for longer", 2),
    ("above target", 2), ("persistent inflation", 2), ("sticky inflation", 2),
    ("inflation elevated", 2), ("inflation risks", 2), ("price pressures", 2),
    ("inflationary", 2), ("services inflation", 2), ("core inflation", 1),
    ("restrictive", 2), ("sufficiently restrictive", 3), ("tighten", 2),
    ("tightening", 2), ("vigilant", 2), ("resolute", 2),
    ("committed to", 2), ("further action", 2), ("additional hikes", 3),
    ("further increases", 2), ("not yet done", 2),
    ("strong labour", 1), ("tight labour", 1), ("low unemployment", 1),
    ("strong demand", 1), ("robust growth", 1), ("overheating", 2),
    ("consumption remains strong", 2), ("wage growth", 1),
]
DOVISH_LEXICON = [
    ("rate cut", 3), ("rate reduction", 3), ("cut rates", 3),
    ("cutting rates", 3), ("lower rates", 3), ("reduce rates", 3),
    ("easing cycle", 3), ("policy easing", 3), ("quantitative easing", 3),
    ("asset purchases", 2), ("bond buying", 2), ("accommodative", 3),
    ("accommodation", 2), ("stimulus", 2),
    ("pause", 2), ("hold rates", 2), ("patient", 2), ("wait and see", 2),
    ("data dependent", 1), ("flexible", 1), ("stand ready", 2),
    ("support growth", 2), ("supportive", 2), ("gradual", 1),
    ("disinflation", 2), ("inflation declining", 2), ("below target", 2),
    ("inflation target achieved", 3), ("price stability restored", 2),
    ("cooling inflation", 2), ("easing inflation", 2), ("inflation falling", 2),
    ("slowing growth", 2), ("economic slowdown", 2), ("recession", 2),
    ("downside risks", 2), ("weak demand", 2), ("labour market softening", 2),
    ("unemployment rising", 2), ("growth below potential", 2),
]
NEGATION_WORDS = {
    "not", "no", "won't", "wont", "don't", "dont", "didn't", "didnt",
    "never", "neither", "nor", "cannot", "can't", "cant", "isn't", "isnt",
    "aren't", "arent", "wasn't", "wasnt", "weren't", "werent", "hasn't",
    "hasnt", "haven't", "havent", "unlikely", "ruled out", "rule out",
    "dismissed", "rejected", "no longer", "not yet", "not currently", "no further",
}
NEGATION_WINDOW = 6

ZEROSHOT_LABELS = [
    "hawkish monetary policy tightening",
    "dovish monetary policy easing",
    "neutral monetary policy",
]


# ── Scoring functions ──────────────────────────────────────────────────────────
def is_negated(text_lower, phrase, match_pos):
    preceding = text_lower[:match_pos].split()[-NEGATION_WINDOW:]
    for neg in NEGATION_WORDS:
        if " " in neg:
            if neg in " ".join(preceding):
                return True
        else:
            if neg in preceding:
                return True
    return False


def score_lexicon(text):
    text_lower = text.lower()
    hawk_score = dove_score = 0.0
    matched_hawk, matched_dove = [], []
    for phrase, weight in HAWKISH_LEXICON:
        pos = text_lower.find(phrase)
        if pos != -1:
            if is_negated(text_lower, phrase, pos):
                dove_score += weight
                matched_dove.append((f"NOT {phrase}", weight))
            else:
                hawk_score += weight
                matched_hawk.append((phrase, weight))
    for phrase, weight in DOVISH_LEXICON:
        pos = text_lower.find(phrase)
        if pos != -1:
            if is_negated(text_lower, phrase, pos):
                hawk_score += weight
                matched_hawk.append((f"NOT {phrase}", weight))
            else:
                dove_score += weight
                matched_dove.append((phrase, weight))
    return hawk_score, dove_score, matched_hawk, matched_dove


def score_finbert(text, tokenizer, finbert):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 8]
    if not sentences:
        sentences = [text]
    scores = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            logits = finbert(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
        scores.append(float(probs[1] - probs[0]))
    return float(np.mean(scores))


def compute_macro_score(text, zeroshot, tokenizer, finbert):
    # Layer 1
    result     = zeroshot(text[:2000], ZEROSHOT_LABELS, multi_label=False)
    scores_map = {l: s for l, s in zip(result["labels"], result["scores"])}
    zs_raw     = scores_map.get("hawkish monetary policy tightening", 0) - \
                 scores_map.get("dovish monetary policy easing", 0)

    # Layer 2
    hawk_raw, dove_raw, matched_hawk, matched_dove = score_lexicon(text)
    lex_score = float(np.tanh((hawk_raw - dove_raw) / 5.0))

    # Layer 3
    fb_raw = score_finbert(text, tokenizer, finbert)

    composite = 0.40 * zs_raw + 0.40 * lex_score + 0.20 * fb_raw
    return composite, zs_raw, hawk_raw, dove_raw, fb_raw, matched_hawk, matched_dove


# ── Pre-computed Section 5 backtest results ────────────────────────────────────
BACKTEST_EVENTS = [
    ("2018-01-31", "DOVISH",   -0.354, -0.72, +1.56, -1.56),
    ("2018-03-21", "DOVISH",   -0.363, +0.46, -0.87, +0.87),
    ("2018-06-13", "DOVISH",   -0.348, +0.00, +0.30, -0.30),
    ("2018-09-26", "DOVISH",   -0.203, +1.00, +2.12, +2.12),
    ("2018-12-19", "DOVISH",   -0.202, -0.48, +0.58, -0.58),
    ("2019-03-20", "DOVISH",   -0.227, -1.72, -2.58, +2.58),
    ("2019-06-19", "DOVISH",   -0.115, +0.36, +0.69, +0.69),
    ("2019-09-18", "DOVISH",   -0.116, -1.30, -1.02, +1.02),
    ("2019-12-11", "DOVISH",   -0.119, -0.25, +0.71, +0.71),
    ("2020-03-03", "DOVISH",   -0.232, -7.82, -5.07, +5.07),
    ("2020-03-15", "DOVISH",   -0.253, +11.93, +2.73, +2.73),
    ("2020-06-10", "HAWKISH",  +0.050, +0.91, +2.03, -0.91),
    ("2020-09-16", "DOVISH",   -0.112, -0.20, -0.25, -0.20),
    ("2021-03-17", "DOVISH",   -0.310, -1.75, -3.22, +1.75),
    ("2021-06-16", "DOVISH",   -0.366, -0.17, -0.16, -0.17),
    ("2021-11-03", "DOVISH",   -0.336, -1.31, -0.75, -1.31),
    ("2022-03-16", "HAWKISH",  +0.054, +1.12, +0.25, -1.12),
    ("2022-05-04", "HAWKISH",  +0.204, +0.61, -2.23, -0.61),
    ("2022-06-15", "HAWKISH",  +0.280, +1.38, -2.17, -1.38),
    ("2022-09-21", "HAWKISH",  +0.283, +1.51, +0.89, -1.51),
    ("2022-12-14", "HAWKISH",  -0.370, +3.76, +3.47, +3.76),
    ("2023-02-01", "HAWKISH",  +0.312, -0.88, -1.23, -0.88),
    ("2023-05-03", "HAWKISH",  +0.335, -0.43, -0.65, -0.43),
    ("2023-09-20", "HAWKISH",  +0.303, +0.38, +0.55, -0.38),
]

BACKTEST_STATS = {
    "hit_rate":     62.5,
    "ic":          +0.348,
    "p_value":      0.048,
    "sharpe":       0.85,
    "max_dd":      -2.7,
    "total_ret":  +12.9,
}


# ── Live market data ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_market_data():
    try:
        spy = round(yf.Ticker("SPY").history(period="2d")['Close'].iloc[-1], 2)
        vix = round(yf.Ticker("^VIX").history(period="2d")['Close'].iloc[-1], 2)
        tlt = round(yf.Ticker("TLT").history(period="2d")['Close'].iloc[-1], 2)
        return spy, vix, tlt
    except Exception:
        return None, None, None



# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DB_FILE          = "trading_history.csv"
FRICTION_BPS     = 5.0
BASE_CAPITAL     = 1_000_000
CONFIDENCE_FLOOR = 0.60
HEADERS = ["Timestamp","Ticker","Direction","Entry_Price",
           "Allocation_Weight_Pct","Notional_Size_USD","Execution_Cost_USD"]
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY","") if hasattr(st,"secrets") else ""

ASSET_POOL = {
    "FIXED_INCOME":{"Class":"Sovereign Fixed Income","Candidates":[
        {"Ticker":"ZT=F","Desc":"2-Year Treasury Futures","Keywords":["rate","fed","fomc","powell","hike","cut","central bank","monetary"],"regime_bias":"hawkish"},
        {"Ticker":"ZN=F","Desc":"10-Year Treasury Futures","Keywords":["yield","10-year","duration","treasury","curve"],"regime_bias":"neutral"},
        {"Ticker":"TLT","Desc":"iShares 20+ Year Treasury ETF","Keywords":["bonds","tlt","long-duration","fixed income","treasuries"],"regime_bias":"dovish"},
    ]},
    "ENERGY_COMMODITIES":{"Class":"Energy & Commodities","Candidates":[
        {"Ticker":"CL=F","Desc":"WTI Crude Oil Futures","Keywords":["oil","crude","barrel","brent","wti","opec","energy"],"regime_bias":"neutral"},
        {"Ticker":"GC=F","Desc":"Gold Futures","Keywords":["gold","precious","metal","safe-haven","bullion","inflation"],"regime_bias":"dovish"},
        {"Ticker":"XLE","Desc":"Energy Select Sector SPDR","Keywords":["shale","drilling","exxon","chevron","gasoline"],"regime_bias":"neutral"},
    ]},
    "DEFENSIVES":{"Class":"Defensive Equities","Candidates":[
        {"Ticker":"XLP","Desc":"Consumer Staples SPDR","Keywords":["consumer","retail","sales","spending","staples"],"regime_bias":"hawkish"},
        {"Ticker":"XLV","Desc":"Health Care SPDR","Keywords":["healthcare","medical","pharma","clinical","drug"],"regime_bias":"hawkish"},
        {"Ticker":"XLU","Desc":"Utilities SPDR","Keywords":["utilities","power","electricity","infrastructure"],"regime_bias":"neutral"},
    ]},
    "TECH_GROWTH":{"Class":"Technology & Growth","Candidates":[
        {"Ticker":"NVDA","Desc":"NVIDIA — AI compute leader","Keywords":["nvidia","gpu","semiconductor","ai","chips"],"regime_bias":"dovish"},
        {"Ticker":"MSFT","Desc":"Microsoft — cloud & software","Keywords":["microsoft","cloud","openai","software","azure"],"regime_bias":"dovish"},
        {"Ticker":"GOOGL","Desc":"Alphabet — search & AI","Keywords":["google","alphabet","search","gemini"],"regime_bias":"dovish"},
    ]},
    "BROAD_MARKET":{"Class":"Broad Market","Candidates":[
        {"Ticker":"SPY","Desc":"S&P 500 ETF","Keywords":["market","s&p","stocks","wall street","index","economy"],"regime_bias":"neutral"},
        {"Ticker":"IWM","Desc":"Russell 2000 ETF","Keywords":["small-cap","russell","domestic","growth"],"regime_bias":"dovish"},
        {"Ticker":"^VIX","Desc":"VIX — fear gauge","Keywords":["vix","volatility","fear","risk-off","uncertainty"],"regime_bias":"hawkish"},
    ]},
}

def initialise_ledger():
    if not os.path.exists(DB_FILE) or os.stat(DB_FILE).st_size == 0:
        pd.DataFrame(columns=HEADERS).to_csv(DB_FILE, index=False)

def load_ledger():
    initialise_ledger()
    try:
        df = pd.read_csv(DB_FILE)
        return df if not df.empty and "Timestamp" in df.columns else pd.DataFrame(columns=HEADERS)
    except Exception:
        return pd.DataFrame(columns=HEADERS)

def clear_ledger():
    pd.DataFrame(columns=HEADERS).to_csv(DB_FILE, index=False)

def get_live_price(ticker, fallback=150.0):
    for period in ("1d","5d","1mo"):
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    return round(float(closes.iloc[-1]), 2)
        except Exception:
            pass
    return fallback

def fetch_headlines():
    if NEWS_API_KEY:
        try:
            url  = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            hls  = [a["title"] for a in resp.json().get("articles",[]) if a.get("title")]
            if hls: return hls[:10], True
        except Exception:
            pass
    return [
        "[DEMO] Federal Reserve holds rates steady, signals data-dependent path.",
        "[DEMO] OPEC cuts oil production targets amid slowing global demand.",
        "[DEMO] NVIDIA reports record quarterly revenue driven by AI chip demand.",
        "[DEMO] US inflation falls to 2.3%, below consensus forecast of 2.6%.",
        "[DEMO] Treasury yields climb as strong jobs data reduces rate cut expectations.",
        "[DEMO] Microsoft Azure cloud revenue grows 29% year-on-year.",
        "[DEMO] Gold hits new highs as investors seek safe-haven assets.",
        "[DEMO] Consumer spending unexpectedly declines in latest retail data.",
        "[DEMO] ECB signals further rate cuts as Eurozone growth stalls.",
        "[DEMO] S&P 500 reaches all-time high amid improving earnings season.",
    ], False

def calculate_position_size(ticker, confidence):
    try:
        hist      = yf.Ticker(ticker).history(period="1mo")["Close"]
        daily_vol = hist.pct_change().dropna().std()
        raw_w     = 0.01 / daily_vol if daily_vol > 0 else 0.15
    except Exception:
        raw_w = 0.15
    scalar   = (confidence - CONFIDENCE_FLOOR) / (1 - CONFIDENCE_FLOOR)
    weight   = min(max(raw_w * scalar, 0.05), 0.25)
    notional = round(BASE_CAPITAL * weight, 2)
    friction = round(notional * (FRICTION_BPS / 10_000), 2)
    return round(weight * 100, 1), notional, friction

def calc_pnl(entry, live, notional, direction):
    units = notional / entry
    if direction == "LONG":
        current_val = units * live
        pnl_pct     = ((live - entry) / entry) * 100
    else:
        pnl_pct     = max((entry - live) / entry, -1.0) * 100
        current_val = notional * (1 + pnl_pct / 100)
    return round(current_val,2), round(current_val-notional,2), round(pnl_pct,2)

def stop_distance(entry, live, direction):
    stop = entry * 0.97 if direction == "LONG" else entry * 1.03
    dist = ((live-stop)/live*100) if direction=="LONG" else ((stop-live)/live*100)
    return round(dist,2), dist < 1.0

def time_open(ts):
    try:
        opened = datetime.datetime.strptime(ts,"%Y-%m-%d %H:%M:%S")
        delta  = datetime.datetime.now() - opened
        hours  = int(delta.total_seconds()//3600)
        if hours<1:    return f"{int(delta.total_seconds()//60)}m"
        elif hours<24: return f"{hours}h"
        else:          return f"{delta.days}d {hours%24}h"
    except: return "—"

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 📡 MacroSignal")
st.sidebar.markdown("---")
section = st.sidebar.radio("Navigate", [
    "1️⃣  Macro Regime Analyser",
    "2️⃣  Trade Signal Generator",
    "3️⃣  Position Journal",
    "4️⃣  Portfolio Dashboard",
    "5️⃣  FOMC Backtest",
])
st.sidebar.markdown("---")
st.sidebar.markdown("""**Run order:**
1. Macro Regime Analyser
2. Trade Signal Generator
3. Position Journal
4. Portfolio Dashboard
5. FOMC Backtest (standalone)
""")
st.sidebar.markdown("---")
st.sidebar.caption("Built by Misha · Warwick University")

if "macro_score" not in st.session_state:
    st.session_state.macro_score = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1
# ══════════════════════════════════════════════════════════════════════════════
if section == "1️⃣  Macro Regime Analyser":

    # Header
    st.markdown("""
    <div class="ms-header">
      <div class="ms-logo">📡 MacroSignal</div>
      <div class="ms-tagline">NLP-driven macro regime classification · Three-layer architecture · FOMC validated</div>
    </div>
    """, unsafe_allow_html=True)

    # Live market strip
    spy, vix, tlt = get_market_data()
    if spy:
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""<div class="stat-card">
            <div class="stat-label">SPY</div>
            <div class="stat-value" style="color:#38bdf8;">${spy}</div>
            <div class="stat-sub">S&P 500 ETF</div></div>""", unsafe_allow_html=True)
        col2.markdown(f"""<div class="stat-card">
            <div class="stat-label">VIX</div>
            <div class="stat-value" style="color:#f59e0b;">{vix}</div>
            <div class="stat-sub">Implied volatility</div></div>""", unsafe_allow_html=True)
        col3.markdown(f"""<div class="stat-card">
            <div class="stat-label">TLT</div>
            <div class="stat-value" style="color:#a78bfa;">${tlt}</div>
            <div class="stat-sub">20Y Treasury ETF</div></div>""", unsafe_allow_html=True)
        col4.markdown(f"""<div class="stat-card">
            <div class="stat-label">Backtest IC</div>
            <div class="stat-value" style="color:#10b981;">+0.348</div>
            <div class="stat-sub">p = 0.048 · significant</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input section
    st.markdown("### Macro Regime Analyser")
    st.markdown("<p style='color:#64748b;font-size:13px;'>Paste any central bank statement, Fed minutes excerpt, or macro commentary. The classifier reads it through three NLP layers and outputs a hawkish/dovish score with cross-asset positioning guidance.</p>", unsafe_allow_html=True)

    default_text = (
    "Central bank officials emphasised contractionary settings, pointing to sticky "
    "services inflation. The board signalled commitment to sustained elevated rates "
    "until consumption cools further."
    )

    user_text = st.text_area(
    label="Input text",
    value=default_text,
    height=110,
    label_visibility="collapsed",
    placeholder="Paste Fed statement, ECB press release, Bloomberg macro summary..."
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        run_clicked = st.button("Analyse →", use_container_width=True)

    # ── Analysis output ────────────────────────────────────────────────────────────
    if run_clicked and user_text.strip():
        with st.spinner("Loading NLP models and scoring text..."):
            zeroshot_model, tok, fb_model = load_models()

        with st.spinner("Running three-layer classification..."):
            composite, zs_raw, hawk_raw, dove_raw, fb_raw, matched_hawk, matched_dove = \
                compute_macro_score(user_text.strip(), zeroshot_model, tok, fb_model)

    # Regime
    if composite > 0.05:
        regime       = "HAWKISH"
        regime_color = "#10b981"
        badge_bg     = "#064e3b"
    elif composite < -0.05:
        regime       = "DOVISH"
        regime_color = "#f43f5e"
        badge_bg     = "#450a0a"
    else:
        regime       = "NEUTRAL"
        regime_color = "#38bdf8"
        badge_bg     = "#0c2a3d"

    st.markdown("<br>", unsafe_allow_html=True)

    # Score display
    bar_width  = min(abs(composite) / 0.5 * 100, 100)
    bar_color  = regime_color
    bar_dir    = "left" if composite < 0 else "right"
    bar_margin = "auto 0" if composite >= 0 else "0 auto"

    score_sign = "+" if composite >= 0 else ""

    st.markdown(f"""
    <div class="score-bar-wrap">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div>
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Composite Score</div>
          <div style="font-size:36px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{regime_color};">
            {score_sign}{composite:.3f}
          </div>
        </div>
        <div style="background:{badge_bg};border:1px solid {regime_color};color:{regime_color};
                    padding:6px 18px;border-radius:4px;font-weight:700;font-size:13px;
                    letter-spacing:.05em;">
          {regime}
        </div>
      </div>

      <div style="position:relative;height:8px;background:#1e293b;border-radius:4px;overflow:hidden;">
        <div style="position:absolute;top:0;bottom:0;left:50%;right:auto;
                    width:{bar_width/2}%;
                    {'left:50%;' if composite >= 0 else f'right:50%;left:auto;width:{bar_width/2}%;'}
                    background:{bar_color};border-radius:4px;"></div>
        <div style="position:absolute;top:0;bottom:0;left:50%;width:1px;background:#334155;"></div>
      </div>

      <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:10px;color:#475569;">
        <span>← Dovish / Easing</span>
        <span>Neutral</span>
        <span>Hawkish / Tightening →</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Three-layer breakdown
    zs_label  = "Hawkish" if zs_raw > 0.05 else ("Dovish" if zs_raw < -0.05 else "Neutral")
    zs_color  = "#10b981" if zs_raw > 0.05 else ("#f43f5e" if zs_raw < -0.05 else "#38bdf8")
    fb_color  = "#10b981" if fb_raw > 0 else "#f43f5e"
    lex_net   = hawk_raw - dove_raw
    lex_color = "#10b981" if lex_net > 0 else ("#f43f5e" if lex_net < 0 else "#64748b")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="layer-card">
        <div class="layer-label">Layer 1 — Zero-Shot NLI (40%)</div>
        <div class="layer-value" style="color:{zs_color};">{zs_label} ({zs_raw:+.3f})</div>
        <div style="font-size:10px;color:#475569;margin-top:4px;">Reads meaning, no keywords needed</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""<div class="layer-card">
        <div class="layer-label">Layer 2 — Keyword Lexicon (40%)</div>
        <div class="layer-value" style="color:{lex_color};">Hawk {hawk_raw:.0f} vs Dove {dove_raw:.0f}</div>
        <div style="font-size:10px;color:#475569;margin-top:4px;">60+ terms · negation detection</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""<div class="layer-card">
        <div class="layer-label">Layer 3 — FinBERT Tone (20%)</div>
        <div class="layer-value" style="color:{fb_color};">{fb_raw:+.3f}</div>
        <div style="font-size:10px;color:#475569;margin-top:4px;">Financial sentiment refinement</div>
    </div>""", unsafe_allow_html=True)

    # Matched terms
    if matched_hawk or matched_dove:
        st.markdown("<br>", unsafe_allow_html=True)
        if matched_hawk:
            hawk_str = ", ".join([f"{p} (+{w})" for p, w in matched_hawk])
            st.markdown(f"<div style='font-size:11px;color:#64748b;'>🟢 <span style='color:#10b981;font-family:JetBrains Mono,monospace;'>{hawk_str}</span></div>", unsafe_allow_html=True)
        if matched_dove:
            dove_str = ", ".join([f"{p} (+{w})" for p, w in matched_dove])
            st.markdown(f"<div style='font-size:11px;color:#64748b;margin-top:4px;'>🔴 <span style='color:#f43f5e;font-family:JetBrains Mono,monospace;'>{dove_str}</span></div>", unsafe_allow_html=True)

    # Asset allocation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Cross-Asset Positioning Guidance**")

    if regime == "HAWKISH":
        allocs = [
            ("💼 Equities",     "Underweight",    "Rotate toward cash-generative defensives — consumer staples, utilities. Reduce rate-sensitive growth exposure."),
            ("📈 Fixed Income", "Short duration", "Shorten bond duration. Prefer floating-rate notes and T-bills over long-dated Treasuries."),
            ("⚡ Derivatives",  "Hedged",         "Use protective collars or put spreads on core equity positions. Elevated rates increase carry costs."),
            ("🌍 FX / Macro",   "Long USD",       "A hawkish Fed typically supports USD. Consider long USD vs rate-sensitive EM currencies."),
        ]
    elif regime == "DOVISH":
        allocs = [
            ("💼 Equities",     "Overweight",     "Increase exposure to high-beta growth and technology. Easing conditions support multiple expansion."),
            ("📈 Fixed Income", "Long duration",  "Extend bond duration. Rate cuts drive price appreciation in long-dated bonds — TLT benefits most."),
            ("⚡ Derivatives",  "Convex upside",  "Deploy call spreads or futures to amplify upside. Lower rates reduce option carry costs."),
            ("🌍 FX / Macro",   "Cyclical carry", "Dovish policy weakens domestic currency. Rotate into growth-correlated currencies — AUD, EM."),
        ]
    else:
        allocs = [
            ("💼 Equities",     "Neutral",        "Focus on stock-specific fundamentals. Strong free cash flow and pricing power matter most."),
            ("📈 Fixed Income", "Barbell",        "Hold a duration barbell: short-term T-bills for liquidity plus select intermediates for yield."),
            ("⚡ Derivatives",  "Income",         "Sell covered calls or iron condors to harvest theta in range-bound conditions."),
            ("🌍 FX / Macro",   "Range-bound",    "Carry trades within defined ranges. Monitor central bank divergence for the next directional move."),
        ]

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, (title, direction, text) in zip([col_a, col_b, col_c, col_d], allocs):
        col.markdown(f"""<div class="alloc-card">
            <div class="alloc-title">{title}</div>
            <div class="alloc-dir">{direction}</div>
            <div class="alloc-text">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='margin-top:1rem;font-size:11px;color:#475569;'>
        ⚠️ For educational purposes only — not financial advice. &nbsp;|&nbsp;
        <span style='color:#38bdf8;font-family:JetBrains Mono,monospace;'>macro_score = {composite:+.4f}</span>
    </div>""", unsafe_allow_html=True)


    # ── Section 5: Backtest results ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
# Footer
if section == "1️⃣  Macro Regime Analyser":
    st.markdown("""
    <div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1e293b;
            font-size:11px;color:#334155;display:flex;justify-content:space-between;">
      <span>MacroSignal · Built by Misha · Warwick University</span>
      <span>Three-layer NLP: Zero-Shot NLI · Keyword Lexicon · FinBERT</span>
    </div>
    """, unsafe_allow_html=True)
