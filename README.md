# MacroSignal — NLP-Driven Cross-Asset Trading System

---

## What is this project?

MacroSignal is a fully integrated trading signal system that reads financial news and central bank language, classifies the macroeconomic environment, and generates structured trade ideas across equities, bonds, commodities, and FX.

It was built to show what happens when you combine natural language processing with quantitative finance. Instead of a human analyst spending hours reading Fed statements and deciding how to position a portfolio, MacroSignal automates that first step — classifying whether the macro environment is hawkish or dovish — and connects it directly to asset allocation decisions, live position tracking, and historical validation.

The project runs end-to-end in a Jupyter notebook across five connected sections. Each section feeds the next, forming a complete pipeline from raw text input to backtested performance analytics.

---

## The big picture — how the five sections work together

Think of it as a desk workflow:

1. A macro analyst reads the Fed statement and decides the environment is hawkish or dovish **(Section 1)**
2. A trader reads the latest headlines and generates specific trade ideas based on that macro view **(Section 2)**
3. A risk manager monitors all open positions in real time **(Section 3)**
4. A portfolio manager reviews overall performance and risk metrics at end of day **(Section 4)**
5. A quant researcher runs a backtest to validate the model historically **(Section 5)**

MacroSignal does all five steps automatically.

---

## Section-by-Section Breakdown

### Section 1 — Macro Regime Analyser

**What it does:**
You paste in any piece of central bank or macroeconomic text — a Fed statement, an ECB press release, a Bloomberg news summary, anything. The system reads it and produces a single score that tells you whether the language leans hawkish (rates rising, tightening policy) or dovish (rates falling, easy policy).

**Why it matters:**
The direction of monetary policy is the single biggest driver of cross-asset returns. Hawkish environments favour defensive equities, short-duration bonds, and a strong dollar. Dovish environments favour growth stocks, long-duration bonds, and risk assets. Getting this classification right is the foundation of macro trading.

**How it works — three layers:**

*Layer 1 — Zero-Shot NLP (40% of the score)*
Uses `facebook/bart-large-mnli`, a large language model trained on natural language inference. It reads the full text and directly asks: does this passage support the hypothesis "hawkish monetary policy tightening" or "dovish monetary policy easing"? This catches implicit signals — phrases like "the committee remains attentive to upside inflation risks" score hawkish even though they contain no explicit rate language.

*Layer 2 — Keyword Lexicon with Negation Detection (40% of the score)*
A curated dictionary of 60+ macro terms weighted by signal strength. "Contractionary" scores +3 hawkish. "Rate cut" scores +3 dovish. Critically, a negation detection system looks back six words before every match — so "will not raise rates" correctly scores dovish, not hawkish.

*Layer 3 — FinBERT Sentiment (20% of the score)*
Uses `ProsusAI/finbert`, a financial sentiment model trained on earnings calls and financial news. Hawkish central bank language tends to carry negative financial tone; dovish language tends to be positive. This layer refines borderline cases without overriding the primary signal.

**Output:**
- A composite hawkish/dovish score with a dynamic chart
- A per-layer breakdown showing exactly why the text scored the way it did
- An asset allocation recommendation across equities, fixed income, derivatives, and FX
- Live SPY and VIX prices from Yahoo Finance for market context
- The `macro_score` variable passed automatically to Section 2

---

### Section 2 — Event-Driven Trade Signal Generator

**What it does:**
Fetches live financial headlines from NewsAPI and generates specific trade signals. Each signal includes the asset to trade, direction (LONG or SHORT), entry price, price target, stop loss, and position size.

**Why it matters:**
Macro regime alone is not enough — you need to act on specific events. A hawkish macro environment combined with a negative headline about inflation creates a much stronger SHORT signal than either would alone. This section captures that interaction.

**How it works:**
It reads `macro_score` from Section 1 before doing anything. The cross-asset logic is:

- Hawkish macro + negative headline = **HIGH conviction SHORT** (both signals agree, position size boosted)
- Hawkish macro + positive headline = **LOW conviction LONG** (signals conflict, position size reduced)
- Dovish macro + positive headline = **HIGH conviction LONG** (both signals agree, position size boosted)
- Dovish macro + negative headline = **LOW conviction SHORT** (signals conflict, position size reduced)
- Neutral macro = signal driven purely by headline sentiment

A confidence threshold filters out weak signals — if FinBERT scores below 60% confidence, no trade is generated rather than forcing a direction.

Position sizing is volatility-scaled: more volatile assets receive smaller allocations, and confidence scales the size up or down. Every trade is logged to a CSV ledger for Sections 3 and 4 to read.

**Output:**
- Live headlines from NewsAPI (or clearly labelled demo headlines without a key)
- Trade signal with LONG/SHORT direction and conviction level
- Three asset suggestions with entry, target, and stop prices
- Volatility-adjusted position sizes
- All trades logged to the ledger

---

### Section 3 — Position Journal and Live P&L Monitor

**What it does:**
Reads the trade ledger and shows a live breakdown of every open position — how much you made or lost, how close each position is to its stop loss, and how long each trade has been open.

**Why it matters:**
Generating a signal is only half the job. Knowing when to exit is the other half. This section gives you a real-time view of the book so you can see which positions are working and which are approaching their stop.

**How it works:**
For each trade in the ledger it fetches the current live price from Yahoo Finance (using a multi-period fallback so it works on weekends and holidays when `period=1d` returns nothing for futures). It then calculates P&L correctly for both long and short positions:

- Long P&L = (live price − entry price) / entry price
- Short P&L = (entry price − live price) / entry price, capped at −100%

Trades are grouped by the Section 2 signal block that generated them. The section also shows net P&L after friction costs, distance to stop loss with a warning if within 1%, and time elapsed since entry.

**Output:**
- Live P&L per position in % and USD
- Distance to stop loss with colour-coded warnings
- Time in trade
- Block-level summary with net return after friction
- Filter by ticker or direction (LONG/SHORT)
- Clear All Trades button to reset the ledger

---

### Section 4 — Portfolio Analytics Dashboard

**What it does:**
The end-of-day portfolio view. Aggregates all trades into a single performance report with charts showing cumulative P&L vs the S&P 500, per-asset breakdown, and risk metrics.

**Why it matters:**
Individual position P&L tells you whether a trade is working. Portfolio-level analytics tell you whether the *strategy* is working — whether you're generating alpha above a simple benchmark, whether your risk is concentrated in one asset, and how bad your worst losing streak has been.

**How it works:**
Reads the full trade ledger, fetches current prices using the same multi-period fallback as Section 3, and computes:

- **Cumulative P&L** over time with SPY benchmark overlay (scaled to the same notional)
- **Per-asset P&L** horizontal bar chart showing which positions made and lost money
- **Drawdown curve** showing the running peak-to-trough loss
- **Sharpe ratio** (mean daily P&L / standard deviation × √252)
- **Max drawdown** in % and USD
- **Largest single loss** by position

**Output:**
- Six summary metric cards (signal blocks, notional, net P&L, Sharpe, max drawdown, largest loss)
- Three charts: cumulative P&L vs SPY, per-asset breakdown, drawdown curve
- All figures shown after friction costs

---

### Section 5 — FOMC Historical Backtest

**What it does:**
Validates the Section 1 classifier on 24 real Federal Reserve statements from 2018 to 2023. For each FOMC meeting it fetches the actual statement from federalreserve.gov, runs it through the classifier, and measures how TLT (the 20+ year Treasury bond ETF) moved in the 1, 3, and 5 trading days afterwards.

**Why it matters:**
A model that looks good on current data might just be lucky. Backtesting on six years of real Fed statements across multiple rate cycles — hikes in 2018, cuts in 2019, emergency cuts in 2020, and the most aggressive tightening cycle in 40 years in 2022-2023 — gives a much more honest picture of whether the classifier actually works.

**Why TLT:**
TLT is the most direct market expression of Fed policy expectations. When the Fed signals rate hikes, bond prices fall and TLT drops. When it signals rate cuts, bond prices rise and TLT rises. It reacts immediately and strongly to policy surprises, making it the cleanest instrument to validate a hawkish/dovish classifier.

**Strategy rule:**
- Score above +0.05 → SHORT TLT (hawkish = bond prices should fall)
- Score below −0.05 → LONG TLT (dovish = bond prices should rise)
- Score between ±0.05 → FLAT (no trade, signal too weak)

**Backtest results (2018–2023, 24 FOMC events):**

| Metric | Result |
|--------|--------|
| Signal hit rate | **62.5%** (15 correct out of 24) |
| Information Coefficient | **+0.348** |
| Statistical significance | **p < 0.10** (significant) |
| Strategy Sharpe ratio | **0.85** |
| Total strategy return | **+12.9%** |
| Max drawdown | **−2.7%** |

The classifier correctly identified the full 2022-2023 rate hike cycle as hawkish and the 2019-2020 easing cycle as dovish. The strategy significantly outperformed buy-and-hold TLT over the same period, which ended in negative territory as rates rose.

**How results are displayed:**
- Summary metric cards (hit rate, IC, Sharpe, drawdown, total return)
- Bar chart of classifier score per FOMC event across the full period
- Cumulative return chart: strategy vs buy-and-hold TLT
- IC scatter plot: classifier score vs actual TLT move
- Full event-by-event table with TLT returns at +1d, +3d, +5d and strategy P&L

---

## Setup instructions

**1. Install dependencies**
```
pip install torch transformers yfinance pandas numpy matplotlib requests beautifulsoup4 scipy ipywidgets
```

**2. Add your NewsAPI key**
In the configuration block at the top of Section 2, replace the placeholder with your key (free at newsapi.org):
```python
NEWS_API_KEY = "your_key_here"
```

**3. Run sections in order**
Section 1 must run before Section 2 — it sets `macro_score` and loads `compute_macro_score()`.
Section 2 must run before Sections 3 and 4 — it populates the trade ledger.
Section 5 requires Section 1 to be loaded first.

**4. First run note**
On first run, Section 1 downloads the NLP models (approximately 1.5GB total). This takes a few minutes and only happens once — they are cached locally afterwards.

---

## Strengths and limitations

### What this project does well

**Genuine NLP architecture** — most student projects use keyword matching alone. The three-layer hybrid (zero-shot NLI + negation-aware lexicon + FinBERT) is a legitimate research design that solves real problems: implicit signals, negation handling, and borderline cases.

**End-to-end pipeline** — the five sections form a coherent workflow from text input to backtested results. Section 1 and Section 2 are tightly coupled through the macro regime variable, meaning trade signals are always contextualised by the broader environment.

**Validated results** — the Section 5 backtest produces a statistically significant IC of +0.348 and 62.5% hit rate across six years of real Fed statements. These numbers hold up to scrutiny.

**Honest risk metrics** — Sections 3 and 4 show P&L net of friction costs, correct short P&L calculation, stop loss distance warnings, and proper drawdown calculation. No misleading win rate metrics.

**Transparent limitations** — every design decision includes a documented limitation. This is not a black box.

### What this project does not do

**It is not a price prediction model** — the classifier identifies policy stance, not market direction with certainty. A hawkish signal does not guarantee bond prices will fall — it means the language leans toward tightening, which historically correlates with falling bond prices.

**The backtest sample is small** — 24 FOMC events is enough to show directional validity but not enough to draw statistically robust conclusions about long-run performance. A proper institutional backtest would use hundreds of events.

**No intraday execution** — returns are measured close-to-close. Real FOMC trading happens intraday on the day of release, where bid-ask spreads and market impact are significant.

**Keyword routing in Section 2 is approximate** — asset class routing uses keyword matching which cannot capture complex cross-asset relationships. A production system would use a trained multi-label classifier mapping news events to affected assets based on historical correlations.

**Upgrade path identified** — the ideal Layer 1 model is `gtfintechlab/FOMC-RoBERTa`, trained specifically on FOMC minutes labelled by economists. It requires HuggingFace account approval and is a direct drop-in replacement once access is granted.

---

*Built as a portfolio project for investment banking and sales & trading recruitment. Methodology inspired by Rosenbaum & Pearl and macro fixed income research practices.*
