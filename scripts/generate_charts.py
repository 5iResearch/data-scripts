"""
generate_charts.py
------------------
Generates 6 interactive Plotly HTML charts for the 5i Research dashboard.
Outputs go to the outputs/ folder.

Charts produced:
  - sp500-rsi.html        S&P 500 coloured by 1-Year RSI
  - sp500-vix.html        S&P 500 coloured by VIX
  - sp500-pmi.html        S&P 500 coloured by ISM PMI
  - sp500-margin.html     S&P 500 coloured by Margin Debt YoY
  - sp500-breadth.html    S&P 500 coloured by Market Breadth
  - sp500-market-model.html  S&P 500 coloured by composite model score

Data sources:
  - Yahoo Finance (S&P 500, VIX) — pulled automatically via yfinance
  - data/ISM.csv             — update monthly from Koyfin
  - data/margin_2.csv        — update monthly from finra.org

Usage:
  python scripts/generate_charts.py
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
DATA_DIR   = os.path.join(ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ISM_CSV    = os.path.join(DATA_DIR, "ISM.csv")
MARGIN_CSV = os.path.join(DATA_DIR, "margin_2.csv")

# ── Config ─────────────────────────────────────────────────────────────────────
START_DATE = "1997-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
SOURCE     = "Source: 5i Research, Yahoo Finance, Koyfin, Finra.org"

SCORE_COLORS = {
    1: "#d62728",   # red
    2: "#ff7f0e",   # orange
    3: "#aec7e8",   # light blue
    4: "#1f77b4",   # dark blue
    5: "#2ca02c",   # green
    0: "#888888",   # grey fallback
}

def _source_annotation():
    return dict(
        text=SOURCE, showarrow=False,
        xref="paper", yref="paper",
        x=1, y=-0.10, xanchor="right", yanchor="bottom",
        font=dict(size=11, color="gray"),
    )

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  ✓  {name}")

def _dual_scatter(x, y_top, y_bot, colors, title, y_top_label, y_bot_label,
                  colorbar_title="", y_bot_range=None):
    """Two-panel scatter: S&P 500 price top, indicator bottom, coloured by score."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=x, y=y_top, mode="markers",
        marker=dict(size=4, color=colors, colorscale="RdYlGn",
                    showscale=True, colorbar=dict(title=colorbar_title, len=0.5, y=0.75)),
        name="S&P 500"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=y_bot, mode="markers",
        marker=dict(size=3, color=colors, colorscale="RdYlGn"),
        name=y_bot_label), row=2, col=1)
    layout = dict(
        title=dict(text=title, font=dict(size=15)),
        yaxis=dict(title=y_top_label),
        yaxis2=dict(title=y_bot_label),
        xaxis2=dict(title="Date"),
        showlegend=False, height=600,
        margin=dict(b=80, t=70),
        template="seaborn",
    )
    if y_bot_range:
        layout["yaxis2"]["range"] = y_bot_range
    fig.update_layout(**layout)
    fig.add_annotation(_source_annotation())
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 1. Fetch base data
# ══════════════════════════════════════════════════════════════════════════════
print("Fetching S&P 500 and VIX from Yahoo Finance…")
sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
sp500.columns = sp500.columns.get_level_values(0)

vix = yf.download("^VIX", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
vix.columns = vix.columns.get_level_values(0)

sp500["vix"] = vix["Close"]
sp500["Forward_Returns"] = (sp500["Close"].shift(-252) / sp500["Close"]) - 1
sp500["priceyoy"]        = (sp500["Close"].shift(252)  / sp500["Close"]) - 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. RSI
# ══════════════════════════════════════════════════════════════════════════════
print("Computing RSI…")
sp500["RSI_1yr"]    = ta.momentum.RSIIndicator(sp500["Close"], window=252).rsi()
sp500["RSI_change"] = (sp500["RSI_1yr"] / sp500["RSI_1yr"].shift(252)) - 1

conditions_rsi = [
    (sp500["RSI_1yr"].between(43, 47)),
    (sp500["RSI_1yr"].between(47, 51) & (sp500["RSI_change"] < 0)),
    (sp500["RSI_1yr"].between(47, 51) & (sp500["RSI_change"] >= 0)),
    (sp500["RSI_1yr"].between(51, 55) & (sp500["RSI_change"] < 0)),
    (sp500["RSI_1yr"].between(51, 55) & (sp500["RSI_change"] >= 0)),
    (sp500["RSI_1yr"] > 55),
]
scores_rsi = [5, 4, 3, 2, 3, 1]
sp500["RSI_score"] = np.select(conditions_rsi, scores_rsi, default=3)
sp500["RSI_Range"] = pd.cut(sp500["RSI_1yr"], [43, 47, 50, 80])


# ══════════════════════════════════════════════════════════════════════════════
# 3. VIX scoring
# ══════════════════════════════════════════════════════════════════════════════
print("Scoring VIX…")
vix_ranges = [0, 18, 21, 27, 36, 120]
sp500["VIX_Range"] = pd.cut(sp500["vix"], vix_ranges)

conditions_vix = [
    sp500["vix"] < 18,
    sp500["vix"].between(18, 21),
    sp500["vix"].between(21, 27),
    sp500["vix"].between(27, 36),
    sp500["vix"] > 36,
]
scores_vix = [2, 3, 3, 4, 5]
sp500["VIX_score"] = np.select(conditions_vix, scores_vix, default=3)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PMI — from CSV
# ══════════════════════════════════════════════════════════════════════════════
print("Loading PMI CSV…")
pmi = pd.read_csv(ISM_CSV)
pmi["Date"] = pd.to_datetime(pmi[" Date"].str.strip(), format="%m-%d-%Y")
pmi.set_index("Date", inplace=True)
pmi.index = pmi.index + pd.offsets.MonthEnd(0)

sp500["Month_End"] = sp500.index.to_period("M").to_timestamp("M")
merged = pd.merge(sp500.reset_index(), pmi[["NAPMPMI Close"]], how="left",
                  left_on="Month_End", right_index=True)
merged["Date"] = merged["Date"] if "Date" in merged.columns else merged.index
merged = merged.set_index("Price") if "Price" in merged.columns else merged
# keep the original DatetimeIndex
merged.index = sp500.index[:len(merged)]
merged["pmi"] = merged["NAPMPMI Close"].ffill()
merged["rolling_PMI"] = merged["pmi"].rolling(window=126).mean()
merged.drop(columns=["Month_End", "NAPMPMI Close"], inplace=True)

conditions_pmi = [
    merged["rolling_PMI"] < 48,
    merged["rolling_PMI"].between(48, 50),
    merged["rolling_PMI"].between(50, 53),
    merged["rolling_PMI"].between(53, 56),
    merged["rolling_PMI"] > 56,
]
scores_pmi = [5, 4, 3, 2, 1]
merged["PMI_score"] = np.select(conditions_pmi, scores_pmi, default=3)
merged["PMI_Range"] = pd.cut(merged["pmi"], [0, 48, 50, 53, 56, 100])


# ══════════════════════════════════════════════════════════════════════════════
# 5. Margin Debt — from CSV
# ══════════════════════════════════════════════════════════════════════════════
print("Loading Margin CSV…")
margin = pd.read_csv(MARGIN_CSV)
margin["Year-Month"] = pd.to_datetime(margin["Year-Month"])
margin = margin.sort_values("Year-Month").set_index("Year-Month")
margin.index = margin.index + pd.offsets.MonthEnd(0)
margin.index.name = "Date"
margin.rename(columns={"Debit Balances in Customers' Securities Margin Accounts": "debit balances"},
              inplace=True)
margin["debit balances"] = margin["debit balances"].astype(str).str.replace(",", "").astype(float)
margin["yoy"] = margin["debit balances"].pct_change(periods=12)

merged["Month_End2"] = merged.index.to_period("M").to_timestamp("M")
merged = pd.merge(merged, margin[["yoy"]], how="left",
                  left_on="Month_End2", right_index=True)
merged["yoy"] = merged["yoy"].ffill()
merged["rolling_yoy"] = merged["yoy"].rolling(window=252).mean()
merged.drop(columns=["Month_End2"], inplace=True)

conditions_margin = [
    merged["rolling_yoy"] < -0.15,
    merged["rolling_yoy"].between(-0.15, 0),
    merged["rolling_yoy"].between(0, 0.15),
    merged["rolling_yoy"].between(0.15, 0.30),
    merged["rolling_yoy"] > 0.30,
]
scores_margin = [5, 4, 3, 2, 1]
merged["Margin_score"] = np.select(conditions_margin, scores_margin, default=3)
merged["Margin_Range"] = pd.cut(merged["rolling_yoy"], [-2, -0.15, 0, 0.15, 0.30, 2])


# ══════════════════════════════════════════════════════════════════════════════
# 6. Breadth — download all S&P 500 constituents
# ══════════════════════════════════════════════════════════════════════════════
print("Fetching S&P 500 breadth (this takes ~1–2 min on first run)…")
try:
    import requests
    tables = pd.read_html(
        requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0"}).text)
    symbols = [t for t in tables[0]["Symbol"].tolist()
               if t not in ["SEDG", "OTIS", "NTAP"]]
    symbols = [s.replace(".", "-") for s in symbols]

    stock_data = yf.download(symbols, start=START_DATE, end=END_DATE,
                              auto_adjust=True, progress=False)["Close"]
    stock_data["advance"] = (stock_data > stock_data.shift(252)).sum(axis=1)
    stock_data["decline"] = (stock_data < stock_data.shift(252)).sum(axis=1)
    stock_data["net %"]   = stock_data["advance"] / (stock_data["advance"] + stock_data["decline"])

    merged = pd.merge(merged, stock_data[["net %"]], how="left",
                      left_index=True, right_index=True)
    merged["net %"] = merged["net %"].ffill()

    conditions_breadth = [
        merged["net %"] < 0.40,
        merged["net %"].between(0.40, 0.45),
        merged["net %"].between(0.45, 0.55),
        merged["net %"].between(0.55, 0.60),
        merged["net %"] > 0.60,
    ]
    scores_breadth = [5, 4, 3, 2, 1]
    merged["Breadth_score"] = np.select(conditions_breadth, scores_breadth, default=3)
    merged["Breadth_Range"] = pd.cut(merged["net %"], [0, 0.40, 0.45, 0.55, 0.60, 1])
    breadth_ok = True
except Exception as e:
    print(f"  ⚠ Breadth download failed: {e}. Skipping breadth chart.")
    merged["Breadth_score"] = 3
    merged["net %"] = np.nan
    breadth_ok = False


# ══════════════════════════════════════════════════════════════════════════════
# 7. Composite Market Model score
# ══════════════════════════════════════════════════════════════════════════════
print("Computing composite model score…")
weights = dict(PMI_score=0.20, VIX_score=0.20, RSI_score=0.20,
               Margin_score=0.20, Breadth_score=0.20)

merged["Weighted_Average"] = (
    merged["PMI_score"]     * weights["PMI_score"]     +
    merged["VIX_score"]     * weights["VIX_score"]     +
    merged["RSI_score"]     * weights["RSI_score"]     +
    merged["Margin_score"]  * weights["Margin_score"]  +
    merged["Breadth_score"] * weights["Breadth_score"]
)
merged["Rolling_12MA"] = merged["Weighted_Average"].rolling(window=21).mean()
merged["Model_score"]  = merged["Rolling_12MA"].round().clip(1, 5).fillna(3).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Generate charts
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating charts…")

# Helper: map integer score → colour string
def score_to_color(series):
    return series.fillna(0).astype(int).map(SCORE_COLORS).fillna("#888888")


# ── Chart 1: RSI ──────────────────────────────────────────────────────────────
fig = _dual_scatter(
    x=sp500.index,
    y_top=sp500["Close"],
    y_bot=sp500["RSI_1yr"],
    colors=sp500["RSI_1yr"],
    title="S&P 500 — 1-Year RSI",
    y_top_label="S&P 500",
    y_bot_label="1-Year RSI",
    colorbar_title="RSI",
    y_bot_range=[40, 65],
)
_save(fig, "sp500-rsi.html")


# ── Chart 2: VIX ─────────────────────────────────────────────────────────────
fig = _dual_scatter(
    x=sp500.index,
    y_top=sp500["Close"],
    y_bot=sp500["vix"],
    colors=sp500["vix"],
    title="S&P 500 — VIX",
    y_top_label="S&P 500",
    y_bot_label="VIX",
    colorbar_title="VIX",
)
_save(fig, "sp500-vix.html")


# ── Chart 3: PMI ─────────────────────────────────────────────────────────────
fig = _dual_scatter(
    x=merged.index,
    y_top=merged["Close"],
    y_bot=merged["pmi"],
    colors=merged["pmi"],
    title="S&P 500 — ISM PMI",
    y_top_label="S&P 500",
    y_bot_label="ISM PMI",
    colorbar_title="PMI",
    y_bot_range=[30, 70],
)
_save(fig, "sp500-pmi.html")


# ── Chart 4: Margin Debt ──────────────────────────────────────────────────────
fig = _dual_scatter(
    x=merged.index,
    y_top=merged["Close"],
    y_bot=merged["yoy"],
    colors=merged["yoy"],
    title="S&P 500 — Margin Debt YoY %",
    y_top_label="S&P 500",
    y_bot_label="Margin Debt YoY",
    colorbar_title="Margin YoY",
    y_bot_range=[-1, 1],
)
_save(fig, "sp500-margin.html")


# ── Chart 5: Breadth ──────────────────────────────────────────────────────────
if breadth_ok:
    fig = _dual_scatter(
        x=merged.index,
        y_top=merged["Close"],
        y_bot=merged["net %"],
        colors=merged["net %"],
        title="S&P 500 — Market Breadth (% of S&P 500 Above 1-Year Ago)",
        y_top_label="S&P 500",
        y_bot_label="% Advancing",
        colorbar_title="Breadth",
        y_bot_range=[0, 1],
    )
    _save(fig, "sp500-breadth.html")
else:
    print("  ⚠ Skipped sp500-breadth.html (breadth data unavailable)")


# ── Chart 6: Composite Market Model ───────────────────────────────────────────
color_map = score_to_color(merged["Model_score"])

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.06)
fig.add_trace(go.Scatter(
    x=merged.index, y=merged["Close"], mode="markers",
    marker=dict(size=4, color=color_map),
    name="S&P 500"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=merged.index, y=merged["Rolling_12MA"], mode="lines",
    line=dict(color="#1f77b4", width=1.5),
    name="Composite Score (21-day MA)"), row=2, col=1)

# Score threshold lines
for lvl, col in [(2, "#ff7f0e"), (3, "#aec7e8"), (4, "#1f77b4")]:
    fig.add_hline(y=lvl, line_dash="dot", line_color=col, opacity=0.5, row=2, col=1)

fig.update_layout(
    title=dict(text="S&P 500 — Composite Market Model Score (1=Bearish → 5=Bullish)",
               font=dict(size=15)),
    yaxis=dict(title="S&P 500"),
    yaxis2=dict(title="Model Score", range=[1, 5]),
    xaxis2=dict(title="Date"),
    height=600, showlegend=True, template="seaborn",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    margin=dict(b=80, t=70),
)
fig.add_annotation(_source_annotation())
_save(fig, "sp500-market-model.html")

print("\nAll charts saved to outputs/")
