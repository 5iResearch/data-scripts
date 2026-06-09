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
  - sp500-market-model.html  S&P 500 coloured by composite model score + score panel

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
import plotly.subplots as sp
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
SOURCE     = "Source: 5i Research · Yahoo Finance · Koyfin · FINRA"

# ── Corporate Color Palette ────────────────────────────────────────────────────
CORP = {
    'orange': '#C67A29',
    'blue':   '#1F79BE',
    'dgrey':  '#363636',
    'green':  '#44A660',
    'red':    '#A22A2A',
}

# ── Signal Colors (Market Model) ───────────────────────────────────────────────
SIGNAL_COLORS = {
    0: '#A8A8A8',       # N/A
    1: CORP['red'],     # Trim
    2: '#8E6AC8',       # Tactical Buy/Hold — purple
    3: '#D4A820',       # Buy — gold
    4: CORP['blue'],    # Strong Buy
    5: CORP['green'],   # Very Strong Buy
}
SIGNAL_LABELS = {
    0: 'N/A',
    1: 'Trim',
    2: 'Tactical Buy/Hold',
    3: 'Buy',
    4: 'Strong Buy',
    5: 'Very Strong Buy',
}

# ── Axis / Layout Helpers ──────────────────────────────────────────────────────
_AXIS_STYLE = dict(
    showgrid=True, gridcolor='#EBEBEB', gridwidth=1,
    zeroline=False, tickfont=dict(size=12), linecolor='#DDDDDD',
    showspikes=True, spikecolor='#BBBBBB', spikethickness=1, spikedash='dot',
)

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.write_html(path, include_plotlyjs="cdn", config={'responsive': True})
    print(f"  ✓  {name}")


def _get_yrange(series, start_date, end_date=None):
    """Compute padded y-axis range for a series within a date window."""
    mask = series.index >= start_date
    if end_date:
        mask &= series.index <= end_date
    vals = series[mask].dropna()
    if len(vals) == 0:
        return [None, None]
    pad = (vals.max() - vals.min()) * 0.06
    return [float(vals.min() - pad), float(vals.max() + pad)]


def _time_buttons(today, all_start, top_series, bot_series,
                  top_key="yaxis.range", bot_key="yaxis2.range"):
    """Build YTD / 1Y / 3Y / 5Y / All time-range buttons with axis rescaling."""
    ytd_start = pd.Timestamp(f"{today.year}-01-01")
    y1_start  = today - pd.DateOffset(years=1)
    y3_start  = today - pd.DateOffset(years=3)
    y5_start  = today - pd.DateOffset(years=5)

    def btn(label, x_args, yr_top, yr_bot):
        args = dict(x_args)
        args[top_key] = yr_top
        args[bot_key] = yr_bot
        return dict(args=[args], label=label, method="relayout")

    buttons = [
        btn("YTD", {"xaxis.range": [str(ytd_start), str(today)]},
            _get_yrange(top_series, ytd_start),
            _get_yrange(bot_series, ytd_start)),
        btn("1Y",  {"xaxis.range": [str(y1_start), str(today)]},
            _get_yrange(top_series, y1_start),
            _get_yrange(bot_series, y1_start)),
        btn("3Y",  {"xaxis.range": [str(y3_start), str(today)]},
            _get_yrange(top_series, y3_start),
            _get_yrange(bot_series, y3_start)),
        btn("5Y",  {"xaxis.range": [str(y5_start), str(today)]},
            _get_yrange(top_series, y5_start),
            _get_yrange(bot_series, y5_start)),
        dict(args=[{"xaxis.autorange": True,
                    top_key: _get_yrange(top_series, all_start),
                    bot_key: _get_yrange(bot_series, all_start)}],
             label="All", method="relayout"),
    ]
    return dict(
        buttons=buttons,
        active=4, direction='left', pad=dict(r=10, t=8),
        showactive=True, type='buttons',
        x=0.5, xanchor='center', y=-0.13, yanchor='top',
        bgcolor='white', bordercolor=CORP['blue'], borderwidth=1,
        font=dict(color=CORP['dgrey'], size=12, family='Arial'),
    )


def _bottom_annotations():
    return [
        dict(
            text="<i>Tip: Click and drag to zoom · Double-click to reset</i>",
            showarrow=False, xref="paper", yref="paper",
            x=0, y=-0.27, xanchor="left", yanchor="bottom",
            font=dict(size=11, color='#999999', family='Arial'),
        ),
        dict(
            text=SOURCE,
            showarrow=False, xref="paper", yref="paper",
            x=1, y=-0.27, xanchor="right", yanchor="bottom",
            font=dict(size=11, color='#999999', family='Arial'),
        ),
    ]


def _dual_scatter(x_top, y_top, x_bot, y_bot,
                  colorscale, color_top, color_bot,
                  title, y_top_label, y_bot_label,
                  hover_top_fmt=",.0f", hover_bot_fmt=".2f",
                  y_bot_range=None, ref_line=None):
    """
    Two-panel scatter chart in the HTML-file style, with generate_charts sizes.

    Panel heights: [0.65, 0.35] — taller top panel (from generate_charts.py)
    Dot sizes: top=4, bottom=3                  (from generate_charts.py)
    Styling: white bg, corporate colors, grey path, time buttons, hover templates
    """
    today     = x_top.max()
    all_start = x_top.min()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],   # keep taller top panel from generate_charts.py
        vertical_spacing=0.06,
    )

    # ── Top panel ─────────────────────────────────────────────────────────────
    # Grey path line (sits behind dots)
    fig.add_trace(go.Scatter(
        x=x_top, y=y_top, mode='lines',
        line=dict(color='#CCCCCC', width=1),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=1)
    # Coloured dots — size=4 (from generate_charts.py)
    fig.add_trace(go.Scatter(
        x=x_top, y=y_top, mode='markers', name='S&P 500',
        marker=dict(size=4, color=color_top, colorscale=colorscale,
                    opacity=0.88, showscale=False, line=dict(width=0)),
        showlegend=False,
        hovertemplate=f'<b>%{{x|%b %Y}}</b><br>S&P 500: %{{y:{hover_top_fmt}}}<extra></extra>',
    ), row=1, col=1)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    # Grey path line
    fig.add_trace(go.Scatter(
        x=x_bot, y=y_bot, mode='lines',
        line=dict(color='#CCCCCC', width=1),
        showlegend=False, hoverinfo='skip',
    ), row=2, col=1)
    # Coloured dots — size=3 (from generate_charts.py)
    fig.add_trace(go.Scatter(
        x=x_bot, y=y_bot, mode='markers', name=y_bot_label,
        marker=dict(size=3, color=color_bot, colorscale=colorscale,
                    opacity=0.88, showscale=False, line=dict(width=0)),
        showlegend=False,
        hovertemplate=f'<b>%{{x|%b %Y}}</b><br>{y_bot_label}: %{{y:{hover_bot_fmt}}}<extra></extra>',
    ), row=2, col=1)

    # Optional reference line on bottom panel
    if ref_line is not None:
        fig.add_hline(y=ref_line, line_dash='dot', line_color='#AAAAAA',
                      line_width=1, opacity=0.7, row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    top_series = pd.Series(y_top.values if hasattr(y_top, 'values') else y_top,
                           index=x_top)
    bot_series = pd.Series(y_bot.values if hasattr(y_bot, 'values') else y_bot,
                           index=x_bot if x_bot is not None else x_top)

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        height=600,                 # keep taller height from generate_charts.py
        margin=dict(l=65, r=45, t=90, b=155),
        showlegend=False,

        title=dict(
            text=f'<b>{title}</b>',
            x=0.5, xanchor='center',
            font=dict(size=20, color=CORP['dgrey'], family='Arial'),
            pad=dict(b=8),
        ),

        font=dict(family='Arial', size=13, color=CORP['dgrey']),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',

        updatemenus=[_time_buttons(today, all_start, top_series, bot_series)],
    )

    # Axis styles
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(title_text=y_top_label, tickformat=hover_top_fmt,
                     **_AXIS_STYLE, row=1, col=1)
    yax2 = dict(title_text=y_bot_label, **_AXIS_STYLE)
    if y_bot_range:
        yax2['range'] = y_bot_range
    fig.update_yaxes(**yax2, row=2, col=1)

    # In-panel labels (Bloomberg style)
    fig.add_annotation(text="<b>S&P 500</b>",
                       xref="x domain", yref="y domain",
                       x=0.01, y=0.97, xanchor="left", yanchor="top",
                       showarrow=False,
                       font=dict(size=13, color=CORP['dgrey'], family='Arial'))
    fig.add_annotation(text=f"<b>{y_bot_label}</b>",
                       xref="x2 domain", yref="y2 domain",
                       x=0.01, y=0.97, xanchor="left", yanchor="top",
                       showarrow=False,
                       font=dict(size=13, color=CORP['dgrey'], family='Arial'))

    for ann in _bottom_annotations():
        fig.add_annotation(**ann)

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
    from io import StringIO
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"})
    tables = pd.read_html(StringIO(resp.text))
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

# ── Chart 1: RSI ──────────────────────────────────────────────────────────────
RSI_COLORSCALE = [
    [0.0, CORP['green']],
    [0.5, '#D4A820'],
    [1.0, CORP['red']],
]
fig = _dual_scatter(
    x_top=sp500.index,   y_top=sp500["Close"],
    x_bot=sp500.index,   y_bot=sp500["RSI_1yr"],
    colorscale=RSI_COLORSCALE,
    color_top=sp500["RSI_1yr"],
    color_bot=sp500["RSI_1yr"],
    title="S&P 500 & 1-Year RSI",
    y_top_label="S&P 500 Price",
    y_bot_label="1-Year RSI",
    hover_top_fmt=",.0f", hover_bot_fmt=".1f",
    y_bot_range=[40, 65],
)
_save(fig, "sp500-rsi.html")


# ── Chart 2: VIX ─────────────────────────────────────────────────────────────
VIX_COLORSCALE = [
    [0.0, CORP['blue']],
    [0.5, CORP['orange']],
    [1.0, CORP['red']],
]
fig = _dual_scatter(
    x_top=sp500.index,   y_top=sp500["Close"],
    x_bot=sp500.index,   y_bot=sp500["vix"],
    colorscale=VIX_COLORSCALE,
    color_top=sp500["vix"],
    color_bot=sp500["vix"],
    title="S&P 500 & VIX",
    y_top_label="S&P 500 Price",
    y_bot_label="VIX",
    hover_top_fmt=",.0f", hover_bot_fmt=".1f",
)
_save(fig, "sp500-vix.html")


# ── Chart 3: PMI ─────────────────────────────────────────────────────────────
PMI_COLORSCALE = [
    [0.0, CORP['red']],
    [0.5, '#D4A820'],
    [1.0, CORP['blue']],
]
fig = _dual_scatter(
    x_top=merged.index,  y_top=merged["Close"],
    x_bot=merged.index,  y_bot=merged["rolling_PMI"],
    colorscale=PMI_COLORSCALE,
    color_top=merged["rolling_PMI"],
    color_bot=merged["rolling_PMI"],
    title="S&P 500 & ISM PMI (6-Mo. Avg.)",
    y_top_label="S&P 500 Price",
    y_bot_label="PMI (6-Mo. Avg.)",
    hover_top_fmt=",.0f", hover_bot_fmt=".1f",
    y_bot_range=[30, 70],
    ref_line=50,
)
_save(fig, "sp500-pmi.html")


# ── Chart 4: Margin Debt ──────────────────────────────────────────────────────
MARGIN_COLORSCALE = [
    [0.0, CORP['red']],
    [0.5, '#D4A820'],
    [1.0, CORP['green']],
]
fig = _dual_scatter(
    x_top=merged.index,  y_top=merged["Close"],
    x_bot=merged.index,  y_bot=merged["yoy"],
    colorscale=MARGIN_COLORSCALE,
    color_top=merged["yoy"],
    color_bot=merged["yoy"],
    title="S&P 500 & Margin Debt YoY %",
    y_top_label="S&P 500 Price",
    y_bot_label="Margin Debt YoY",
    hover_top_fmt=",.0f", hover_bot_fmt=".1%",
    y_bot_range=[-1, 1],
    ref_line=0,
)
_save(fig, "sp500-margin.html")


# ── Chart 5: Breadth ──────────────────────────────────────────────────────────
if breadth_ok:
    BREADTH_COLORSCALE = [
        [0.0, CORP['red']],
        [0.5, '#D4A820'],
        [1.0, CORP['green']],
    ]
    fig = _dual_scatter(
        x_top=merged.index,  y_top=merged["Close"],
        x_bot=merged.index,  y_bot=merged["net %"],
        colorscale=BREADTH_COLORSCALE,
        color_top=merged["net %"],
        color_bot=merged["net %"],
        title="S&P 500 & Market Breadth (% Above 1-Year Ago)",
        y_top_label="S&P 500 Price",
        y_bot_label="% Advancing",
        hover_top_fmt=",.0f", hover_bot_fmt=".1%",
        y_bot_range=[0, 1],
        ref_line=0.5,
    )
    _save(fig, "sp500-breadth.html")
else:
    print("  ⚠ Skipped sp500-breadth.html (breadth data unavailable)")


# ── Chart 6: Composite Market Model ───────────────────────────────────────────
# Two panels:
#   Top:    S&P 500 price dots coloured by signal (size=4, corporate signal palette)
#   Bottom: Rolling composite score line with signal-coloured dots (size=3)
#           Uses the same SIGNAL_COLORS as the top panel

merged["Weighted_Average"] = merged["Weighted_Average"].fillna(0).astype(int)
latest_signal = int(merged["Weighted_Average"].iloc[-1])
current_color = SIGNAL_COLORS[latest_signal]
current_label = SIGNAL_LABELS[latest_signal]

today     = merged.index.max()
all_start = merged.index.min()

ytd_start = pd.Timestamp(f"{today.year}-01-01")
y1_start  = today - pd.DateOffset(years=1)
y3_start  = today - pd.DateOffset(years=3)
y5_start  = today - pd.DateOffset(years=5)

close_series = merged["Close"]
score_series = merged["Rolling_12MA"].dropna()

def _mm_yranges(start):
    return (
        _get_yrange(close_series, start),
        _get_yrange(score_series, start),
    )

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],   # taller top panel (generate_charts.py)
    vertical_spacing=0.06,
)

# ── Top panel: S&P 500 coloured by signal ─────────────────────────────────────
# Grey path line
fig.add_trace(go.Scatter(
    x=merged.index, y=merged["Close"], mode='lines',
    line=dict(color='#CCCCCC', width=1),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)

# Signal-coloured dots, size=4 (generate_charts.py)
for i in range(6):
    mask   = merged["Weighted_Average"] == i
    subset = merged[mask]
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset["Close"],
        mode='markers',
        marker=dict(size=4, color=SIGNAL_COLORS[i], opacity=0.88, line=dict(width=0)),
        name=SIGNAL_LABELS[i],
        legendgroup=str(i),
        hovertemplate=(
            '<b>%{x|%b %Y}</b><br>'
            f'S&P 500: %{{y:,.0f}}<br>'
            f'Signal: {SIGNAL_LABELS[i]}'
            '<extra></extra>'
        ),
    ), row=1, col=1)

# ── Bottom panel: Rolling score line + signal-coloured dots ───────────────────
# Grey path line
fig.add_trace(go.Scatter(
    x=merged.index, y=merged["Rolling_12MA"], mode='lines',
    line=dict(color='#CCCCCC', width=1),
    showlegend=False, hoverinfo='skip',
), row=2, col=1)

# Signal-coloured dots, size=3 (generate_charts.py), same colours as top panel
for i in range(6):
    mask   = merged["Weighted_Average"] == i
    subset = merged[mask]
    fig.add_trace(go.Scatter(
        x=subset.index,
        y=subset["Rolling_12MA"],
        mode='markers',
        marker=dict(size=3, color=SIGNAL_COLORS[i], opacity=0.88, line=dict(width=0)),
        name=SIGNAL_LABELS[i],
        legendgroup=str(i),
        showlegend=False,
        hovertemplate=(
            '<b>%{x|%b %Y}</b><br>'
            f'Score: %{{y:.2f}}<br>'
            f'Signal: {SIGNAL_LABELS[i]}'
            '<extra></extra>'
        ),
    ), row=2, col=1)

# Score threshold reference lines
for lvl, col in [(2, SIGNAL_COLORS[2]), (3, SIGNAL_COLORS[3]), (4, SIGNAL_COLORS[4])]:
    fig.add_hline(y=lvl, line_dash='dot', line_color=col, opacity=0.4, row=2, col=1)

# ── Time-range buttons (both panels) ──────────────────────────────────────────
yr_ytd_top, yr_ytd_bot = _mm_yranges(ytd_start)
yr_1y_top,  yr_1y_bot  = _mm_yranges(y1_start)
yr_3y_top,  yr_3y_bot  = _mm_yranges(y3_start)
yr_5y_top,  yr_5y_bot  = _mm_yranges(y5_start)
yr_all_top, yr_all_bot = _mm_yranges(all_start)

mm_buttons = dict(
    buttons=[
        dict(args=[{"xaxis.range": [str(ytd_start), str(today)],
                    "yaxis.range": yr_ytd_top, "yaxis2.range": yr_ytd_bot}],
             label="YTD", method="relayout"),
        dict(args=[{"xaxis.range": [str(y1_start), str(today)],
                    "yaxis.range": yr_1y_top, "yaxis2.range": yr_1y_bot}],
             label="1Y", method="relayout"),
        dict(args=[{"xaxis.range": [str(y3_start), str(today)],
                    "yaxis.range": yr_3y_top, "yaxis2.range": yr_3y_bot}],
             label="3Y", method="relayout"),
        dict(args=[{"xaxis.range": [str(y5_start), str(today)],
                    "yaxis.range": yr_5y_top, "yaxis2.range": yr_5y_bot}],
             label="5Y", method="relayout"),
        dict(args=[{"xaxis.autorange": True,
                    "yaxis.range": yr_all_top, "yaxis2.range": yr_all_bot}],
             label="All", method="relayout"),
    ],
    active=4, direction='left', pad=dict(r=10, t=8),
    showactive=True, type='buttons',
    x=0.5, xanchor='center', y=-0.13, yanchor='top',
    bgcolor='white', bordercolor=CORP['blue'], borderwidth=1,
    font=dict(color=CORP['dgrey'], size=12, family='Arial'),
)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    template='plotly_white',
    autosize=True,
    height=600,             # taller height (generate_charts.py)
    margin=dict(l=65, r=45, t=90, b=155),

    title=dict(
        text='<b>S&P 500 — Macro Market Model</b>',
        x=0.5, xanchor='center',
        font=dict(size=20, color=CORP['dgrey'], family='Arial'),
        pad=dict(b=8),
    ),

    font=dict(family='Arial', size=13, color=CORP['dgrey']),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified',

    legend=dict(
        orientation='h',
        yanchor='bottom', y=1.01,
        xanchor='center', x=0.5,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#E0E0E0', borderwidth=1,
        itemsizing='constant',
        tracegroupgap=0,
    ),

    updatemenus=[mm_buttons],
)

fig.update_xaxes(**_AXIS_STYLE)
fig.update_yaxes(title_text='S&P 500 Price', tickformat=',.0f', **_AXIS_STYLE, row=1, col=1)
fig.update_yaxes(title_text='Model Score (21-day MA)', tickformat='.2f',
                 range=[1, 5], **_AXIS_STYLE, row=2, col=1)

# In-panel labels
fig.add_annotation(text="<b>S&P 500</b>",
                   xref="x domain", yref="y domain",
                   x=0.01, y=0.97, xanchor="left", yanchor="top",
                   showarrow=False,
                   font=dict(size=13, color=CORP['dgrey'], family='Arial'))
fig.add_annotation(text="<b>Model Score</b>",
                   xref="x2 domain", yref="y2 domain",
                   x=0.01, y=0.97, xanchor="left", yanchor="top",
                   showarrow=False,
                   font=dict(size=13, color=CORP['dgrey'], family='Arial'))

# Current signal badge
fig.add_annotation(
    text=f"<b>Current Signal: {current_label}</b>",
    showarrow=False, xref="paper", yref="paper",
    x=0.07, y=0.91, xanchor="left", yanchor="top",
    font=dict(size=13, color='white', family='Arial Black'),
    bgcolor=current_color, bordercolor=current_color,
    borderwidth=2, borderpad=6,
)

for ann in _bottom_annotations():
    fig.add_annotation(**ann)

_save(fig, "sp500-market-model.html")

print("\nAll charts saved to outputs/")
