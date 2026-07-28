"""
Daily "Inverse ETF Sentiment" report, adapted from the "Daily Notebook".

Reproduces a specific subset of that notebook's charts:
  1. QQQ vs 200-Week Moving Average
  2. S&P 500 Cumulative 2-Year Returns From Interest Rate Cuts
  3. QQQ Price vs Levered Inverse ETF Notional Volume Ratio (SQQQ / TQQQ)
  4. QQQ Price, SQQQ/TQQQ Ratio & Weekly RSI
  5. IWM Price vs Levered Inverse ETF Notional Volume Ratio (TZA / TNA)
  6. GLD Price vs Levered Inverse ETF Notional Volume Ratio (DUST / NUGT)
  7. SLV Price vs Levered Inverse ETF Notional Volume Ratio (ZSL / AGQ)
  8. Froth Index (5-Year, Z-Score Oscillator): Growth/Defensive, Growth, Defensive

Note: the source notebook's growth_tickers list has a typo (missing commas
between 'CRWD' 'RKLB' 'RBRK' 'ACHR' 'MARA' 'MSTR') that silently concatenates
those into one invalid ticker and drops them from the Froth Index. Fixed here
so all growth tickers are actually included.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "inverse-etf-sentiment")

TODAY = datetime.today()


# ── 1. QQQ vs 200-Week MA ─────────────────────────────────────────────────────
def chart_qqq_200w_ma():
    qqq = yf.download("QQQ", start="1999-01-01", progress=False)
    qqq.columns = qqq.columns.get_level_values(0)
    qqq_weekly = qqq["Close"].resample("W-FRI").last().to_frame()
    qqq_weekly["200W_MA"] = qqq_weekly["Close"].rolling(200).mean()
    qqq_weekly["Pct_Above_MA"] = (qqq_weekly["Close"] - qqq_weekly["200W_MA"]) / qqq_weekly["200W_MA"] * 100
    qqq_weekly["Above"] = qqq_weekly["Pct_Above_MA"].clip(lower=0)
    qqq_weekly["Below"] = qqq_weekly["Pct_Above_MA"].clip(upper=0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05,
                         subplot_titles=("QQQ vs 200-Week MA", "% Above/Below 200W MA"))
    fig.add_trace(go.Scatter(x=qqq_weekly.index, y=qqq_weekly["Close"], mode="lines", name="QQQ", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=qqq_weekly.index, y=qqq_weekly["200W_MA"], mode="lines", name="200W MA", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=qqq_weekly.index, y=qqq_weekly["Above"], fill="tozeroy", mode="none",
                              name="% Above 200W MA", fillcolor="rgba(0,200,0,0.3)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=qqq_weekly.index, y=qqq_weekly["Below"], fill="tozeroy", mode="none",
                              name="% Below 200W MA", fillcolor="rgba(255,0,0,0.3)"), row=2, col=1)
    fig.update_yaxes(type="log", title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="% Above/Below MA", row=2, col=1)
    fig.update_layout(template="plotly_white", legend=dict(x=0.01, y=0.99), height=700)
    return fig


# ── 2. S&P 500 2-Year Returns From Rate Cuts ──────────────────────────────────
def chart_rate_cut_returns():
    windows = [
        ("1984", "1984-10-01", "1986-10-01"),
        ("1989", "1989-06-05", "1991-06-05"),
        ("1995", "1995-07-06", "1997-07-06"),
        ("2007", "2007-09-18", "2009-09-18"),
        ("2019", "2019-08-01", "2021-08-01"),
        ("2000", "2000-12-01", "2002-12-01"),
        ("2024", "2024-09-04", TODAY.strftime("%Y-%m-%d")),
    ]
    fig = go.Figure()
    for label, start, end in windows:
        data = yf.download("^GSPC", start=start, end=end, progress=False)
        data.columns = data.columns.get_level_values(0)
        close = data["Close"].reset_index(drop=True)
        cumulative = (1 + close.pct_change()).cumprod() - 1
        line_color = "black" if label == "2024" else None
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values, mode="lines", name=label,
                                  line=dict(color=line_color) if line_color else None))

    fig.update_layout(
        xaxis_title="# of Days", yaxis_title="Cumulative Return", height=700, width=1050,
        title="S&P 500 Cumulative 2-Year Returns From Interest Rate Cuts", showlegend=True,
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.add_annotation(text="Source: 5i Research, Yahoo Finance", xref="paper", yref="paper",
                        x=1.1, y=-0.17, showarrow=False, font=dict(size=14, color="#4B8EA9"), align="right", valign="bottom")
    return fig


# ── Shared: notional-volume ratio chart (levered inverse vs long) ───────────
def _notional_ratio_data(inv_ticker, lev_ticker, price_ticker, period="15y"):
    data = yf.download([inv_ticker, lev_ticker, price_ticker], period=period, interval="1d", progress=False)
    adj_close = data["Close"]
    volume = data["Volume"]
    notional = adj_close[[inv_ticker, lev_ticker]] * volume[[inv_ticker, lev_ticker]]
    notional.columns = [f"{inv_ticker}_Notional", f"{lev_ticker}_Notional"]
    ratio = (notional[f"{inv_ticker}_Notional"] / notional[f"{lev_ticker}_Notional"]) * 100
    ratio = ratio.dropna()
    return adj_close, ratio


def chart_levered_inverse_ratio(inv_ticker, lev_ticker, price_ticker, price_label, high_thresh, low_thresh, title, period="15y"):
    adj_close, ratio = _notional_ratio_data(inv_ticker, lev_ticker, price_ticker, period)
    high_stress_dates = ratio[ratio > high_thresh].index
    high_stress_prices = adj_close.loc[high_stress_dates, price_ticker]
    low_stress_dates = ratio[ratio < low_thresh].index
    low_stress_prices = adj_close.loc[low_stress_dates, price_ticker]

    def ratio_to_color(r):
        if r > high_thresh: return "blue"
        elif r < low_thresh: return "red"
        else: return "gray"

    ratio_colors = [ratio_to_color(r) for r in ratio]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"{price_label} Price (Highlight >{high_thresh}% / <{low_thresh}% Ratio)",
                        f"{inv_ticker} Notional Volume as % of {lev_ticker}"))
    fig.add_trace(go.Scatter(x=adj_close.index, y=adj_close[price_ticker], mode="lines",
                              name=f"{price_label} Price", line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=high_stress_dates, y=high_stress_prices, mode="markers",
        marker=dict(size=6, symbol="circle-open", color="blue"), name=f"Ratio > {high_thresh}%"), row=1, col=1)
    fig.add_trace(go.Scatter(x=low_stress_dates, y=low_stress_prices, mode="markers",
        marker=dict(size=6, symbol="circle-open", color="red"), name=f"Ratio < {low_thresh}%"), row=1, col=1)

    for i in range(1, len(ratio)):
        fig.add_trace(go.Scatter(x=ratio.index[i - 1:i + 1], y=ratio.iloc[i - 1:i + 1], mode="lines",
                                  line=dict(color=ratio_colors[i], width=2), showlegend=False), row=2, col=1)

    fig.update_layout(height=650, title=title, showlegend=True)
    fig.update_yaxes(title_text=f"{price_label} Price ($)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Inverse % of Long", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.add_annotation(x=1, y=-0.1, xref="paper", yref="paper", text="Source: 5i Research, Yahoo Finance",
                        showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=10, color="gray"))
    return fig


# ── 4. QQQ Price, SQQQ/TQQQ Ratio & Weekly RSI ───────────────────────────────
def chart_qqq_ratio_weekly_rsi():
    data = yf.download(["SQQQ", "TQQQ", "QQQ"], period="15y", interval="1d", progress=False)
    adj_close = data["Close"]
    volume = data["Volume"]
    notional = adj_close[["SQQQ", "TQQQ"]] * volume[["SQQQ", "TQQQ"]]
    notional.columns = ["SQQQ_Notional", "TQQQ_Notional"]
    ratio = (notional["SQQQ_Notional"] / notional["TQQQ_Notional"]) * 100
    ratio = ratio.dropna()
    high_stress_dates = ratio[ratio > 90].index
    low_stress_dates = ratio[ratio < 14].index

    qqq_weekly = adj_close["QQQ"].resample("W-FRI").last()
    delta = qqq_weekly.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))

    def nearest_week_index(date, weekly_index):
        pos = weekly_index.searchsorted(date)
        if pos == 0: return 0
        elif pos >= len(weekly_index): return len(weekly_index) - 1
        before, after = weekly_index[pos - 1], weekly_index[pos]
        return pos - 1 if abs((date - before).days) <= abs((after - date).days) else pos

    high_rsi_idx = [nearest_week_index(d, qqq_weekly.index) for d in high_stress_dates]
    low_rsi_idx = [nearest_week_index(d, qqq_weekly.index) for d in low_stress_dates]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("QQQ Price", "SQQQ/TQQQ Ratio", "QQQ Weekly RSI"))
    fig.add_trace(go.Scatter(x=adj_close.index, y=adj_close["QQQ"], mode="lines", name="QQQ Price", line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, mode="lines", name="SQQQ/TQQQ %", line=dict(color="black")), row=2, col=1)
    fig.add_trace(go.Scatter(x=qqq_weekly.index, y=rsi, mode="lines", name="RSI", line=dict(color="purple")), row=3, col=1)
    fig.add_trace(go.Scatter(x=qqq_weekly.index[high_rsi_idx], y=rsi.iloc[high_rsi_idx], mode="markers",
        marker=dict(color="blue", size=6, symbol="circle-open"), name="High Stress"), row=3, col=1)
    fig.add_trace(go.Scatter(x=qqq_weekly.index[low_rsi_idx], y=rsi.iloc[low_rsi_idx], mode="markers",
        marker=dict(color="red", size=6, symbol="circle-open"), name="Low Stress"), row=3, col=1)

    fig.update_layout(height=900, title="QQQ Price, SQQQ/TQQQ Ratio & Weekly RSI")
    fig.update_yaxes(title_text="QQQ Price ($)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Ratio (%)", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


# ── 8. Froth Index (stock-level growth vs defensive) ─────────────────────────
GROWTH_TICKERS = [
    "TSLA", "ROKU", "COIN", "PLTR", "HOOD", "DKNG", "ARKK", "SHOP", "NVDA",
    "SPOT", "LMND", "RBLX", "CRWD", "RKLB", "RBRK", "ACHR", "MARA", "MSTR",
]
DEFENSIVE_TICKERS = ["KO", "PEP", "PG", "JNJ", "CL", "MO", "KMB", "WMT", "GIS", "XLP", "NEE", "DUK", "SO", "AWK", "O"]
FROTH_START = "2005-01-01"


def _normalized_index(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = data.dropna(axis=1, how="all")
    data = data.apply(lambda x: x / x[x.first_valid_index()], axis=0)
    data = data.ffill()
    return data.mean(axis=1)


def compute_froth_series():
    end_date = TODAY.strftime("%Y-%m-%d")
    growth_index = _normalized_index(GROWTH_TICKERS, FROTH_START, end_date)
    defensive_index = _normalized_index(DEFENSIVE_TICKERS, FROTH_START, end_date)

    window = 252
    growth_roll = growth_index / growth_index.rolling(window).apply(lambda x: x[0], raw=True)
    defensive_roll = defensive_index / defensive_index.rolling(window).apply(lambda x: x[0], raw=True)
    rs_ratio = np.log(growth_roll / defensive_roll)
    return growth_index, defensive_index, rs_ratio


def chart_froth_index_overall(rs_ratio):
    rolling_mean = rs_ratio.rolling(window=252).mean()
    rolling_std = rs_ratio.rolling(window=252).std()
    zscore = (rs_ratio - rolling_mean) / rolling_std

    def zscore_to_color(z):
        if z > 1.5: return "red"
        elif z > 0.5: return "orange"
        elif z > -0.5: return "gray"
        elif z > -1.5: return "lightblue"
        else: return "blue"

    colors = [zscore_to_color(z) if pd.notna(z) else "rgba(0,0,0,0)" for z in zscore]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.05,
                         subplot_titles=("Relative Strength (Growth / Defensive)", "Froth Index"))
    fig.add_trace(go.Scatter(x=rs_ratio.index, y=rs_ratio, mode="lines", name="RS Ratio", line=dict(color="black", width=2)), row=1, col=1)
    for i in range(1, len(zscore)):
        fig.add_trace(go.Scatter(x=zscore.index[i - 1:i + 1], y=zscore.iloc[i - 1:i + 1], mode="lines",
                                  line=dict(color=colors[i], width=3), showlegend=False, hoverinfo="x+y"), row=2, col=1)
    for y_val, label, color in [(2, "+2sigma", "green"), (-2, "-2sigma", "red"), (0, "Neutral", "gray")]:
        fig.add_trace(go.Scatter(x=zscore.index, y=[y_val] * len(zscore), mode="lines", name=label,
                                  line=dict(color=color, dash="dash" if y_val != 0 else "dot")), row=2, col=1)

    fig.update_layout(height=700, template="plotly_white", title_text="Froth Index: Z-Score Oscillator (Growth vs Defensive)",
                       showlegend=True, xaxis2_title="Date", yaxis1_title="Relative Strength", yaxis2_title="Froth Index")
    return fig


def _zscore_5y_chart(series, rs_ratio_for_cutoff, row1_title, chart_title, y1_title):
    five_years_ago = rs_ratio_for_cutoff.index.max() - pd.DateOffset(years=5)
    s5y = series[series.index >= five_years_ago]
    rolling_mean = s5y.rolling(window=20).mean()
    rolling_std = s5y.rolling(window=20).std()
    zscore = (s5y - rolling_mean) / rolling_std

    def zscore_to_color(z):
        if z > 2.25 or z < -2.25: return "blue"
        elif -2 <= z <= -1: return "red"
        else: return "gray"

    colors = [zscore_to_color(z) if pd.notna(z) else "rgba(0,0,0,0)" for z in zscore]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.4, 0.6], vertical_spacing=0.05,
                         subplot_titles=(row1_title, chart_title))
    for i in range(1, len(s5y)):
        fig.add_trace(go.Scatter(x=s5y.index[i - 1:i + 1], y=s5y.iloc[i - 1:i + 1], mode="lines",
                                  line=dict(color=colors[i], width=2), showlegend=False, hoverinfo="x+y"), row=1, col=1)
    for i in range(1, len(zscore)):
        fig.add_trace(go.Scatter(x=zscore.index[i - 1:i + 1], y=zscore.iloc[i - 1:i + 1], mode="lines",
                                  line=dict(color=colors[i], width=3), showlegend=False, hoverinfo="x+y"), row=2, col=1)
    for y_val, label, color, dash in [(2, "+2sigma", "green", "dash"), (-2, "-2sigma", "red", "dash"), (0, "Neutral", "gray", "dot")]:
        fig.add_trace(go.Scatter(x=zscore.index, y=[y_val] * len(zscore), mode="lines", name=label,
                                  line=dict(color=color, dash=dash)), row=2, col=1)

    fig.update_layout(height=700, width=1050, template="plotly_white", title_text=chart_title, showlegend=True,
                       xaxis2_title="Date", yaxis1_title=y1_title, yaxis2_title="Z-Score")
    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def build_report() -> str:
    parts = []

    print("QQQ vs 200-Week MA...")
    parts.append(section_header("QQQ vs 200-Week Moving Average"))
    parts.append(fig_to_div(chart_qqq_200w_ma()))

    print("S&P 500 returns from rate cuts...")
    parts.append(section_header("S&P 500 Cumulative 2-Year Returns From Interest Rate Cuts"))
    parts.append(fig_to_div(chart_rate_cut_returns()))

    print("QQQ vs SQQQ/TQQQ...")
    parts.append(section_header("QQQ Price vs Levered Inverse ETF"))
    parts.append(fig_to_div(chart_levered_inverse_ratio(
        "SQQQ", "TQQQ", "QQQ", "QQQ", 90, 14, "QQQ Price vs Levered Inverse ETF Notional Volume Ratio (SQQQ / TQQQ)")))

    print("QQQ + weekly RSI...")
    parts.append(section_header("QQQ Price, SQQQ/TQQQ Ratio & Weekly RSI"))
    parts.append(fig_to_div(chart_qqq_ratio_weekly_rsi()))

    print("IWM vs TZA/TNA...")
    parts.append(section_header("IWM Price vs Levered Inverse ETF"))
    parts.append(fig_to_div(chart_levered_inverse_ratio(
        "TZA", "TNA", "IWM", "IWM", 130, 20, "IWM Price vs Levered Inverse ETF Notional Volume Ratio (TZA / TNA)")))

    print("GLD vs DUST/NUGT...")
    parts.append(section_header("Gold (GLD) vs Levered Inverse ETF", "DUST (miners, -2x) / NUGT (miners, +2x)"))
    parts.append(fig_to_div(chart_levered_inverse_ratio(
        "DUST", "NUGT", "GLD", "GLD", 150, 20, "GLD Price vs Levered Inverse ETF Notional Volume Ratio (DUST / NUGT)")))

    print("SLV vs ZSL/AGQ...")
    parts.append(section_header("SLV Price vs Levered Inverse ETF"))
    parts.append(fig_to_div(chart_levered_inverse_ratio(
        "ZSL", "AGQ", "SLV", "SLV", 150, 20, "SLV Price vs Levered Inverse ETF Notional Volume Ratio (ZSL / AGQ)")))

    print("Froth Index (growth vs defensive)...")
    growth_index, defensive_index, rs_ratio = compute_froth_series()
    parts.append(section_header("Froth Index", "Growth vs Defensive stock baskets"))
    parts.append(fig_to_div(chart_froth_index_overall(rs_ratio)))
    parts.append(fig_to_div(_zscore_5y_chart(
        rs_ratio, rs_ratio, "RS Ratio (Growth / Defensive)",
        "Froth Index (Last 5 Years): Growth/Defensive RS Ratio &amp; Z-Score Oscillator", "RS Ratio")))
    parts.append(fig_to_div(_zscore_5y_chart(
        growth_index, rs_ratio, "Growth Index",
        "Froth Index (Last 5 Years): Growth Index &amp; Z-Score Oscillator", "Growth Index")))
    parts.append(fig_to_div(_zscore_5y_chart(
        defensive_index, rs_ratio, "Defensive Index",
        "Froth Index (Last 5 Years): Defensive Index &amp; Z-Score Oscillator", "Defensive Index")))

    return "\n".join(parts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Inverse ETF Sentiment</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #F5F5F7; color: #1C1C1E; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #D0D0D5; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #6E6E73; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #1C1C1E; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #6E6E73; font-size: 13px; margin-top: 6px; }}
</style>
</head>
<body>
<header>
  <h1>Inverse ETF Sentiment</h1>
  <div class="meta">Generated {date_str} &middot; QQQ trend, rate-cut seasonality, levered inverse-ETF stress ratios, and Froth Index</div>
</header>
{body}
</body>
</html>
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()
    body = build_report()
    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body=body)

    out_path = os.path.join(OUTPUT_DIR, "Inverse_ETF_Sentiment.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
