"""
Daily Industry & Broad-Market RSI report, adapted from the "Industry RSIs"
notebook's chart cells (weekly/monthly RSI for broad-market and sector ETFs).

Excludes the crypto section from the source notebook — broad markets and
sector ETFs only. Renders every chart with the notebook's branded Plotly
style (dark theme, RSI overbought/oversold zones, signal dots, logo) and
bundles them into a single self-contained HTML report so hover tooltips
stay interactive, rather than flattening to a static PDF:
  Broad Market ETFs (Weekly RSI-14 + Weekly RSI-52)
  -> Industry ETFs (Weekly RSI-14)
  -> Industry ETFs (Monthly RSI-14)
"""

import base64
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "industry-rsi")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")

# ── Corporate colours (matches Industry RSIs notebook) ───────────────────────
ORANGE = "#C67A29"
BLUE = "#1F79BE"
DGREY = "#363636"
LGREY = "#4A4A4A"
GREEN = "#44A660"
RED = "#A22A2A"
TEXTCLR = "#E8E8E8"

SIGNAL_COLORS = {"buy": GREEN, "hold": "#C0C0C0", "sell": RED}

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

MARKET_SYMBOLS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "QQQE": "Nasdaq 100 Equal Weight",
    "RSP": "S&P 500 Equal Weight",
}

SECTOR_SYMBOLS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

INDUSTRY_SYMBOLS = {
    "XME": "Metals & Mining",
    "VNQ": "Real Estate",
    "GDX": "Gold Miners",
    "AMLP": "MLPs",
    "ITB": "Homebuilders",
    "OIH": "Oil Services",
    "KRE": "Regional Banking",
    "XRT": "Retail",
    "MOO": "Agriculture",
    "FDN": "Internet",
    "IBB": "Biotechnology",
    "SMH": "Semiconductors",
    "XOP": "Oil & Gas Exploration",
    "PBW": "Clean Energy",
    "KIE": "Insurance",
    "PHO": "Water Resources",
    "IGV": "Software",
    "TAN": "Solar Energy",
    "JETS": "Airlines",
    "HACK": "Cybersecurity",
}

_cache: dict = {}


def fetch_all(tickers: list) -> dict:
    missing = [t for t in tickers if t not in _cache]
    if missing:
        raw = yf.download(missing, period="max", auto_adjust=True, progress=False, threads=True)
        if len(missing) == 1:
            raw.columns = pd.MultiIndex.from_product([raw.columns, missing])
        for t in missing:
            try:
                df = raw.xs(t, axis=1, level=1).dropna(how="all")
                _cache[t] = df
            except Exception:
                _cache[t] = pd.DataFrame()
    return {t: _cache[t] for t in tickers}


def _add_signals(df: pd.DataFrame, thresholds=(40, 75)) -> pd.DataFrame:
    oversold, overbought = thresholds
    cond = [df["RSI"] < oversold, df["RSI"] > overbought]
    vals = ["buy", "sell"]
    df["Signal"] = np.select(cond, vals, default="hold")
    return df


def compute_rsi(daily_df: pd.DataFrame, freq: str = "W", rsi_length: int = 14, thresholds=(40, 75)) -> pd.DataFrame:
    df = daily_df[["Close"]].resample(freq).last().dropna()
    df["RSI"] = RSIIndicator(df["Close"], rsi_length).rsi()
    return _add_signals(df.dropna(), thresholds=thresholds)


def get_weekly_rsi(ticker: str, rsi_length: int = 14, thresholds=(40, 75)) -> pd.DataFrame:
    data = fetch_all([ticker])[ticker]
    return compute_rsi(data, "W", rsi_length, thresholds=thresholds)


def get_monthly_rsi(ticker: str, rsi_length: int = 14) -> pd.DataFrame:
    data = fetch_all([ticker])[ticker]
    return compute_rsi(data, "ME", rsi_length)


def plot_price_rsi(df: pd.DataFrame, ticker: str, industry: str, log_price: bool = False, rsi_range=None, thresholds=(40, 75)) -> go.Figure:
    dot_colors = [SIGNAL_COLORS[s] for s in df["Signal"]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.06,
        subplot_titles=("", ""),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], mode="lines+markers",
            line=dict(color=BLUE, width=1.8),
            marker=dict(color=dot_colors, size=5, line=dict(width=0)),
            name="Price",
            hovertemplate="%{x|%b %d %Y}<br>Price: %{y:,.2f}<extra></extra>",
        ), row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI"], mode="lines+markers",
            line=dict(color=ORANGE, width=1.8),
            marker=dict(color=dot_colors, size=5, line=dict(width=0)),
            name="RSI",
            hovertemplate="%{x|%b %d %Y}<br>RSI: %{y:.1f}<extra></extra>",
        ), row=2, col=1,
    )

    oversold, overbought = thresholds
    fig.add_hrect(y0=overbought, y1=100, row=2, col=1, fillcolor=RED, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=oversold, row=2, col=1, fillcolor=GREEN, opacity=0.08, line_width=0)

    # No in-plot text label - the shaded bands already convey the thresholds, and a text label
    # sitting directly on the line tends to overlap the most recent (rightmost) RSI data.
    for y, color in [(oversold, GREEN), (overbought, RED)]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=1.2, row=2, col=1)

    fig.update_layout(
        height=680, width=1200,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(
            text=f"<b>{ticker}</b>  ·  {industry}  ·  Weekly RSI",
            font=dict(size=18, color=TEXTCLR), x=0.04, y=0.97,
        ),
        showlegend=False,
        margin=dict(t=70, b=40, l=60, r=40),
        hovermode="x unified",
    )

    ytype = "log" if log_price else "linear"
    fig.update_yaxes(title_text="Price", row=1, col=1, type=ytype, gridcolor="#555", gridwidth=0.5, zeroline=False, tickfont=dict(size=11))
    _rsi_range = rsi_range if rsi_range is not None else [0, 100]
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=_rsi_range, gridcolor="#555", gridwidth=0.5, zeroline=False, tickfont=dict(size=11))
    fig.update_xaxes(gridcolor="#555", gridwidth=0.4, showticklabels=True, tickfont=dict(size=11))

    # No Buy/Hold/Sell text legend - the dashed lines + shaded bands on the RSI panel already show
    # the thresholds; a text legend here was overlapping the RSI line data on narrower renders.

    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.04,
        sizex=0.16, sizey=0.16, xanchor="right", yanchor="bottom",
        opacity=0.90, layer="above",
    ))

    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def build_report() -> str:
    parts = []

    print("Downloading broad-market ETFs...")
    fetch_all(list(MARKET_SYMBOLS.keys()))
    parts.append(section_header("Broad Market ETFs", "Weekly RSI-14 &middot; Weekly RSI-52 (1-Year)"))
    for ticker, name in MARKET_SYMBOLS.items():
        try:
            df_w = get_weekly_rsi(ticker, rsi_length=14)
            parts.append(fig_to_div(plot_price_rsi(df_w, ticker, name, log_price=True)))

            # A 52-period RSI is much smoother than a 14-period one and rarely reaches 40/75 -
            # 45/65 are the meaningful oversold/overbought levels at this length, and both actually
            # fall inside the [30, 70] display range below (40/75 didn't - the red line was clipped off).
            # Same thresholds passed to get_weekly_rsi() too, so the dot colors (buy/hold/sell) agree
            # with where the dashed lines actually are, instead of using the 14-period 40/75 cutoffs.
            df_52 = get_weekly_rsi(ticker, rsi_length=52, thresholds=(45, 65))
            fig_52 = plot_price_rsi(df_52, ticker, name, log_price=True, rsi_range=[30, 70], thresholds=(45, 65))
            fig_52.update_layout(title_text=f"<b>{ticker}</b>  ·  {name}  ·  Weekly RSI (1-Year / 52-period)")
            parts.append(fig_to_div(fig_52))
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print("Downloading sector ETFs...")
    fetch_all(list(SECTOR_SYMBOLS.keys()))
    parts.append(section_header("Sector ETFs", "Weekly RSI-14 &middot; Weekly RSI-52 (1-Year)"))
    for ticker, name in SECTOR_SYMBOLS.items():
        try:
            df_w = get_weekly_rsi(ticker, rsi_length=14)
            parts.append(fig_to_div(plot_price_rsi(df_w, ticker, name, log_price=True)))

            df_52 = get_weekly_rsi(ticker, rsi_length=52, thresholds=(45, 65))
            fig_52 = plot_price_rsi(df_52, ticker, name, log_price=True, rsi_range=[30, 70], thresholds=(45, 65))
            fig_52.update_layout(title_text=f"<b>{ticker}</b>  ·  {name}  ·  Weekly RSI (1-Year / 52-period)")
            parts.append(fig_to_div(fig_52))
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print("Downloading industry ETFs...")
    fetch_all(list(INDUSTRY_SYMBOLS.keys()))

    parts.append(section_header("Industry ETFs", "Weekly RSI-14"))
    for ticker, industry in INDUSTRY_SYMBOLS.items():
        try:
            df = get_weekly_rsi(ticker)
            parts.append(fig_to_div(plot_price_rsi(df, ticker, industry)))
        except Exception as e:
            print(f"Error {ticker}: {e}")

    parts.append(section_header("Industry ETFs", "Monthly RSI-14"))
    for ticker, industry in INDUSTRY_SYMBOLS.items():
        try:
            df = get_monthly_rsi(ticker)
            fig = plot_price_rsi(df, ticker, industry)
            fig.update_layout(title_text=f"<b>{ticker}</b>  ·  {industry}  ·  Monthly RSI")
            parts.append(fig_to_div(fig))
        except Exception as e:
            print(f"Error {ticker}: {e}")

    return "\n".join(parts)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Industry &amp; Broad Market RSI Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin-top: 6px; }}
  .chart-wrap {{ padding: 8px 24px; }}
</style>
</head>
<body>
<header>
  <h1>Industry &amp; Broad Market RSI Report</h1>
  <div class="meta">Generated {date_str} &middot; Broad market and sector ETFs, weekly and monthly RSI &middot; Crypto excluded</div>
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

    out_path = os.path.join(OUTPUT_DIR, "Industry_RSI_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
