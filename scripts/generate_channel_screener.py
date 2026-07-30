"""
10-Year Regression Channel Screener.

Idea: find stocks that have been in a clean long-term uptrend (like the
QQQ/SPY "regression channel" Chart of Interest) but are currently sitting
near the bottom of that channel — i.e. a normal pullback within an
established trend, not a trend break. Two extra filters keep the list
honest:

  1. R^2 of the 10y log-price regression must be high (a clean channel,
     not a noisy one) — candidates are ranked by R^2 and only the top
     half survive.
  2. The stock must have actually outperformed QQQ cumulatively over the
     same 10-year window (ratio of stock/QQQ higher today than 10y ago).

Universe: S&P 500 + Nasdaq-100 + every ticker in data/us_1w_rev_est_screener.csv.
"""

import base64
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import load_nasdaq100_symbols, load_sp500_symbols

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "channel-screener")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
REV_SCREENER_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")

BENCH_TICKER = "QQQ"
LOOKBACK_PERIOD = "10y"
MIN_TRADING_DAYS = 2400  # ~95% of the ~2520 trading days in 10y; excludes recent IPOs
CHANNEL_SIGMA = 2.0  # width of the plotted upper/lower channel bands
BOTTOM_Z_THRESHOLD = -1.5  # current residual must be at/below this to count as "at the bottom"
TOP_R2_FRACTION = 0.5  # keep only the top half of survivors, ranked by R^2
CHUNK_SIZE = 250  # yfinance batch download size

ORANGE = "#C67A29"
BLUE = "#1F79BE"
DGREY = "#363636"
LGREY = "#4A4A4A"
GREEN = "#44A660"
RED = "#A22A2A"
TEXTCLR = "#E8E8E8"

with open(LOGO_PATH, "rb") as f:
    LOGO_B64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load_universe():
    sp500 = load_sp500_symbols()
    nasdaq100 = load_nasdaq100_symbols()
    rev_df = pd.read_csv(REV_SCREENER_PATH)
    rev_tickers = rev_df["Ticker"].dropna().astype(str).tolist()
    name_map = dict(zip(rev_df["Ticker"].astype(str), rev_df["Name"].astype(str)))

    combined = set(sp500) | set(nasdaq100) | set(rev_tickers)
    normalized = sorted({t.strip().upper().replace(".", "-") for t in combined if t and t.strip()})
    return normalized, name_map


def close_series_single(ticker, raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty or "Close" not in raw:
        return None
    close = raw["Close"].dropna()
    return close if not close.empty else None


def batch_download_closes(tickers, period, chunk_size):
    close_map = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            raw = yf.download(
                chunk, period=period, group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception as exc:
            print(f"  batch error: {exc}")
            continue
        if len(chunk) == 1:
            close = close_series_single(chunk[0], raw)
            if close is not None:
                close_map[chunk[0]] = close
            continue
        for t in chunk:
            try:
                close = raw[t]["Close"].dropna()
                if not close.empty:
                    close_map[t] = close
            except (KeyError, TypeError):
                pass
    return close_map


def score_ticker(close, bench_close):
    if len(close) < MIN_TRADING_DAYS:
        return None

    x = np.arange(len(close))
    y = np.log(close.values)
    slope, intercept, r_value, _, _ = linregress(x, y)
    if slope <= 0:
        return None

    fitted = intercept + slope * x
    resid = y - fitted
    std = resid.std()
    if std == 0:
        return None
    z_last = resid[-1] / std
    if z_last > BOTTOM_Z_THRESHOLD:
        return None

    aligned = pd.DataFrame({"stock": close, "bench": bench_close}).dropna()
    if len(aligned) < MIN_TRADING_DAYS:
        return None
    ratio_start = aligned["stock"].iloc[0] / aligned["bench"].iloc[0]
    ratio_end = aligned["stock"].iloc[-1] / aligned["bench"].iloc[-1]
    if ratio_end <= ratio_start:
        return None

    stock_return_10y = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[0] - 1) * 100
    bench_return_10y = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100
    annual_return_pct = (np.exp(slope * 252) - 1) * 100

    return {
        "Ticker": close.name,
        "R2": r_value**2,
        "Z_Score": z_last,
        "Annual_Trend_Return_%": annual_return_pct,
        "10Y_Return_%": stock_return_10y,
        f"10Y_{BENCH_TICKER}_Return_%": bench_return_10y,
        "Outperformance_%": stock_return_10y - bench_return_10y,
        "_x": x,
        "_y": y,
        "_slope": slope,
        "_intercept": intercept,
        "_std": std,
        "_close": close,
    }


def build_channel_chart(row, company_name):
    close = row["_close"]
    x, slope, intercept, std = row["_x"], row["_slope"], row["_intercept"], row["_std"]
    center = np.exp(intercept + slope * x)
    upper = np.exp(intercept + slope * x + CHANNEL_SIGMA * std)
    lower = np.exp(intercept + slope * x - CHANNEL_SIGMA * std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=upper, mode="lines", line=dict(color=GREY_LINE, width=1, dash="dot"), name=f"+{CHANNEL_SIGMA:.0f}σ"))
    fig.add_trace(go.Scatter(x=close.index, y=lower, mode="lines", line=dict(color=GREY_LINE, width=1, dash="dot"), name=f"-{CHANNEL_SIGMA:.0f}σ", fill="tonexty", fillcolor="rgba(255,255,255,0.05)"))
    fig.add_trace(go.Scatter(x=close.index, y=center, mode="lines", line=dict(color=ORANGE, width=1.5, dash="dash"), name="Regression"))
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", line=dict(color=BLUE, width=1.8), name="Close"))
    fig.add_trace(go.Scatter(
        x=[close.index[-1]], y=[close.values[-1]], mode="markers",
        marker=dict(color=RED, size=9, line=dict(color=TEXTCLR, width=1)),
        name="Last", hovertemplate=f"Z={row['Z_Score']:.2f}σ<extra></extra>",
    ))

    fig.update_layout(
        height=480, width=1100,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(
            text=(
                f"<b>{row['Ticker']}</b>  ·  {company_name}  ·  "
                f"R²={row['R2']:.2f}  ·  Z={row['Z_Score']:.2f}σ  ·  "
                f"10Y Outperformance vs {BENCH_TICKER}: {row['Outperformance_%']:+.0f}%"
            ),
            font=dict(size=15, color=TEXTCLR), x=0.03, y=0.96,
        ),
        margin=dict(t=60, b=40, l=60, r=160),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.0, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Price (log)", type="log", gridcolor="#555", gridwidth=0.5, zeroline=False)
    fig.update_xaxes(gridcolor="#555", gridwidth=0.4)

    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.12,
        sizex=0.14, sizey=0.14, xanchor="right", yanchor="bottom",
        opacity=0.90, layer="above",
    ))
    return fig


GREY_LINE = "#8E8E93"


def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def build_summary_table(rows):
    df = pd.DataFrame(rows).drop(columns=[c for c in rows[0] if c.startswith("_")])
    df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    cols = ["Ticker", "R2", "Z_Score", "10Y_Return_%", f"10Y_{BENCH_TICKER}_Return_%", "Outperformance_%", "Annual_Trend_Return_%"]
    df = df[cols]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        html.append(
            "<tr>"
            f"<td>{r['Ticker']}</td>"
            f"<td>{r['R2']:.2f}</td>"
            f"<td>{r['Z_Score']:.2f}</td>"
            f"<td>{r['10Y_Return_%']:.0f}%</td>"
            f"<td>{r[f'10Y_{BENCH_TICKER}_Return_%']:.0f}%</td>"
            f"<td>{r['Outperformance_%']:+.0f}%</td>"
            f"<td>{r['Annual_Trend_Return_%']:.1f}%</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    universe, name_map = load_universe()
    print(f"Universe: {len(universe)} unique tickers")

    print(f"Downloading {BENCH_TICKER} ({LOOKBACK_PERIOD})...")
    bench_raw = yf.download(BENCH_TICKER, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False)
    bench_close = close_series_single(BENCH_TICKER, bench_raw)
    if bench_close is None:
        raise RuntimeError(f"Could not download {BENCH_TICKER} close prices")

    print(f"Downloading {len(universe)} tickers ({LOOKBACK_PERIOD})...")
    close_map = batch_download_closes(universe, LOOKBACK_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(close_map)} tickers")

    rows = []
    for ticker, close in close_map.items():
        try:
            close = close.rename(ticker)
            row = score_ticker(close, bench_close)
            if row is not None:
                rows.append(row)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"{len(rows)} tickers pass the uptrend + bottom-of-channel + outperformance filters")

    rows.sort(key=lambda r: r["R2"], reverse=True)
    keep_n = max(1, int(np.ceil(len(rows) * TOP_R2_FRACTION))) if rows else 0
    survivors = rows[:keep_n]
    print(f"Keeping top {len(survivors)} by R^2 (top {TOP_R2_FRACTION:.0%})")

    parts = []
    if survivors:
        parts.append(f'<div class="section"><h2>Candidates</h2><div class="section-sub">{len(survivors)} of {len(rows)} filtered names, ranked by R² (10y log-price regression fit), highest first</div>{build_summary_table(survivors)}</div>')
        for row in survivors:
            company_name = name_map.get(row["Ticker"], row["Ticker"])
            try:
                fig = build_channel_chart(row, company_name)
                parts.append(f'<div class="chart-wrap">{fig_to_div(fig)}</div>')
            except Exception as exc:
                print(f"  chart error {row['Ticker']}: {exc}")
    else:
        parts.append('<div class="section"><h2>Candidates</h2><div class="section-sub">No tickers passed all filters today.</div></div>')

    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Channel_Screener_latest.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>10-Year Regression Channel Screener</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin: 6px 0 16px; }}
  .chart-wrap {{ padding: 8px 24px; }}
  table.summary {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.summary th, table.summary td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #3A3A3C; }}
  table.summary th:first-child, table.summary td:first-child {{ text-align: left; }}
  table.summary th {{ color: #C67A29; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>10-Year Regression Channel Screener</h1>
  <div class="meta">Generated {date_str} &middot; Long-term uptrend, currently at the bottom of the channel, historical outperformer vs QQQ &middot; Universe: S&amp;P 500 + Nasdaq-100 + revenue-revision screener</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
