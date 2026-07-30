"""
Weekly RSI Bottoming Screener.

Idea: find names that look like they could be putting in a bottom after a
real decline — not just noise. Four filters:

  1. Weekly RSI(14) recently crossed back above 30 (oversold -> recovering).
  2. Sometime in the past 2 years, weekly RSI was above 75 (proves the name
     has real momentum in it when it runs, i.e. not a perpetual laggard).
  3. Over the full 10-year lookback, the stock has cumulatively
     outperformed QQQ (ratio of stock/QQQ higher today than 10y ago).
  4. Analyst sales (revenue) estimate revision trend charted alongside,
     same panel as the Rev Revision Screener / Channel Screener.

Universe: S&P 500 + Nasdaq-100 + every ticker in data/us_1w_rev_est_screener.csv
(same universe as generate_channel_screener.py).
"""

import base64
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import load_nasdaq100_symbols, load_sp500_symbols

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rsi-bottoming-screener")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
REV_SCREENER_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")

BENCH_TICKER = "QQQ"
LOOKBACK_PERIOD = "10y"
MIN_TRADING_DAYS = 2400  # ~95% of the ~2520 trading days in 10y; excludes recent IPOs
RSI_LENGTH = 14
OVERSOLD = 30
OVERBOUGHT = 75
RECENT_CROSS_WEEKS = 16  # the RSI>30 crossover must have happened within this many weeks
OVERBOUGHT_LOOKBACK_YEARS = 2  # RSI must have touched OVERBOUGHT sometime in this window
CHUNK_SIZE = 250  # yfinance batch download size

ORANGE = "#C67A29"
BLUE = "#1F79BE"
DBLUE = "#4B8EA9"
DGREY = "#363636"
LGREY = "#4A4A4A"
GREEN = "#44A660"
RED = "#A22A2A"
TEXTCLR = "#E8E8E8"
GREY_LINE = "#8E8E93"

REV_COL_MAP = {
    "Ticker": "ticker",
    "Revenues Est Avg Rev % (FY1E - 1W)": "fy1_1w", "Revenues Est Avg Rev % (FY1E - 1M)": "fy1_1m",
    "Revenues Est Avg Rev % (FY1E - 3M)": "fy1_3m", "Revenues Est Avg Rev % (FY1E - 6M)": "fy1_6m",
    "Revenues Est Avg Rev % (FY1E - 1Y)": "fy1_1y",
    "Revenues Est Avg Rev % (FY2E - 1W)": "fy2_1w", "Revenues Est Avg Rev % (FY2E - 1M)": "fy2_1m",
    "Revenues Est Avg Rev % (FY2E - 3M)": "fy2_3m", "Revenues Est Avg Rev % (FY2E - 6M)": "fy2_6m",
    "Revenues Est Avg Rev % (FY2E - 1Y)": "fy2_1y",
    "Revenues Est Avg Rev % (FY3E - 1W)": "fy3_1w", "Revenues Est Avg Rev % (FY3E - 1M)": "fy3_1m",
    "Revenues Est Avg Rev % (FY3E - 3M)": "fy3_3m", "Revenues Est Avg Rev % (FY3E - 6M)": "fy3_6m",
    "Revenues Est Avg Rev % (FY3E - 1Y)": "fy3_1y",
}
REV_WIN_KEYS = ["1w", "1m", "3m", "6m", "1y"]
REV_WIN_LABELS = ["1W", "1M", "3M", "6M", "1Y"]
FY_COLORS = [BLUE, DBLUE, ORANGE]

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


def load_revision_map():
    raw = pd.read_csv(REV_SCREENER_PATH)
    df = raw.rename(columns=REV_COL_MAP)
    keep = [c for c in REV_COL_MAP.values() if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        if c != "ticker":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return df.set_index("ticker").to_dict("index")


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


def score_ticker(close, bench_close, today):
    if len(close) < MIN_TRADING_DAYS:
        return None

    weekly_close = close.resample("W").last().dropna()
    rsi = RSIIndicator(weekly_close, RSI_LENGTH).rsi().dropna()
    if len(rsi) < RSI_LENGTH * 2:
        return None

    crossed_above = (rsi.shift(1) < OVERSOLD) & (rsi >= OVERSOLD)
    cross_dates = rsi.index[crossed_above]
    if len(cross_dates) == 0:
        return None
    last_cross = cross_dates[-1]
    weeks_since_cross = (today - last_cross.to_pydatetime()).days / 7
    if weeks_since_cross > RECENT_CROSS_WEEKS or rsi.iloc[-1] < OVERSOLD:
        return None

    two_year_cutoff = today - timedelta(days=365 * OVERBOUGHT_LOOKBACK_YEARS)
    past_rsi = rsi[rsi.index >= pd.Timestamp(two_year_cutoff)]
    if past_rsi.empty or past_rsi.max() < OVERBOUGHT:
        return None
    max_rsi_date = past_rsi.idxmax()

    aligned = pd.DataFrame({"stock": close, "bench": bench_close}).dropna()
    if len(aligned) < MIN_TRADING_DAYS:
        return None
    ratio_start = aligned["stock"].iloc[0] / aligned["bench"].iloc[0]
    ratio_end = aligned["stock"].iloc[-1] / aligned["bench"].iloc[-1]
    if ratio_end <= ratio_start:
        return None

    stock_return_10y = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[0] - 1) * 100
    bench_return_10y = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100

    return {
        "Ticker": close.name,
        "Current_RSI": rsi.iloc[-1],
        "Weeks_Since_Cross": weeks_since_cross,
        "Max_RSI_2Y": past_rsi.max(),
        "10Y_Return_%": stock_return_10y,
        f"10Y_{BENCH_TICKER}_Return_%": bench_return_10y,
        "Outperformance_%": stock_return_10y - bench_return_10y,
        "_weekly_close": weekly_close,
        "_rsi": rsi,
        "_last_cross": last_cross,
        "_max_rsi_date": max_rsi_date,
    }


def build_chart(row, company_name, revision_row=None):
    weekly_close, rsi = row["_weekly_close"], row["_rsi"]

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.6, 0.4], column_widths=[0.62, 0.38],
        vertical_spacing=0.06, horizontal_spacing=0.09,
        specs=[[{}, {"rowspan": 2}], [{}, None]],
    )

    fig.add_trace(go.Scatter(
        x=weekly_close.index, y=weekly_close.values, mode="lines",
        line=dict(color=BLUE, width=1.8), name="Weekly Close", showlegend=False,
        hovertemplate="%{x|%b %d %Y}<br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi.values, mode="lines",
        line=dict(color=ORANGE, width=1.8), name="Weekly RSI(14)", showlegend=False,
        hovertemplate="%{x|%b %d %Y}<br>RSI: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[row["_last_cross"]], y=[rsi.loc[row["_last_cross"]]], mode="markers",
        marker=dict(color=GREEN, size=10, line=dict(color=TEXTCLR, width=1)),
        name="Cross > 30", showlegend=False,
        hovertemplate=f"Crossed above {OVERSOLD}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[row["_max_rsi_date"]], y=[rsi.loc[row["_max_rsi_date"]]], mode="markers",
        marker=dict(color=RED, size=9, symbol="diamond", line=dict(color=TEXTCLR, width=1)),
        name=f"{OVERBOUGHT_LOOKBACK_YEARS}Y High RSI", showlegend=False,
        hovertemplate=f"2Y peak RSI<extra></extra>",
    ), row=2, col=1)

    fig.add_hrect(y0=OVERBOUGHT, y1=100, row=2, col=1, fillcolor=RED, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=OVERSOLD, row=2, col=1, fillcolor=GREEN, opacity=0.08, line_width=0)
    fig.add_hline(y=OVERSOLD, line_dash="dash", line_color=GREEN, line_width=1.2, row=2, col=1)
    fig.add_hline(y=OVERBOUGHT, line_dash="dash", line_color=RED, line_width=1.2, row=2, col=1)

    if revision_row:
        for fy_idx, fy in enumerate(["fy1", "fy2", "fy3"]):
            vals = [revision_row.get(f"{fy}_{w}", np.nan) for w in REV_WIN_KEYS]
            vals = [v * 100 if pd.notna(v) else None for v in vals]
            fig.add_trace(go.Bar(
                x=REV_WIN_LABELS, y=vals, name=f"FY{fy_idx + 1}E", marker_color=FY_COLORS[fy_idx],
                opacity=0.85, hovertemplate="%{x}: %{y:+.2f}%<extra>FY" + str(fy_idx + 1) + "E</extra>",
            ), row=1, col=2)
    else:
        fig.add_annotation(
            x=0.81, y=0.5, xref="paper", yref="paper", text="No revision data",
            showarrow=False, font=dict(size=12, color=GREY_LINE),
        )

    fig.update_layout(
        height=520, width=1600,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(
            text=(
                f"<b>{row['Ticker']}</b>  ·  {company_name}  ·  "
                f"RSI={row['Current_RSI']:.0f} (crossed >{OVERSOLD} {row['Weeks_Since_Cross']:.1f}w ago)  ·  "
                f"{OVERBOUGHT_LOOKBACK_YEARS}Y peak RSI={row['Max_RSI_2Y']:.0f}  ·  "
                f"10Y Outperformance vs {BENCH_TICKER}: {row['Outperformance_%']:+.0f}%"
            ),
            font=dict(size=14, color=TEXTCLR), x=0.03, y=0.97,
        ),
        margin=dict(t=90, b=40, l=60, r=40),
        barmode="group",
        legend=dict(orientation="h", y=1.13, x=0.64, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Price", gridcolor="#555", gridwidth=0.5, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], gridcolor="#555", gridwidth=0.5, zeroline=False, row=2, col=1)
    fig.update_yaxes(title_text="Revenue Revision %", ticksuffix="%", gridcolor="#555", gridwidth=0.5, zeroline=True, zerolinecolor="#555", row=1, col=2)
    fig.update_xaxes(gridcolor="#555", gridwidth=0.4)

    fig.add_layout_image(dict(
        source=LOGO_B64, xref="paper", yref="paper", x=1.0, y=1.16,
        sizex=0.10, sizey=0.10, xanchor="right", yanchor="bottom",
        opacity=0.90, layer="above",
    ))
    return fig


def fig_to_div(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def build_summary_table(rows):
    df = pd.DataFrame(rows).drop(columns=[c for c in rows[0] if c.startswith("_")])
    df = df.sort_values(["Weeks_Since_Cross", "Outperformance_%"], ascending=[True, False]).reset_index(drop=True)
    cols = ["Ticker", "Current_RSI", "Weeks_Since_Cross", "Max_RSI_2Y", "10Y_Return_%", f"10Y_{BENCH_TICKER}_Return_%", "Outperformance_%"]
    df = df[cols]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        html.append(
            "<tr>"
            f"<td>{r['Ticker']}</td>"
            f"<td>{r['Current_RSI']:.0f}</td>"
            f"<td>{r['Weeks_Since_Cross']:.1f}</td>"
            f"<td>{r['Max_RSI_2Y']:.0f}</td>"
            f"<td>{r['10Y_Return_%']:.0f}%</td>"
            f"<td>{r[f'10Y_{BENCH_TICKER}_Return_%']:.0f}%</td>"
            f"<td>{r['Outperformance_%']:+.0f}%</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    universe, name_map = load_universe()
    revision_map = load_revision_map()
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
            row = score_ticker(close, bench_close, today)
            if row is not None:
                rows.append(row)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"{len(rows)} tickers pass the RSI-bottoming + outperformance filters")

    parts = []
    if rows:
        rows_sorted = sorted(rows, key=lambda r: (r["Weeks_Since_Cross"], -r["Outperformance_%"]))
        parts.append(
            f'<div class="section"><h2>Candidates</h2><div class="section-sub">'
            f"{len(rows_sorted)} names: weekly RSI crossed back above {OVERSOLD} within {RECENT_CROSS_WEEKS} weeks, "
            f"touched {OVERBOUGHT}+ sometime in the past {OVERBOUGHT_LOOKBACK_YEARS}Y, and have outperformed "
            f"{BENCH_TICKER} over 10Y &mdash; sorted by freshest crossover first</div>{build_summary_table(rows_sorted)}</div>"
        )
        for row in rows_sorted:
            company_name = name_map.get(row["Ticker"], row["Ticker"])
            revision_row = revision_map.get(row["Ticker"])
            try:
                fig = build_chart(row, company_name, revision_row)
                parts.append(f'<div class="chart-wrap">{fig_to_div(fig)}</div>')
            except Exception as exc:
                print(f"  chart error {row['Ticker']}: {exc}")
    else:
        parts.append('<div class="section"><h2>Candidates</h2><div class="section-sub">No tickers passed all filters today.</div></div>')

    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "RSI_Bottoming_Screener_latest.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Weekly RSI Bottoming Screener</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin: 6px 0 16px; }}
  .chart-wrap {{ padding: 8px 12px; }}
  table.summary {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.summary th, table.summary td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #3A3A3C; }}
  table.summary th:first-child, table.summary td:first-child {{ text-align: left; }}
  table.summary th {{ color: #C67A29; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>Weekly RSI Bottoming Screener</h1>
  <div class="meta">Generated {date_str} &middot; Weekly RSI recently recovered above 30, was overbought (75+) within the past 2 years, 10Y outperformer vs QQQ &middot; Universe: S&amp;P 500 + Nasdaq-100 + revenue-revision screener</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
