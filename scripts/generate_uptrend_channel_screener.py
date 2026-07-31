"""
1-Year Uptrend Regression Channel Screener.

Idea: find stocks that have been in a clean, strong uptrend over the past
year — a tight regression channel (high R^2) combined with a strong actual
1-year return, the same combo that would surface a name like UMAC after it
broke into a persistent uptrend. Unlike the 10-year channel screener (which
looks for pullbacks/extensions at the edge of a long-term channel), this one
ranks the whole universe by a blended "quality of trend" score and lists the
single best combo names first, regardless of where price currently sits
inside the channel.

Composite score = 50% percentile-rank(R^2) + 50% percentile-rank(1Y return),
computed only over names that already pass the hard filters:

  1. Positive regression slope (an actual uptrend, not a downtrend that
     happens to be a tight line).
  2. R^2 of the 1y log-price regression >= MIN_R2 (a clean channel).
  3. 1Y return >= MIN_1Y_RETURN_PCT (an uptrend that has actually paid off,
     not just a statistically tight sideways drift).

Universe: S&P 500 + Nasdaq-100 + every ticker in data/koyfin_us.csv and
data/us_1w_rev_est_screener.csv (mid/large-cap skew — see README note in
the repo about small-cap coverage limits of these source CSVs).
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
from plotly.subplots import make_subplots
from scipy.stats import linregress

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import (
    CAP_FILTER_CONTROL_HTML,
    CAP_FILTER_CSS,
    CAP_FILTER_JS,
    cap_tier,
    load_nasdaq100_symbols,
    load_sp500_symbols,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "uptrend-channel-screener")
LOGO_PATH = os.path.join(REPO_ROOT, "assets", "Logo_Transparent_1200px.png")
REV_SCREENER_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")
KOYFIN_US_PATH = os.path.join(REPO_ROOT, "data", "koyfin_us.csv")

BENCH_TICKER = "QQQ"
LOOKBACK_PERIOD = "1y"
MIN_TRADING_DAYS = 230  # ~91% of the ~252 trading days in 1y; excludes recent IPOs
CHANNEL_SIGMA = 2.0  # width of the plotted upper/lower channel bands
MIN_R2 = 0.60  # regression channel must be reasonably clean to even qualify
MIN_1Y_RETURN_PCT = 15.0  # uptrend must have actually paid off
R2_WEIGHT = 0.5  # composite score weighting: correlation quality...
RETURN_WEIGHT = 0.5  # ...vs. actual 1Y return
TOP_N = 60  # cap the number of charts rendered, after ranking by composite
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
    koyfin_df = pd.read_csv(KOYFIN_US_PATH)

    rev_tickers = rev_df["Ticker"].dropna().astype(str).tolist()
    koyfin_tickers = koyfin_df["Ticker"].dropna().astype(str).tolist()

    name_map = dict(zip(rev_df["Ticker"].astype(str), rev_df["Name"].astype(str)))
    name_map.update(dict(zip(koyfin_df["Ticker"].astype(str), koyfin_df["Name"].astype(str))))

    combined = set(sp500) | set(nasdaq100) | set(rev_tickers) | set(koyfin_tickers)
    normalized = sorted({t.strip().upper().replace(".", "-") for t in combined if t and t.strip()})
    return normalized, name_map


def load_revision_map():
    """Ticker -> dict of FY1E/FY2E/FY3E revenue-estimate revision %s (1W/1M/3M/6M/1Y),
    for the analyst revision cascade panel. Only covers tickers present in the
    rev-screener CSV; other universe names simply won't get this panel."""
    raw = pd.read_csv(REV_SCREENER_PATH)
    df = raw.rename(columns=REV_COL_MAP)
    keep = [c for c in REV_COL_MAP.values() if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        if c != "ticker":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return df.set_index("ticker").to_dict("index")


def load_cap_map():
    """Ticker -> Market Cap ($M), for the cap-tier filter. Merges both source
    CSVs; koyfin_us.csv values win on overlap since that export skews more
    current for the large/mid-cap names it covers."""
    cap_map = {}
    for path in (REV_SCREENER_PATH, KOYFIN_US_PATH):
        raw = pd.read_csv(path)
        tickers = raw["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        mktcap = pd.to_numeric(raw["Market Cap"], errors="coerce")
        cap_map.update(dict(zip(tickers, mktcap)))
    return cap_map


def load_sector_map():
    """Ticker -> (Sector, Industry) lookup from the Koyfin CSV export. Empty
    for tickers not present there, since sector/industry labels are a
    nice-to-have annotation, not a screening input."""
    if not os.path.exists(KOYFIN_US_PATH):
        print(f"  no Koyfin export at {KOYFIN_US_PATH}, sector/industry labels will be blank")
        return {}
    df = pd.read_csv(KOYFIN_US_PATH)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return {
        row["Ticker"]: (row.get("Sector", "") or "", row.get("Industry", "") or "")
        for _, row in df.iterrows()
    }


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

    r2 = r_value**2
    if r2 < MIN_R2:
        return None

    fitted = intercept + slope * x
    resid = y - fitted
    std = resid.std()
    if std == 0:
        return None
    z_last = resid[-1] / std

    return_1y = (close.iloc[-1] / close.iloc[0] - 1) * 100
    if return_1y < MIN_1Y_RETURN_PCT:
        return None

    aligned = pd.DataFrame({"stock": close, "bench": bench_close}).dropna()
    bench_return_1y = (
        (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100 if len(aligned) >= 2 else np.nan
    )
    annual_trend_return_pct = (np.exp(slope * 252) - 1) * 100

    return {
        "Ticker": close.name,
        "R2": r2,
        "Z_Score": z_last,
        "1Y_Return_%": return_1y,
        f"1Y_{BENCH_TICKER}_Return_%": bench_return_1y,
        "Outperformance_%": return_1y - bench_return_1y if pd.notna(bench_return_1y) else np.nan,
        "Annual_Trend_Return_%": annual_trend_return_pct,
        "_x": x,
        "_y": y,
        "_slope": slope,
        "_intercept": intercept,
        "_std": std,
        "_close": close,
    }


def add_composite_scores(rows):
    """Blend percentile-rank(R^2) and percentile-rank(1Y return) into a single
    0-100 composite score so the list surfaces the best combo of a clean
    channel AND a strong actual return, not just one or the other."""
    if not rows:
        return rows
    r2_series = pd.Series([r["R2"] for r in rows])
    ret_series = pd.Series([r["1Y_Return_%"] for r in rows])
    r2_pct = r2_series.rank(pct=True)
    ret_pct = ret_series.rank(pct=True)
    for row, r2p, retp in zip(rows, r2_pct, ret_pct):
        row["Composite_Score"] = (R2_WEIGHT * r2p + RETURN_WEIGHT * retp) * 100
    return rows


def build_channel_chart(row, company_name, revision_row=None, sector_industry=None):
    close = row["_close"]
    x, slope, intercept, std = row["_x"], row["_slope"], row["_intercept"], row["_std"]
    center = np.exp(intercept + slope * x)
    upper = np.exp(intercept + slope * x + CHANNEL_SIGMA * std)
    lower = np.exp(intercept + slope * x - CHANNEL_SIGMA * std)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=["1Y Regression Channel", "Analyst Revenue Revision Trend"],
        horizontal_spacing=0.09,
    )

    fig.add_trace(go.Scatter(x=close.index, y=upper, mode="lines", line=dict(color=GREY_LINE, width=1, dash="dot"), name=f"+{CHANNEL_SIGMA:.0f}σ", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=close.index, y=lower, mode="lines", line=dict(color=GREY_LINE, width=1, dash="dot"), name=f"-{CHANNEL_SIGMA:.0f}σ", fill="tonexty", fillcolor="rgba(255,255,255,0.05)", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=close.index, y=center, mode="lines", line=dict(color=ORANGE, width=1.5, dash="dash"), name="Regression", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", line=dict(color=BLUE, width=1.8), name="Close", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[close.index[-1]], y=[close.values[-1]], mode="markers",
        marker=dict(color=RED, size=9, line=dict(color=TEXTCLR, width=1)),
        name="Last", showlegend=False, hovertemplate=f"Z={row['Z_Score']:.2f}σ<extra></extra>",
    ), row=1, col=1)

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

    sector_line = f"<br><span style=\"font-size:12px;color:{GREY_LINE}\">{sector_industry}</span>" if sector_industry else ""
    fig.update_layout(
        height=520, width=1600,
        paper_bgcolor=DGREY, plot_bgcolor=LGREY,
        font=dict(family="Arial, sans-serif", color=TEXTCLR, size=12),
        title=dict(
            text=(
                f"<b>{row['Ticker']}</b>  ·  {company_name}  ·  "
                f"Score={row['Composite_Score']:.0f}  ·  R²={row['R2']:.2f}  ·  "
                f"1Y Return: {row['1Y_Return_%']:+.0f}%  ·  Z={row['Z_Score']:.2f}σ{sector_line}"
            ),
            font=dict(size=15, color=TEXTCLR), x=0.03, y=0.97,
        ),
        margin=dict(t=105 if sector_industry else 90, b=40, l=60, r=40),
        hovermode="x unified",
        barmode="group",
        legend=dict(orientation="h", y=1.13, x=0.64, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Price (log)", type="log", gridcolor="#555", gridwidth=0.5, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Revenue Revision %", ticksuffix="%", gridcolor="#555", gridwidth=0.5, zeroline=True, zerolinecolor="#555", row=1, col=2)
    fig.update_xaxes(gridcolor="#555", gridwidth=0.4)
    for ann in fig["layout"]["annotations"]:
        if ann.text in ("1Y Regression Channel", "Analyst Revenue Revision Trend"):
            ann.font.size = 12
            ann.font.color = TEXTCLR

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
    df = df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    cols = ["Ticker", "Cap", "Composite_Score", "R2", "1Y_Return_%", f"1Y_{BENCH_TICKER}_Return_%", "Outperformance_%", "Z_Score", "Annual_Trend_Return_%"]
    df = df[cols]
    html = ['<table class="summary"><thead><tr>']
    html.append("".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        html.append(
            "<tr>"
            f"<td>{r['Ticker']}</td>"
            f"<td>{r['Cap']}</td>"
            f"<td>{r['Composite_Score']:.0f}</td>"
            f"<td>{r['R2']:.2f}</td>"
            f"<td>{r['1Y_Return_%']:.0f}%</td>"
            f"<td>{r[f'1Y_{BENCH_TICKER}_Return_%']:.0f}%</td>"
            f"<td>{r['Outperformance_%']:+.0f}%</td>"
            f"<td>{r['Z_Score']:.2f}</td>"
            f"<td>{r['Annual_Trend_Return_%']:.1f}%</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return "".join(html)


def build_section(rows, name_map, revision_map, sector_map=None):
    rows = sorted(rows, key=lambda r: r["Composite_Score"], reverse=True)
    survivors = rows[:TOP_N]
    print(f"{len(rows)} candidates pass all filters, keeping top {len(survivors)} by composite score")

    parts = ['<div class="section"><h2>Strongest 1Y Uptrend Channels</h2>']
    if survivors:
        parts.append(
            f'<div class="section-sub">{len(survivors)} of {len(rows)} filtered names, '
            f"ranked by composite score ({R2_WEIGHT:.0%} R² percentile + {RETURN_WEIGHT:.0%} 1Y return percentile)"
            f"</div>{build_summary_table(survivors)}"
        )
    else:
        parts.append('<div class="section-sub">No tickers passed all filters today.</div>')
    parts.append("</div>")

    for row in survivors:
        company_name = name_map.get(row["Ticker"], row["Ticker"])
        revision_row = revision_map.get(row["Ticker"])
        sector, industry = (sector_map or {}).get(row["Ticker"], ("", ""))
        sector_industry = " | ".join(s for s in (sector, industry) if s)
        try:
            fig = build_channel_chart(row, company_name, revision_row, sector_industry)
            parts.append(f'<div class="chart-wrap" data-cap="{row["Cap"]}">{fig_to_div(fig)}</div>')
        except Exception as exc:
            print(f"  chart error {row['Ticker']}: {exc}")

    return "\n".join(parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    universe, name_map = load_universe()
    revision_map = load_revision_map()
    cap_map = load_cap_map()
    sector_map = load_sector_map()
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
                row["Cap"] = cap_tier(cap_map.get(ticker))
                rows.append(row)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"{len(rows)} tickers pass the uptrend + R^2 + return filters")
    rows = add_composite_scores(rows)

    body = build_section(rows, name_map, revision_map, sector_map)

    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body=body,
                                 cap_css=CAP_FILTER_CSS, cap_js=CAP_FILTER_JS, cap_control=CAP_FILTER_CONTROL_HTML)
    out_path = os.path.join(OUTPUT_DIR, "Uptrend_Channel_Screener_latest.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>1-Year Uptrend Channel Screener</title>
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
{cap_css}
</style>
{cap_js}
</head>
<body>
<header>
  <h1>1-Year Uptrend Channel Screener</h1>
  <div class="meta">Generated {date_str} &middot; Strongest 1-year regression-channel fit (R&sup2;) combined with strongest actual 1-year return &middot; Universe: S&amp;P 500 + Nasdaq-100 + koyfin_us.csv + revenue-revision screener</div>
  {cap_control}
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
