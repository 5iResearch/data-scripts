"""
Recurring Market Musings screen: wired into the daily GitHub Actions
automation (.github/workflows/market_musings_ratio_channel_study.yml).
Run manually: python scripts/market_musings_ratio_channel_study.py

Reuses the universe/loader/download plumbing from generate_ratio_channel_screener.py.

For each benchmark (SPY/S&P 500, QQQ/Nasdaq-100, XIC/TSX), screens the 15
cleanest (highest R^2) names currently at the bottom of their 10y
price/benchmark ratio channel, separately the 15 cleanest in the middle, and
separately the 15 cleanest at the top. Every name must have: R^2 >=
SCREEN_MIN_R2, a positive ratio-regression slope, and positive cumulative 10Y
relative strength (i.e. it has actually outperformed its benchmark, not just
a statistically tight line).

Layout: the Bottom-of-Channel section leads with all three benchmark tables
together, then all three benchmarks' charts follow (each delineated with its
own "vs BENCH" header) — these are the names the article is about. Middle
and Top sections keep the original table-then-charts-per-benchmark layout,
included for reference/context underneath.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ratio_channel_screener import (
    BENCH_LABELS,
    BENCH_QQQ,
    BENCH_SPY,
    BENCH_XIC,
    CHUNK_SIZE,
    MIN_TRADING_DAYS,
    REPO_ROOT,
    batch_download_closes,
    build_channel_chart,
    close_series_single,
    display_ticker,
    fig_to_div,
    load_cap_map,
    load_revision_map,
    load_sector_map,
    load_universe,
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "market-musings")

SCREEN_MIN_R2 = 0.50  # looser than the daily screener's 0.60 — this is an editorial piece, not the live screen
Z_THRESHOLD = 1.5  # |Z| >= this counts as "at a channel extreme" (same bar as the daily screener); note the
                    # plotted chart bands use the screener's own CHANNEL_SIGMA (2.0), a separate constant —
                    # unchanged, pre-existing behavior from build_channel_chart.
TOP_N_PER_TABLE = 15

DOWNLOAD_PERIOD = "10y"  # only need the trailing WINDOW_TRADING_DAYS for the "as of today" fit
WINDOW_TRADING_DAYS = 2520  # ~10 trading years — the window used for every regression fit


def build_aligned(close, bench_close):
    return pd.DataFrame({"stock": close, "bench": bench_close}).dropna()


def fit_ratio_regression(aligned):
    """Fits a single log(stock/bench) regression over the given aligned frame
    as-is. Returns None if the fit doesn't qualify."""
    if len(aligned) < MIN_TRADING_DAYS:
        return None

    ratio = aligned["stock"] / aligned["bench"]
    x = np.arange(len(ratio))
    y = np.log(ratio.values)
    slope, intercept, r_value, _, _ = linregress(x, y)
    r2 = r_value**2
    if slope <= 0 or r2 < SCREEN_MIN_R2:
        return None

    rel_strength_pct = (ratio.iloc[-1] / ratio.iloc[0] - 1) * 100
    if rel_strength_pct <= 0:
        return None  # must have actually outperformed cumulatively, not just a positive-slope fit

    resid = y - (intercept + slope * x)
    std = resid.std()
    if std == 0:
        return None
    z_last = resid[-1] / std

    stock_return_pct = (aligned["stock"].iloc[-1] / aligned["stock"].iloc[0] - 1) * 100
    bench_return_pct = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0] - 1) * 100
    annual_trend_pct = (np.exp(slope * 252) - 1) * 100

    return {
        "R2": r2,
        "Z_Score": z_last,
        "10Y_Return_%": stock_return_pct,
        "10Y_Bench_Return_%": bench_return_pct,
        "Relative_Strength_10Y_%": rel_strength_pct,
        "Annual_Trend_%": annual_trend_pct,
        "_ratio": ratio,
        "_x": x,
        "_slope": slope,
        "_intercept": intercept,
        "_std": std,
    }


def band_for_z(z_val):
    if z_val <= -Z_THRESHOLD:
        return "Bottom"
    if z_val >= Z_THRESHOLD:
        return "Top"
    return "Middle"


def format_cap(mktcap_millions):
    if pd.isna(mktcap_millions):
        return "—"
    if mktcap_millions >= 1000:
        return f"${mktcap_millions / 1000:.1f}B"
    return f"${mktcap_millions:.0f}M"


def build_screen_table(rows, name_map, cap_map, sector_map):
    rows = sorted(rows, key=lambda r: r["R2"], reverse=True)[:TOP_N_PER_TABLE]
    if not rows:
        return rows, '<div class="section-sub">No names passed all filters.</div>'

    cols = ["Ticker", "Name", "Sector", "Industry", "Market Cap", "10Y Return %", "10Y Bench Return %",
            "Relative Strength 10Y %"]
    html = ['<table class="summary screen-table"><thead><tr>']
    html.append("".join(f"<th>{c}</th>" for c in cols))
    html.append("</tr></thead><tbody>")
    for r in rows:
        disp = display_ticker(r["Ticker"])
        name = name_map.get(r["Ticker"], disp)
        sector, industry = sector_map.get(disp, ("—", "—"))
        cap = format_cap(cap_map.get(r["Ticker"]))
        html.append(
            "<tr>"
            f"<td>{disp}</td>"
            f"<td>{name}</td>"
            f"<td>{sector or '—'}</td>"
            f"<td>{industry or '—'}</td>"
            f"<td>{cap}</td>"
            f"<td>{r['10Y_Return_%']:.0f}%</td>"
            f"<td>{r['10Y_Bench_Return_%']:.0f}%</td>"
            f"<td>{r['Relative_Strength_10Y_%']:+.0f}%</td>"
            "</tr>"
        )
    html.append("</tbody></table>")
    return rows, "".join(html)


def build_screen_charts(rows, name_map, revision_map, sector_map):
    parts = []
    for row in rows:
        disp = display_ticker(row["Ticker"])
        company_name = name_map.get(row["Ticker"], disp)
        revision_row = revision_map.get(row["Ticker"])
        sector, industry = sector_map.get(disp, ("", ""))
        sector_industry = " | ".join(s for s in (sector, industry) if s)
        try:
            fig = build_channel_chart(row, company_name, revision_row, sector_industry)
            parts.append(f'<div class="chart-wrap">{fig_to_div(fig)}</div>')
        except Exception as exc:
            print(f"  chart error {row['Ticker']}: {exc}")
    return "\n".join(parts)


def build_bottom_section(all_rows, name_map, cap_map, sector_map, revision_map):
    """Bottom-of-Channel leads the report: all three benchmark tables first
    (SPY, QQQ, XIC), then all three benchmarks' charts follow, each
    delineated with its own 'vs BENCH' header."""
    parts = ['<div class="section"><h2>Bottom of Channel</h2>']
    parts.append(
        '<div class="section-sub">R² &ge; 0.50, positive ratio-regression slope, positive cumulative 10Y '
        f"relative strength, |Z| &ge; {Z_THRESHOLD}. Ranked by R² (cleanest trend first), top 15 shown.</div>"
    )

    kept_by_bench = {}
    for bench_ticker in (BENCH_SPY, BENCH_QQQ, BENCH_XIC):
        bench_label = BENCH_LABELS[bench_ticker]
        rows = [r for r in all_rows if r["Bench"] == bench_label and r["Position"] == "Bottom"]
        kept, table_html = build_screen_table(rows, name_map, cap_map, sector_map)
        kept_by_bench[bench_label] = kept
        parts.append(f'<h3>vs {bench_label}</h3>')
        parts.append(f'<div class="section-sub">{len(kept)} of {len(rows)} qualifying names shown.</div>')
        parts.append(table_html)

    for bench_ticker in (BENCH_SPY, BENCH_QQQ, BENCH_XIC):
        bench_label = BENCH_LABELS[bench_ticker]
        parts.append(f'<h3>vs {bench_label}</h3>')
        parts.append(build_screen_charts(kept_by_bench[bench_label], name_map, revision_map, sector_map))

    parts.append("</div>")
    return "\n".join(parts)


def build_screen_section(position, all_rows, name_map, cap_map, sector_map, revision_map):
    title = f"{position} of Channel"
    position_clause = f"|Z| &lt; {Z_THRESHOLD}"
    parts = [f'<div class="section"><h2>{title}</h2>']
    parts.append(
        '<div class="section-sub">R² &ge; 0.50, positive ratio-regression slope, positive cumulative 10Y '
        f"relative strength, {position_clause}. Ranked by R² (cleanest trend first), top 15 shown.</div>"
    )
    for bench_ticker in (BENCH_SPY, BENCH_QQQ, BENCH_XIC):
        bench_label = BENCH_LABELS[bench_ticker]
        rows = [r for r in all_rows if r["Bench"] == bench_label and r["Position"] == position]
        kept, table_html = build_screen_table(rows, name_map, cap_map, sector_map)
        parts.append(f'<h3>vs {bench_label}</h3>')
        parts.append(f'<div class="section-sub">{len(kept)} of {len(rows)} qualifying names shown.</div>')
        parts.append(table_html)
        parts.append(build_screen_charts(kept, name_map, revision_map, sector_map))
    parts.append("</div>")
    return "\n".join(parts)


# Names that don't appear in the Koyfin sector/industry exports (outside its coverage universe).
# Filled in manually, matching the GICS-style sector/industry naming already used elsewhere in this table.
SUPPLEMENTAL_SECTOR_MAP = {
    "DRS": ("Industrials", "Aerospace and Defense"),
    "LPLA": ("Financials", "Capital Markets"),
    "ENSG": ("Health Care", "Health Care Providers and Services"),
    "GOLF": ("Consumer Discretionary", "Leisure Products"),
    "HLI": ("Financials", "Capital Markets"),
    "AGYS": ("Information Technology", "Software"),
    "KNSL": ("Financials", "Insurance"),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()

    ticker_info, name_map = load_universe()
    revision_map = load_revision_map()
    cap_map = load_cap_map()
    sector_map = load_sector_map()
    sector_map.update(SUPPLEMENTAL_SECTOR_MAP)
    print(f"Universe: {len(ticker_info)} unique tickers")

    bench_closes = {}
    for bench_ticker in (BENCH_SPY, BENCH_QQQ, BENCH_XIC):
        print(f"Downloading {bench_ticker} ({DOWNLOAD_PERIOD})...")
        bench_raw = yf.download(bench_ticker, period=DOWNLOAD_PERIOD, auto_adjust=True, progress=False)
        bench_close = close_series_single(bench_ticker, bench_raw)
        if bench_close is None:
            raise RuntimeError(f"Could not download {bench_ticker} close prices")
        bench_closes[bench_ticker] = bench_close

    all_tickers = sorted(ticker_info.keys())
    print(f"Downloading {len(all_tickers)} tickers ({DOWNLOAD_PERIOD})...")
    close_map = batch_download_closes(all_tickers, DOWNLOAD_PERIOD, CHUNK_SIZE)
    print(f"Got price history for {len(close_map)} tickers")

    screen_rows = []
    for ticker, close in close_map.items():
        try:
            info = ticker_info[ticker]
            bench_ticker = info["bench"]
            bench_label = BENCH_LABELS[bench_ticker]
            close = close.rename(ticker)
            aligned_full = build_aligned(close, bench_closes[bench_ticker])
            if len(aligned_full) < MIN_TRADING_DAYS:
                continue

            fit = fit_ratio_regression(aligned_full.tail(WINDOW_TRADING_DAYS))
            if fit is None:
                continue

            band = band_for_z(fit["Z_Score"])
            row = dict(fit)
            row["Ticker"] = ticker
            row["Bench"] = bench_label
            row["Position"] = band
            screen_rows.append(row)
        except Exception as exc:
            print(f"  error scoring {ticker}: {exc}")

    print(f"Screen: {len(screen_rows)} names currently qualifying (Bottom/Middle/Top)")

    parts = []
    parts.append(build_bottom_section(screen_rows, name_map, cap_map, sector_map, revision_map))
    parts.append(build_screen_section("Middle", screen_rows, name_map, cap_map, sector_map, revision_map))
    parts.append(build_screen_section("Top", screen_rows, name_map, cap_map, sector_map, revision_map))

    html = PAGE_TEMPLATE.format(date_str=today.strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Ratio_Channel_Market_Musings_Study.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Market Musings: Relative-Strength Channel Study</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #1C1C1E; color: #E8E8E8; font-family: Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #3A3A3C; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; }}
  header .meta {{ color: #8E8E93; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #E8E8E8; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section h3 {{ margin: 20px 0 4px; font-size: 15px; color: #C67A29; }}
  .section-sub {{ color: #8E8E93; font-size: 13px; margin: 6px 0 16px; }}
  .callout {{ background: #2A2D3A; border-left: 3px solid #C67A29; padding: 14px 18px; margin: 8px 0 20px;
    font-size: 14px; line-height: 1.5; color: #E8E8E8; border-radius: 0 4px 4px 0; }}
  .chart-wrap {{ padding: 8px 12px; }}
  table.summary {{ border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; margin-bottom: 20px;
    border: 1px solid #2D3148; border-radius: 8px; overflow: hidden; }}
  table.summary th, table.summary td {{ padding: 9px 12px; text-align: center; border-bottom: 1px solid #2D3148;
    border-right: 1px solid #2D3148; }}
  table.summary th:last-child, table.summary td:last-child {{ border-right: none; }}
  table.summary tbody tr:last-child td {{ border-bottom: none; }}
  table.summary th {{ background: #1F79BE; color: #FFFFFF; font-weight: 700; letter-spacing: 0.2px;
    border-right-color: rgba(255,255,255,0.25); }}
  table.summary tbody tr:nth-child(even) {{ background: rgba(31,121,190,0.10); }}
  table.summary tbody tr:hover {{ background: rgba(198,122,41,0.14); }}

  table.screen-table {{ table-layout: fixed; }}
  table.screen-table th, table.screen-table td {{ font-size: 11.5px; padding: 5px 6px; line-height: 1.3;
    word-wrap: break-word; overflow-wrap: break-word; }}
  table.screen-table th:nth-child(1), table.screen-table td:nth-child(1) {{ width: 7%; }}
  table.screen-table th:nth-child(2), table.screen-table td:nth-child(2) {{ width: 19%; text-align: left; }}
  table.screen-table th:nth-child(3), table.screen-table td:nth-child(3) {{ width: 13%; white-space: normal; }}
  table.screen-table th:nth-child(4), table.screen-table td:nth-child(4) {{ width: 19%; white-space: normal; }}
  table.screen-table th:nth-child(5), table.screen-table td:nth-child(5) {{ width: 10%; }}
  table.screen-table th:nth-child(6), table.screen-table td:nth-child(6) {{ width: 10.5%; }}
  table.screen-table th:nth-child(7), table.screen-table td:nth-child(7) {{ width: 10.5%; }}
  table.screen-table th:nth-child(8), table.screen-table td:nth-child(8) {{ width: 11%; }}
</style>
</head>
<body>
<header>
  <h1>Market Musings: Relative-Strength Channel Study</h1>
  <div class="meta">Generated {date_str} &middot; 10y price/benchmark ratio regression (SPY / QQQ / XIC)</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
