"""
Daily relative-performance chart pack, adapted from the "Benchmark Beaters"
notebook's chart cells (6-month and 10-year relative performance vs XIC.TO /
SPY, for both Canadian and US screened tickers).

Screens the same universes as generate_benchmark_beaters.py (via
common_screening.py) for 6-month relative highs, then charts each screened
ticker's relative performance over both a 6-month and a 10-year window,
combining every chart into a single dated PDF:
  Canadian 6-Month -> US 6-Month -> Canadian 10-Year -> US 10-Year

This script is self-contained: it does not depend on the Koyfin CSV exports
or on generate_benchmark_beaters.py having run first, so it can run on its
own schedule.
"""

import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_screening import load_sp500_symbols, load_tsx_symbols, screen_market

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "relative-performance-charts")

DARK_BG = "#1C1C1E"
PANEL_BG = "#2C2C2E"
GRID = "#3A3A3C"
ORANGE = "#C67A29"
GREEN = "#2ECC71"
RED = "#E74C3C"
TEXT = "#E5E5EA"
SUBTEXT = "#8E8E93"
GREEN_FILL = "#1a3d2b"
RED_FILL = "#3d1a1a"

SIX_MONTH_DAYS = 180
TEN_YEAR_DAYS = 10 * 365
SIX_MONTH_MIN_POINTS = 10
TEN_YEAR_MIN_POINTS = 20


def batch_close(tickers, bench, start, end):
    all_tickers = tickers + [bench]
    if len(all_tickers) == 1:
        raw = yf.download(all_tickers, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close_map = {all_tickers[0]: raw["Close"]}
    else:
        raw = yf.download(
            all_tickers, start=start, end=end, group_by="ticker", auto_adjust=False, progress=False
        )
        close_map = {}
        for t in tickers:
            try:
                close_map[t] = raw[t]["Close"]
            except (KeyError, TypeError):
                pass
    try:
        bench_close = raw[bench]["Close"].ffill().dropna()
    except (KeyError, TypeError):
        bench_close = close_map.get(bench, pd.Series(dtype=float))
    return close_map, bench_close


def compute_relative(stock_close, bench_close, min_points):
    sc = pd.to_numeric(stock_close, errors="coerce").dropna()
    bc = bench_close.reindex(sc.index).ffill().dropna()
    idx = sc.index.intersection(bc.index)
    sc, bc = sc.loc[idx], bc.loc[idx]
    if len(sc) < min_points:
        return None
    r0 = float(sc.iloc[0]) / float(bc.iloc[0])
    if r0 == 0:
        return None
    return (sc / bc) / r0 * 100.0


def draw_chart(pdf, ticker, company_name, rel, bench_label, period_label, as_of_str):
    ret = float(rel.iloc[-1]) - 100.0
    sign = "+" if ret >= 0 else ""
    line_color = GREEN if ret >= 0 else RED
    badge_bg = GREEN_FILL if ret >= 0 else RED_FILL

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=SUBTEXT, length=0, labelsize=8)

    fig.text(0.05, 0.93, f"{ticker}  ·  {company_name}", fontsize=13, fontweight="bold", color=TEXT, va="bottom")
    fig.text(
        0.05, 0.89,
        f"{period_label} Relative Performance vs {bench_label}  ·  As of {as_of_str}",
        fontsize=8, color=SUBTEXT, va="top",
    )

    ax.axhline(100, color=SUBTEXT, lw=1.0, linestyle="--", alpha=0.5, zorder=1)
    rv = rel.values.astype(float)
    ax.fill_between(rel.index, rv, 100, where=(rv >= 100), color=GREEN_FILL, alpha=0.45, zorder=1)
    ax.fill_between(rel.index, rv, 100, where=(rv < 100), color=RED_FILL, alpha=0.45, zorder=1)
    ax.plot(rel.index, rv, color=ORANGE, lw=1.8, zorder=3, solid_capstyle="round")

    idx_max = rel.idxmax()
    ax.scatter(idx_max, float(rel[idx_max]), color=ORANGE, s=45, zorder=5, edgecolors="none")
    ax.annotate(
        f"High\n{float(rel[idx_max]):.1f}",
        xy=(idx_max, float(rel[idx_max])), xytext=(0, 12), textcoords="offset points",
        ha="center", va="bottom", fontsize=7, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.3", fc=GRID, ec=ORANGE, lw=0.7, alpha=0.9),
    )

    ax.text(
        0.985, 0.90, f"{sign}{ret:.1f}% vs {bench_label}",
        transform=ax.transAxes, fontsize=12, fontweight="bold", color=line_color,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc=badge_bg, ec=line_color, lw=1.0, alpha=0.95),
    )

    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    if period_label.startswith("6"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    plt.setp(ax.get_xticklabels(), color=SUBTEXT, fontsize=8)
    plt.setp(ax.get_yticklabels(), color=SUBTEXT, fontsize=8)
    ax.set_ylabel("Relative Performance (100 = Benchmark)", color=SUBTEXT, fontsize=7.5, labelpad=6)
    fig.text(0.97, 0.03, "Source: Yahoo Finance", ha="right", va="bottom", fontsize=7, color=SUBTEXT)

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.86])
    pdf.savefig(fig, facecolor=DARK_BG)
    plt.close(fig)


def draw_section_cover(pdf, title, subtitle):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    ax.text(0.5, 0.6, title, fontsize=26, fontweight="bold", color=TEXT, ha="center", va="center")
    ax.text(0.5, 0.42, subtitle, fontsize=12, color=ORANGE, ha="center", va="center")
    pdf.savefig(fig, facecolor=DARK_BG)
    plt.close(fig)


def build_section(pdf, tickers, bench, bench_label, days, period_label, min_points, market_label):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    as_of_str = end_date.strftime("%B %d, %Y")

    if not tickers:
        print(f"[{market_label} {period_label}] no tickers to chart")
        return

    print(f"[{market_label} {period_label}] batch downloading {len(tickers)} tickers...")
    close_map, bench_close = batch_close(tickers, bench, start_date, end_date)

    scored = []
    for ticker in list(close_map.keys()):
        rel = compute_relative(close_map[ticker], bench_close, min_points)
        if rel is None:
            continue
        scored.append((ticker, rel, float(rel.iloc[-1]) - 100.0))
    scored.sort(key=lambda x: x[2], reverse=True)
    print(f"[{market_label} {period_label}] charting {len(scored)} tickers (sorted by outperformance)")

    for ticker, rel, _ in scored:
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            name = ticker
        draw_chart(pdf, ticker, name, rel, bench_label, period_label, as_of_str)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")

    tsx_symbols = load_tsx_symbols()
    cdn_screen = screen_market(tsx_symbols, "XIC.TO", "Cdn")
    cdn_tickers = cdn_screen["Ticker"].tolist() if not cdn_screen.empty else []

    sp500_symbols = load_sp500_symbols()
    us_screen = screen_market(sp500_symbols, "SPY", "US")
    us_tickers = us_screen["Ticker"].tolist() if not us_screen.empty else []

    pdf_path = os.path.join(OUTPUT_DIR, f"Relative_Performance_Charts_{date_str}.pdf")
    with PdfPages(pdf_path) as pdf:
        draw_section_cover(pdf, "Benchmark Beaters", f"Relative Performance Charts  ·  {today.strftime('%B %d, %Y')}")

        draw_section_cover(pdf, "Canadian Stocks", "6-Month Relative Performance vs XIC.TO")
        build_section(pdf, cdn_tickers, "XIC.TO", "XIC.TO", SIX_MONTH_DAYS, "6-Month", SIX_MONTH_MIN_POINTS, "Cdn")

        draw_section_cover(pdf, "U.S. Stocks", "6-Month Relative Performance vs SPY")
        build_section(pdf, us_tickers, "SPY", "SPY", SIX_MONTH_DAYS, "6-Month", SIX_MONTH_MIN_POINTS, "US")

        draw_section_cover(pdf, "Canadian Stocks", "10-Year Relative Performance vs XIC.TO")
        build_section(pdf, cdn_tickers, "XIC.TO", "XIC.TO", TEN_YEAR_DAYS, "10-Year", TEN_YEAR_MIN_POINTS, "Cdn")

        draw_section_cover(pdf, "U.S. Stocks", "10-Year Relative Performance vs SPY")
        build_section(pdf, us_tickers, "SPY", "SPY", TEN_YEAR_DAYS, "10-Year", TEN_YEAR_MIN_POINTS, "US")

    print(f"Saved: {pdf_path}")

    latest_pdf = os.path.join(OUTPUT_DIR, "Relative_Performance_Charts_latest.pdf")
    shutil.copyfile(pdf_path, latest_pdf)
    print(f"Saved: {latest_pdf}")


if __name__ == "__main__":
    main()
