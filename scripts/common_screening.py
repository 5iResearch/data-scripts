"""
Shared 6-month relative-high screening logic used by both the Benchmark
Beaters report (generate_benchmark_beaters.py) and the relative-performance
chart pack (generate_relative_performance_charts.py), so both pull from the
exact same universe/ticker definitions.
"""

import io
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSX_UNIVERSE_PATH = os.path.join(REPO_ROOT, "data", "tsx_universe.csv")

INITIAL_WINDOW_DAYS = 180
RELATIVE_WINDOW_DAYS = 180
LAST_DAYS = 5
MIN_TRADING_DAYS = 100  # ~80% of the ~126 trading days in a 180-day window;
                        # filters out halted/thinly-traded/recently-listed tickers
                        # whose sparse price history can make start≈end look "flat" by coincidence


def load_tsx_symbols():
    tsx_list = pd.read_csv(TSX_UNIVERSE_PATH)
    symbols = tsx_list["Symbol"].dropna().astype(str).tolist()
    symbols = [s.replace(".", "-") + ".TO" for s in symbols]
    return symbols


def load_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text))[0]
    return table["Symbol"].tolist()


def close_series(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        return None
    close = data["Close"].dropna()
    if len(close) < MIN_TRADING_DAYS:
        return None
    return close


def find_relative_new_highs(symbols, bench_ticker, end_date):
    start_date = end_date - timedelta(days=INITIAL_WINDOW_DAYS)
    bench_close = close_series(bench_ticker, start_date, end_date)
    if bench_close is None:
        return []

    new_highs = []
    for ticker in symbols:
        try:
            stock_close = close_series(ticker, start_date, end_date)
            if stock_close is None:
                continue
            relative_perf = stock_close.reindex(bench_close.index) / bench_close
            relative_perf = relative_perf.dropna()
            if relative_perf.empty:
                continue
            max_relative = relative_perf.max()
            if any(np.isclose(relative_perf.iloc[-LAST_DAYS:], max_relative)):
                new_highs.append(ticker)
        except Exception as exc:
            print(f"  error screening {ticker}: {exc}")
    return new_highs


def made_relative_high_vs_bench(ticker, bench_close_full, end_date_window):
    start_window = end_date_window - timedelta(days=RELATIVE_WINDOW_DAYS)
    stock_close = close_series(ticker, start_window, end_date_window)
    if stock_close is None:
        return False
    bench_close = bench_close_full.reindex(stock_close.index).dropna()
    combined = pd.DataFrame({"stock": stock_close, "bench": bench_close}).dropna()
    if combined.empty:
        return False
    combined["rel_perf"] = combined["stock"] / combined["bench"]
    max_relative = combined["rel_perf"].max()
    return bool(any(np.isclose(combined["rel_perf"].iloc[-LAST_DAYS:], max_relative, rtol=1e-3)))


def six_month_performance(ticker, end_date_window):
    start_window = end_date_window - timedelta(days=RELATIVE_WINDOW_DAYS)
    close = close_series(ticker, start_window, end_date_window)
    if close is None:
        return np.nan
    close = close.dropna()
    if len(close) < 2:
        return np.nan
    return (close.iloc[-1] / close.iloc[0] - 1) * 100


def six_month_relative_performance(ticker, bench_close_full, end_date_window):
    start_window = end_date_window - timedelta(days=RELATIVE_WINDOW_DAYS)
    stock_close = close_series(ticker, start_window, end_date_window)
    if stock_close is None:
        return np.nan
    stock_close = stock_close.dropna()
    bench_close = bench_close_full[
        (bench_close_full.index >= stock_close.index.min())
        & (bench_close_full.index <= stock_close.index.max())
    ]
    aligned = stock_close.reindex(bench_close.index).dropna()
    bench_aligned = bench_close.reindex(aligned.index).dropna()
    if len(aligned) < 2:
        return np.nan
    stock_ret = aligned.iloc[-1] / aligned.iloc[0] - 1
    bench_ret = bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1
    return (stock_ret - bench_ret) * 100


def screen_market(symbols, bench_ticker, market_label):
    end_date = datetime.now()
    print(f"[{market_label}] screening {len(symbols)} tickers for 6M relative highs vs {bench_ticker}...")
    new_highs = find_relative_new_highs(symbols, bench_ticker, end_date)
    print(f"[{market_label}] {len(new_highs)} tickers made a relative high in the last {LAST_DAYS} days")

    start_date = end_date - timedelta(days=RELATIVE_WINDOW_DAYS)
    bench_close_full = close_series(bench_ticker, start_date, end_date)

    rows = []
    for ticker in new_highs:
        weeks_on_list = 0
        for week_offset in range(1, 5):
            past_end_date = end_date - timedelta(days=7 * week_offset)
            if made_relative_high_vs_bench(ticker, bench_close_full, past_end_date):
                weeks_on_list += 1
            else:
                break
        signal = "New" if weeks_on_list == 0 else ("Repeat" if weeks_on_list <= 2 else "Streak")

        perf_6m = six_month_performance(ticker, end_date)
        rel_6m = six_month_relative_performance(ticker, bench_close_full, end_date)
        if np.isnan(perf_6m) or np.isnan(rel_6m):
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Weeks_on_List": weeks_on_list,
                "Signal": signal,
                "6M_Perf_%": round(perf_6m, 1),
                "6M_Relative_%": round(rel_6m, 1),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        signal_order = {"New": 0, "Repeat": 1, "Streak": 2}
        df["_signal_rank"] = df["Signal"].map(signal_order)
        df = df.sort_values(
            ["_signal_rank", "6M_Perf_%"], ascending=[True, False]
        ).drop(columns="_signal_rank").reset_index(drop=True)
    return df
