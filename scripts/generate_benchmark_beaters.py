"""
Daily "Benchmark Beaters" screener -> filled Excel workbook -> PDF.

Screens the TSX (250M+) universe against XIC.TO and the S&P 500 universe
against SPY for stocks making a 6-month relative high in the last 5 trading
days, tracks how many consecutive weeks each ticker has shown up (New /
Repeat / Streak), enriches the list with company data pulled from a Koyfin
screener export, and writes the results into the print-ready "Cdn Hardcoded"
/ "US Hardcoded" tabs of templates/benchmark_beaters_template.xlsx. The
workbook is then converted to PDF via headless LibreOffice.

The only manual step left is exporting the two Koyfin screener CSVs
(data/koyfin_cdn.csv, data/koyfin_us.csv) before the workflow runs.
"""

import io
import os
import shutil
import subprocess
import sys
from copy import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(REPO_ROOT, "templates", "benchmark_beaters_template.xlsx")
TSX_UNIVERSE_PATH = os.path.join(REPO_ROOT, "data", "tsx_universe.csv")
KOYFIN_CDN_PATH = os.path.join(REPO_ROOT, "data", "koyfin_cdn.csv")
KOYFIN_US_PATH = os.path.join(REPO_ROOT, "data", "koyfin_us.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "benchmark-beaters")

INITIAL_WINDOW_DAYS = 180
RELATIVE_WINDOW_DAYS = 180
LAST_DAYS = 5
MIN_TRADING_DAYS = 100  # ~80% of the ~126 trading days in a 180-day window;
                        # filters out halted/thinly-traded/recently-listed tickers
                        # whose sparse price history can make start≈end look "flat" by coincidence
NAME_SUFFIXES = ["Ltd.", "Inc.", "Corp."]

DATA_START_ROW = 8
DATA_END_COL = 10  # column J (Business Description) - last real data column
TABLE_COLS = list(range(2, DATA_END_COL + 1))  # B..J

CTA_TEXT = (
    "\U0001F513 Want to know which of these we would actually buy? "
    "Start your 14-Day Free Trial at www.5iresearch.ca/bb"
)
DISCLOSURE_TEXT = (
    "Disclosure: While this table does not constitute any formal opinion on names presented, "
    "authors may own shares in non-Canadian securities listed above."
)
ORANGE = "FFC67A29"
ORANGE_FILL = "FFFCEFE0"
LIGHT_GRAY_FILL = "FFF2F2F2"

SIGNAL_BADGE = {
    # (label, ARGB fill) - plain cell formatting, not emoji: emoji fall back to
    # mismatched monochrome glyphs (e.g. a flat blue box) under LibreOffice's
    # headless PDF renderer, which has no color-emoji font installed.
    "New": ("NEW", "FF2E86DE"),
    "Repeat": ("RPT", "FF8E44AD"),
    "Streak": ("STRK", "FFC0392B"),
}

# A fixed row height was too short for the longest wrapped Business
# Description / Company text -> LibreOffice rendered the overflow bleeding
# past the row's colored fill into the row below, which showed up as the
# left/right edge columns looking mismatched. Size each row to what its
# actual (longest) wrapped cell needs instead.
MIN_ROW_HEIGHT = 26
LINE_HEIGHT = 17     # deliberately generous - erring tall is a cosmetic
PADDING = 16          # nit, erring short reproduces the color-bleed bug
DESC_CHARS_PER_LINE = 75    # column J, width 56
COMPANY_CHARS_PER_LINE = 20  # column D, width 26, bold


def _row_height_for(company, description):
    import math
    desc_lines = math.ceil(len(description or "") / DESC_CHARS_PER_LINE) or 1
    company_lines = math.ceil(len(company or "") / COMPANY_CHARS_PER_LINE) or 1
    lines = max(desc_lines, company_lines, 1)
    return max(MIN_ROW_HEIGHT, lines * LINE_HEIGHT + PADDING)


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


def load_koyfin(path, market_label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing Koyfin export for {market_label}: {path}\n"
            "Export the screener from Koyfin as CSV and commit it at that path before running."
        )
    df = pd.read_csv(path)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df.set_index("Ticker")


def short_description(text):
    if not isinstance(text, str) or not text:
        return ""
    normalized = text
    for suffix in NAME_SUFFIXES:
        normalized = normalized.replace(suffix, suffix.rstrip("."))
    return normalized.split(".")[0].strip()


def build_hardcoded_table(screen_df, koyfin_df, strip_suffix=None):
    records = []
    for _, row in screen_df.iterrows():
        raw_ticker = row["Ticker"]
        lookup_ticker = raw_ticker.replace(strip_suffix, "") if strip_suffix else raw_ticker
        if lookup_ticker not in koyfin_df.index:
            print(f"  no Koyfin match for {raw_ticker}, skipping")
            continue
        info = koyfin_df.loc[lookup_ticker]
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        records.append(
            {
                "Ticker": lookup_ticker,
                "Company": info.get("Name", ""),
                "Sector": info.get("Sector", ""),
                "Industry": info.get("Industry", ""),
                "6-Mo Return": round(row["6M_Perf_%"] / 100, 4),
                "Signal": row["Signal"],
                "Mkt Cap ($M)": info.get("Market Cap", ""),
                "Business Description": short_description(info.get("Description", "")),
            }
        )
    return pd.DataFrame(records)


def _copy_row_style(ws, dst_row, src_row):
    for col in TABLE_COLS:
        dst = ws.cell(row=dst_row, column=col)
        dst._style = copy(ws.cell(row=src_row, column=col)._style)


def _set_conditional_formatting_range(ws, last_data_row):
    """The color-scale on the Return column is scoped to a fixed range in the
    template; repoint it at the actual number of data rows each run so it
    never colors (or fails to color) rows that don't exist today."""
    to_replace = [(cf, cf.rules) for cf in list(ws.conditional_formatting) if str(cf.sqref).startswith("G8:G")]
    for cf, rules in to_replace:
        del ws.conditional_formatting._cf_rules[cf]
        for rule in rules:
            ws.conditional_formatting.add(f"G8:G{last_data_row}", rule)


def _write_banner(ws, row, height, text, fill_hex, font_color, bold, italic=False, border_hex=None):
    ws.row_dimensions[row].height = height
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=DATA_END_COL)
    cell = ws.cell(row=row, column=2, value=text)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.font = Font(bold=bold, italic=italic, color=font_color, size=10)
    fill = PatternFill(fill_type="solid", fgColor=fill_hex)
    border = Border(*(Side(style="thin", color=border_hex),) * 4) if border_hex else Border()
    for col in TABLE_COLS:
        c = ws.cell(row=row, column=col)
        c.fill = fill
        c.border = border


def write_hardcoded_sheet(ws, table_df):
    n = len(table_df)

    for offset in range(n):
        row = DATA_START_ROW + offset
        src_row = DATA_START_ROW if offset % 2 == 0 else DATA_START_ROW + 1
        if row != src_row:
            _copy_row_style(ws, row, src_row)
        record = table_df.iloc[offset]
        ws.row_dimensions[row].height = _row_height_for(record["Company"], record["Business Description"])
        label, badge_fill = SIGNAL_BADGE.get(record["Signal"], ("", None))
        badge_cell = ws.cell(row=row, column=2, value=label)  # B: signal badge
        if badge_fill:
            badge_cell.fill = PatternFill(fill_type="solid", fgColor=badge_fill)
            badge_cell.font = Font(bold=True, color="FFFFFFFF", size=7)
            badge_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row, column=3, value=record["Ticker"])
        ws.cell(row=row, column=4, value=record["Company"])
        ws.cell(row=row, column=5, value=record["Sector"])
        ws.cell(row=row, column=6, value=record["Industry"])
        ws.cell(row=row, column=7, value=record["6-Mo Return"])
        ws.cell(row=row, column=8, value=record["Signal"])
        ws.cell(row=row, column=9, value=record["Mkt Cap ($M)"])
        ws.cell(row=row, column=10, value=record["Business Description"])

    if n == 0:
        last_data_row = DATA_START_ROW - 1
    else:
        last_data_row = DATA_START_ROW + n - 1
        # heavier bottom border under the last row, matching the template's
        # original "end of table" styling
        for col in TABLE_COLS:
            c = ws.cell(row=last_data_row, column=col)
            b = c.border
            c.border = Border(left=b.left, right=b.right, top=b.top, bottom=Side(style="medium"))

    _set_conditional_formatting_range(ws, max(last_data_row, DATA_START_ROW))

    # The template has pre-baked banding/fill for a wide fixed row range (a
    # leftover of the original hand-built file). Neutralize the spacer row
    # between the table and the CTA banner so no stray colored cell (e.g. the
    # Return column's base green fill) shows through from that inherited style.
    spacer_row = last_data_row + 1
    for col in TABLE_COLS:
        c = ws.cell(row=spacer_row, column=col)
        c.value = None
        c.fill = PatternFill(fill_type=None)
        c.border = Border()
        c.number_format = "General"
    ws.row_dimensions[spacer_row].height = 8

    cta_row = last_data_row + 2
    _write_banner(
        ws, cta_row, height=34, text=CTA_TEXT,
        fill_hex=ORANGE_FILL, font_color=ORANGE, bold=True, border_hex=ORANGE,
    )
    disclosure_row = cta_row + 1
    _write_banner(
        ws, disclosure_row, height=28, text=DISCLOSURE_TEXT,
        fill_hex=LIGHT_GRAY_FILL, font_color="FF666666", bold=False, italic=True,
    )

    ws.print_area = f"A1:K{disclosure_row}"


PRINT_SHEETS = ["Cdn Hardcoded", "US Hardcoded"]


def make_print_ready_copy(source_xlsx_path, dest_xlsx_path):
    """PDF export should only contain the two print-ready tabs, not the
    Instructions/Date/legacy manual-workflow tabs. LibreOffice's --convert-to
    exports every sheet in the workbook, so build a stripped-down copy first."""
    wb = load_workbook(source_xlsx_path)
    for sheet_name in list(wb.sheetnames):
        if sheet_name not in PRINT_SHEETS:
            del wb[sheet_name]
    wb.save(dest_xlsx_path)


def convert_to_pdf(xlsx_path, out_dir):
    subprocess.run(
        [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            out_dir,
            xlsx_path,
        ],
        check=True,
        timeout=300,
    )
    pdf_path = os.path.join(out_dir, os.path.splitext(os.path.basename(xlsx_path))[0] + ".pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"LibreOffice did not produce expected PDF at {pdf_path}")
    return pdf_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")

    tsx_symbols = load_tsx_symbols()
    cdn_screen = screen_market(tsx_symbols, "XIC.TO", "Cdn")

    sp500_symbols = load_sp500_symbols()
    us_screen = screen_market(sp500_symbols, "SPY", "US")

    koyfin_cdn = load_koyfin(KOYFIN_CDN_PATH, "Cdn")
    koyfin_us = load_koyfin(KOYFIN_US_PATH, "US")

    cdn_table = build_hardcoded_table(cdn_screen, koyfin_cdn, strip_suffix=".TO")
    us_table = build_hardcoded_table(us_screen, koyfin_us)

    wb = load_workbook(TEMPLATE_PATH)
    wb["Date"]["B2"] = today
    wb["Cdn Hardcoded"]["I2"] = today
    wb["US Hardcoded"]["I2"] = today
    write_hardcoded_sheet(wb["Cdn Hardcoded"], cdn_table)
    write_hardcoded_sheet(wb["US Hardcoded"], us_table)

    xlsx_out = os.path.join(OUTPUT_DIR, f"Benchmark_Beaters_{date_str}.xlsx")
    wb.save(xlsx_out)
    print(f"Saved workbook: {xlsx_out}")

    print_ready_path = os.path.join(OUTPUT_DIR, f"_print_ready_{date_str}.xlsx")
    make_print_ready_copy(xlsx_out, print_ready_path)
    pdf_out = convert_to_pdf(print_ready_path, OUTPUT_DIR)
    final_pdf = os.path.join(OUTPUT_DIR, f"Benchmark_Beaters_{date_str}.pdf")
    os.replace(pdf_out, final_pdf)
    os.remove(print_ready_path)
    print(f"Saved PDF: {final_pdf}")

    latest_xlsx = os.path.join(OUTPUT_DIR, "Benchmark_Beaters_latest.xlsx")
    wb.save(latest_xlsx)
    shutil.copyfile(final_pdf, os.path.join(OUTPUT_DIR, "Benchmark_Beaters_latest.pdf"))


if __name__ == "__main__":
    main()
