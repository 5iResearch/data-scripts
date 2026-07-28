"""
Daily "Revenue Revision Benchmark Beaters" report, adapted from the
"us_benchmark_beaters_rev_screener" notebook.

Two-stage screen: (1) stock at a fresh 6-month relative high vs its
benchmark, (2) analyst revenue estimate revisions confirming the move.
Reproduces only the ranked top-40 table and top-40 spotlight chart series
for each of the notebook's three sections (not the matplotlib overview /
cascade-profile / heatmap charts):
  1. US S&P 500 vs SPY
  2. All-US (~2,000-name CSV universe) vs SPY
  3. Canada vs XIC.TO

Depends on two manually-refreshed CSVs (revenue estimate revision exports,
same pattern as the Koyfin CSVs used by generate_benchmark_beaters.py):
  data/us_1w_rev_est_screener.csv
  data/cdn_1w_rev_est_screener.csv
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "rev-revision-benchmark-beaters")
US_CSV_PATH = os.path.join(REPO_ROOT, "data", "us_1w_rev_est_screener.csv")
CDN_CSV_PATH = os.path.join(REPO_ROOT, "data", "cdn_1w_rev_est_screener.csv")

ORANGE, BLUE, DBLUE, GREEN, RED = "#C67A29", "#1F79BE", "#4B8EA9", "#44A660", "#A22A2A"
BLUE2, WATCH_COLOR = "#297ABC", "#8b90a4"
BG, PANEL, GRID, TEXT, SUBTEXT = "#0f1117", "#1a1d27", "#2d3148", "#e2e5f0", "#8b90a4"

TIER_COLORS = {
    "Structural Cascade": ORANGE, "Full Cascade": BLUE, "Strong": DBLUE,
    "Partial": BLUE2, "Weak": WATCH_COLOR,
}
STREAK_COLORS = {"Streak": ORANGE, "Repeat": BLUE, "New": WATCH_COLOR}

TODAY = datetime.today().strftime("%Y-%m-%d")
END = datetime.today()
PRICE_DAYS = 240
START = END - timedelta(days=PRICE_DAYS)
SPY_6M_START = END - timedelta(days=180)

BENCH = "SPY"
NEW_HIGH_DAYS = 5
CHUNK_SIZE = 200
SPOTLIGHT_TOP_N = 40

COL_MAP = {
    "Ticker": "ticker", "Name": "name",
    "Last Price": "price", "Market Cap": "mktcap",
    "Trading Region": "region", "Trading Country": "country",
    "Below 52W High %": "below_52wh",
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

WINDOWS = ["1m", "3m", "6m", "1y"]
WINDOW_X = [1, 3, 6, 12]
FY_PERIODS = ["fy1", "fy2", "fy3"]
ALL_WIN_KEYS = ["1w", "1m", "3m", "6m", "1y"]
ALL_WIN_LABELS = ["1W", "1M", "3M", "6M", "1Y"]
FY_COLORS_P = [BLUE, DBLUE, ORANGE]

DISPLAY_COLS = [
    ("rank", "Rank"), ("ticker", "Ticker"), ("name", "Name"), ("tier", "Tier"), ("signal", "Signal"),
    ("cascade_score", "Cascade (0-9)"), ("avg_1w", "Avg 1W Rev%"),
    ("fy1_1w", "FY1E 1W"), ("fy2_1w", "FY2E 1W"), ("fy3_1w", "FY3E 1W"),
    ("rel_6m", "6M vs Bench"), ("ret_6m", "6M Return"), ("weeks_streak", "Streak Wks"),
    ("avg_magnitude_1y", "Avg 1Y Rev%"), ("all_fy_pos_1w", "All FY 1W+"),
]


def load_rev_csv(path):
    raw = pd.read_csv(path)
    df = raw.rename(columns=COL_MAP)
    df = df[[c for c in COL_MAP.values() if c in df.columns]].copy()
    for c in [c for c in df.columns if c.startswith("fy")] + ["price", "mktcap", "below_52wh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fy1_1w", "fy2_1w"], how="all").reset_index(drop=True)
    return df


def score_row(row):
    res = {}
    cascade_total = 0
    slope_r2s, magnitude_1y, momentum_1w = [], [], []
    all_fy_pos_1w, all_fy_pos_1m = [], []

    for fy in FY_PERIODS:
        v1w = row.get(f"{fy}_1w", np.nan)
        vals = [row.get(f"{fy}_{w}", np.nan) for w in WINDOWS]
        v1m, v3m, v6m, v1y = vals

        c1 = int(pd.notna(v1y) and pd.notna(v6m) and v1y > v6m)
        c2 = int(pd.notna(v6m) and pd.notna(v3m) and v6m > v3m)
        c3 = int(pd.notna(v3m) and pd.notna(v1m) and v3m > v1m)
        cascade_fy = c1 + c2 + c3
        cascade_total += cascade_fy
        res[f"{fy}_cascade"] = cascade_fy

        valid = [(x, v) for x, v in zip(WINDOW_X, vals) if pd.notna(v)]
        if len(valid) >= 3:
            xs, ys = zip(*valid)
            slope, _, r, _, _ = sp_stats.linregress(xs, ys)
            slope_r2s.append(r ** 2)
        if pd.notna(v1y): magnitude_1y.append(v1y)
        if pd.notna(v1w): momentum_1w.append(v1w)
        all_fy_pos_1w.append(pd.notna(v1w) and v1w > 0)
        all_fy_pos_1m.append(pd.notna(v1m) and v1m > 0)

    res["cascade_score"] = cascade_total
    res["avg_magnitude_1y"] = np.nanmean(magnitude_1y) if magnitude_1y else np.nan
    res["avg_1w"] = np.nanmean(momentum_1w) if momentum_1w else np.nan
    res["all_fy_pos_1w"] = all(all_fy_pos_1w)
    res["all_fy_pos_1m"] = all(all_fy_pos_1m)
    return pd.Series(res)


def winsorize_zscore(series, pct=2):
    s = series.dropna()
    if len(s) < 5:
        return series * 0
    lo, hi = np.nanpercentile(s, pct), np.nanpercentile(s, 100 - pct)
    clipped = series.clip(lo, hi)
    mean, std = np.nanmean(clipped), np.nanstd(clipped)
    return (clipped - mean) / std if std > 0 else clipped * 0


def assign_tier(row):
    cs = row["cascade_score"]
    mag = row["avg_magnitude_1y"]
    if cs == 9 and pd.notna(mag) and mag > 0.10:
        return "Structural Cascade"
    elif cs == 9:
        return "Full Cascade"
    elif cs >= 7:
        return "Strong"
    elif cs >= 5:
        return "Partial"
    return "Weak"


def download_prices(tickers, chunk_size=CHUNK_SIZE, label=""):
    price_data = {}
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for idx, chunk in enumerate(chunks):
        print(f"  {label} price chunk {idx + 1}/{len(chunks)} ({len(chunk)} tickers)...")
        try:
            raw = yf.download(chunk, start=START.strftime("%Y-%m-%d"), end=END.strftime("%Y-%m-%d"),
                               auto_adjust=True, progress=False, threads=True)
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
            for tkr in close.columns:
                s = close[tkr].dropna()
                if len(s) >= 20:
                    price_data[tkr] = s
        except Exception as e:
            print(f"    Chunk {idx + 1} error: {e}")
    return price_data


def screen_relative_highs(price_data, bench_6m, bench_start, ticker_map=None):
    rel_map = {}
    records = []
    for raw_tkr, price_s in price_data.items():
        out_tkr = ticker_map.get(raw_tkr, raw_tkr) if ticker_map else raw_tkr
        try:
            s6m = price_s[price_s.index >= bench_start]
            bch = bench_6m.reindex(s6m.index).ffill().dropna()
            s6m = s6m.reindex(bch.index).dropna()
            if len(s6m) < 20:
                continue
            r0 = float(s6m.iloc[0]) / float(bch.iloc[0])
            if r0 == 0:
                continue
            rel = (s6m / bch) / r0 * 100.0

            max_rel = rel.max()
            if not any(np.isclose(rel.iloc[-NEW_HIGH_DAYS:], max_rel, rtol=1e-3)):
                continue

            streak = 0
            for w in range(1, 5):
                cutoff = rel.index[-1] - timedelta(days=7 * w)
                past = rel[rel.index <= cutoff]
                if len(past) < 10:
                    break
                if any(np.isclose(past.iloc[-NEW_HIGH_DAYS:], past.max(), rtol=1e-3)):
                    streak += 1
                else:
                    break

            signal = "New" if streak == 0 else ("Repeat" if streak <= 2 else "Streak")
            ret_6m = float(s6m.iloc[-1] / s6m.iloc[0] - 1) * 100
            bench_ret_6m = float(bch.iloc[-1] / bch.iloc[0] - 1) * 100
            rel_6m = ret_6m - bench_ret_6m

            rel_map[out_tkr] = rel
            records.append({"ticker": out_tkr, "ret_6m": round(ret_6m, 2), "rel_6m": round(rel_6m, 2),
                             "weeks_streak": streak, "signal": signal})
        except Exception:
            pass
    return pd.DataFrame(records), rel_map


def join_score_rank(price_df, rev_df, extra_gate=None):
    joined = price_df.merge(rev_df, on="ticker", how="inner")
    joined = joined.dropna(subset=["fy1_1w"]).reset_index(drop=True)
    joined = joined[(joined["fy1_1w"] > 0) & (joined["fy2_1w"] > 0)]
    if extra_gate is not None:
        joined = joined[extra_gate(joined)]
    joined = joined.reset_index(drop=True)
    if joined.empty:
        return joined

    s = joined.copy()
    s["z_1w"] = winsorize_zscore(s["avg_1w"])
    s["z_cascade"] = winsorize_zscore(s["cascade_score"])
    s["z_rel_6m"] = winsorize_zscore(s["rel_6m"])
    s["z_streak"] = winsorize_zscore(s["weeks_streak"].astype(float))
    s["combined"] = s["z_1w"] * 0.40 + s["z_cascade"] * 0.30 + s["z_rel_6m"] * 0.20 + s["z_streak"] * 0.10
    s["tier"] = s.apply(assign_tier, axis=1)
    ranked = s.sort_values("combined", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def pct(val, d=2):
    return f"{val * 100:+.{d}f}%" if pd.notna(val) else "—"


def rel_pct(val, d=1):
    return f"{val:+.{d}f}%" if pd.notna(val) else "—"


def style_tier(val):
    return f"color: {TIER_COLORS.get(val, TEXT)}; font-weight: bold"


def style_signal(val):
    return f"color: {STREAK_COLORS.get(val, TEXT)}; font-weight: bold"


def build_table_html(ranked, bench_label, caption_label):
    cols = [(k, v.replace("Bench", bench_label)) for k, v in DISPLAY_COLS]
    top40 = ranked.head(40)[[c for c, _ in cols]].rename(columns=dict(cols)).copy()

    for col in ["Avg 1W Rev%", "FY1E 1W", "FY2E 1W", "FY3E 1W", "Avg 1Y Rev%"]:
        top40[col] = top40[col].apply(pct)
    for col in [f"6M vs {bench_label}", "6M Return"]:
        top40[col] = top40[col].apply(rel_pct)
    top40["Cascade (0-9)"] = top40["Cascade (0-9)"].apply(lambda v: f"{int(v)}/9" if pd.notna(v) else "—")
    top40["Streak Wks"] = top40["Streak Wks"].apply(lambda v: str(int(v)) if pd.notna(v) else "—")
    top40["All FY 1W+"] = top40["All FY 1W+"].map({True: "✓", False: ""})

    styler = (
        top40.style
        .map(style_tier, subset=["Tier"])
        .map(style_signal, subset=["Signal"])
        .set_properties(**{"background-color": PANEL, "color": TEXT, "border": f"1px solid {GRID}"})
        .set_table_styles([{"selector": "th", "props": [
            ("background-color", "#252840"), ("color", "#e2e5f0"),
            ("font-weight", "bold"), ("border", f"1px solid {GRID}")]}])
        .set_caption(f"Top 40 — {caption_label} | {TODAY} | "
                     "1W rev (40%) + cascade (30%) + 6M vs bench (20%) + streak (10%)")
        .hide(axis="index")
    )
    return styler.to_html()


def make_spotlight(row, rel_series, bench_label="SPY"):
    ticker, name, tier = row["ticker"], row["name"], row["tier"]
    tc = TIER_COLORS.get(tier, WATCH_COLOR)
    signal, rank, cs = row["signal"], int(row["rank"]), int(row["cascade_score"])
    avg1w, rel6m, streak = row["avg_1w"], row["rel_6m"], int(row["weeks_streak"])
    w1_str = f"{avg1w * 100:+.2f}%" if pd.notna(avg1w) else "—"
    r6_str = f"{rel6m:+.1f}%" if pd.notna(rel6m) else "—"
    perf_title = f"6M Relative Performance vs {bench_label}"

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                         subplot_titles=[perf_title, "Revenue Revision Cascade"], horizontal_spacing=0.10)

    if rel_series is not None and len(rel_series) > 5:
        rv = rel_series.values.astype(float)
        net_ret = float(rv[-1]) - 100.0
        line_c = GREEN if net_ret >= 0 else RED
        h = (GREEN if net_ret >= 0 else RED).lstrip("#")
        fill_c = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.12)"

        fig.add_trace(go.Scatter(x=list(rel_series.index), y=[100.0] * len(rel_series), mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(rel_series.index), y=rv.tolist(), mode="lines",
            line=dict(color=line_c, width=2.2), fill="tonexty", fillcolor=fill_c, showlegend=False,
            hovertemplate="%{x|%b %d %Y}<br>Rel: %{y:.1f}<extra></extra>"), row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color=SUBTEXT, line_width=1, row=1, col=1)
        fig.add_annotation(x=0.02, y=0.96, xref="x domain", yref="y domain",
            text=f"6M vs {bench_label}: <b>{net_ret:+.1f}%</b>", showarrow=False,
            font=dict(size=13, color=line_c), bgcolor=BG, bordercolor=line_c, borderwidth=1)
    else:
        fig.add_annotation(x=0.3, y=0.5, xref="paper", yref="paper", text="No price data",
                            showarrow=False, font=dict(size=12, color=WATCH_COLOR))

    for fy_idx, fy in enumerate(["fy1", "fy2", "fy3"]):
        vals = [row.get(f"{fy}_{w}", np.nan) * 100 for w in ALL_WIN_KEYS]
        fig.add_trace(go.Bar(x=ALL_WIN_LABELS, y=vals, name=f"FY{fy_idx + 1}E", marker_color=FY_COLORS_P[fy_idx],
            opacity=0.85, hovertemplate="%{x}: %{y:+.2f}%<extra>FY" + str(fy_idx + 1) + "E</extra>"), row=1, col=2)

    streak_label = {"Streak": "Streak", "Repeat": "Repeat", "New": "New"}.get(signal, signal)
    subtitle = (f"Rank #{rank} | {tier} | {streak_label} ({streak}W) | "
                f"6M vs {bench_label}: {r6_str} | Cascade {cs}/9 | Avg 1W rev: {w1_str}")

    fig.update_layout(
        title=dict(text=(f'<b><span style="font-size:26px;color:{tc}">{ticker}</span>'
                         f'<span style="font-size:18px;color:#e2e5f0">  {name}</span></b><br>'
                         f'<span style="font-size:12px;color:#8b90a4">{subtitle}</span>'),
                   x=0.0, xanchor="left", y=0.97, yanchor="top", pad=dict(l=10, t=10)),
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT, family="DejaVu Sans, Arial"),
        height=460, margin=dict(l=60, r=60, t=120, b=60), barmode="group", showlegend=True,
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1, x=0.57, y=0.25, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID)
    fig.update_yaxes(title_text=f"Relative Performance (100 = {bench_label})", row=1, col=1)
    fig.update_yaxes(title_text="Revenue Revision %", ticksuffix="%", row=1, col=2)
    for ann in fig["layout"]["annotations"]:
        if ann.text in (perf_title, "Revenue Revision Cascade"):
            ann.font.size = 12
            ann.font.color = "#e2e5f0"
    return fig


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"responsive": True})


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section"><h2>{title}</h2>{sub}</div>'


def build_section(ranked, bench_label, table_caption, section_title, section_sub):
    parts = [section_header(section_title, section_sub)]
    if ranked.empty:
        parts.append('<div class="section" style="color:#8b90a4;">No names passed the screen today.</div>')
        return parts
    parts.append(f'<div class="table-wrap">{build_table_html(ranked, bench_label, table_caption)}</div>')

    rel_map = ranked.attrs.get("rel_map", {})
    for _, row in ranked.head(SPOTLIGHT_TOP_N).iterrows():
        rser = rel_map.get(row["ticker"])
        fig = make_spotlight(row, rser, bench_label=bench_label)
        parts.append(fig_to_div(fig))
    return parts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parts = []

    # ── Section 1: US S&P 500 ────────────────────────────────────────────────
    print("=== US S&P 500 ===")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    us_symbols = pd.read_html(resp.text)[0]["Symbol"].str.replace(".", "-", regex=False).tolist()

    spy_all = yf.download(BENCH, start=START.strftime("%Y-%m-%d"), end=END.strftime("%Y-%m-%d"),
                           auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    spy_6m = spy_all[spy_all.index >= SPY_6M_START.strftime("%Y-%m-%d")]

    rev_df = load_rev_csv(US_CSV_PATH)
    rev_scores = rev_df.apply(score_row, axis=1)
    rev_df = pd.concat([rev_df, rev_scores], axis=1)

    us_price_data = download_prices(us_symbols, label="S&P500")
    us_price_df, us_rel_map = screen_relative_highs(us_price_data, spy_6m, SPY_6M_START.strftime("%Y-%m-%d"))
    us_ranked = join_score_rank(us_price_df, rev_df)
    us_ranked.attrs["rel_map"] = us_rel_map
    parts += build_section(us_ranked, "SPY", "US S&P 500 Benchmark Beaters with Revenue Revision Momentum",
                            "Section 1 - US S&P 500", "6M relative high vs SPY, gated on FY1E+FY2E 1W revisions positive")

    # ── Section 2: All-US ~2,000 names ───────────────────────────────────────
    print("=== All-US ~2,000 names ===")
    us_all_tickers = rev_df["ticker"].dropna().unique().tolist()
    us_all_price_data = download_prices(us_all_tickers, label="All-US")
    us_all_price_df, us_all_rel_map = screen_relative_highs(us_all_price_data, spy_6m, SPY_6M_START.strftime("%Y-%m-%d"))
    us_all_ranked = join_score_rank(us_all_price_df, rev_df)
    us_all_ranked.attrs["rel_map"] = us_all_rel_map
    parts += build_section(us_all_ranked, "SPY", "All-US Benchmark Beaters (Full CSV ~2k) with Revenue Revision Momentum",
                            "Section 2 - All-US (~2,000-Name CSV Universe)", "Same screen, full revision-CSV universe rather than just S&P 500")

    # ── Section 3: Canada vs XIC.TO ──────────────────────────────────────────
    print("=== Canada vs XIC.TO ===")
    CDN_BENCH = "XIC.TO"
    CDN_6M_START = END - timedelta(days=180)

    cdn_rev_df = load_rev_csv(CDN_CSV_PATH)
    cdn_scores = cdn_rev_df.apply(score_row, axis=1)
    cdn_rev_df = pd.concat([cdn_rev_df, cdn_scores], axis=1)

    xic_all = yf.download(CDN_BENCH, start=START.strftime("%Y-%m-%d"), end=END.strftime("%Y-%m-%d"),
                           auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    xic_6m = xic_all[xic_all.index >= CDN_6M_START.strftime("%Y-%m-%d")]

    cdn_csv_tickers = cdn_rev_df["ticker"].dropna().unique().tolist()
    cdn_yf_tickers = [t.replace(".TO", "").replace(".to", "") + ".TO" for t in cdn_csv_tickers]
    cdn_ticker_map = {yf_t: csv_t for yf_t, csv_t in zip(cdn_yf_tickers, cdn_csv_tickers)}

    cdn_price_data = download_prices(cdn_yf_tickers, label="CDN")
    cdn_price_df, cdn_rel_map = screen_relative_highs(cdn_price_data, xic_6m, CDN_6M_START.strftime("%Y-%m-%d"),
                                                        ticker_map=cdn_ticker_map)

    def cdn_gate(df):
        return (df["fy1_1m"] > 0) & (df["fy2_1m"] > 0)

    cdn_ranked = join_score_rank(cdn_price_df, cdn_rev_df, extra_gate=cdn_gate)
    cdn_ranked.attrs["rel_map"] = cdn_rel_map
    parts += build_section(cdn_ranked, "XIC.TO", "CDN Benchmark Beaters vs XIC.TO with Revenue Revision Momentum",
                            "Section 3 - Canada vs XIC.TO",
                            "6M relative high vs XIC.TO, gated on FY1E+FY2E 1W AND 1M revisions positive")

    html = PAGE_TEMPLATE.format(date_str=datetime.now().strftime("%B %d, %Y"), body="\n".join(parts))
    out_path = os.path.join(OUTPUT_DIR, "Rev_Revision_Benchmark_Beaters.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Revenue Revision Benchmark Beaters</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ background: #0f1117; color: #c8ccd8; font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0 0 40px; }}
  header {{ padding: 24px 32px 16px; border-bottom: 1px solid #2d3148; }}
  header h1 {{ margin: 0 0 4px; font-size: 24px; color: #e2e5f0; }}
  header .meta {{ color: #8b90a4; font-size: 13px; }}
  .section {{ padding: 28px 32px 4px; }}
  .section h2 {{ margin: 0; font-size: 20px; color: #e2e5f0; border-bottom: 2px solid #C67A29; display: inline-block; padding-bottom: 4px; }}
  .section-sub {{ color: #8b90a4; font-size: 13px; margin-top: 6px; }}
  .table-wrap {{ padding: 12px 32px 8px; overflow-x: auto; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  caption {{ color: #e2e5f0; text-align: left; padding-bottom: 8px; font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>Revenue Revision Benchmark Beaters</h1>
  <div class="meta">Generated {date_str} &middot; Stocks at fresh 6-month relative highs with confirming analyst revenue estimate revisions &middot; Top 40 per section</div>
</header>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    main()
